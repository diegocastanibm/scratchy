// SPDX-License-Identifier: Apache-2.0
//
// MLX affine 4-bit GEMM for the H100: C = A · dequant(W)ᵀ, with A [M, K] and C [M, N] row-major
// bf16 and W an MLX affine weight of N rows of K codes, accumulated in f32. `dequant(W)[n, j] =
// scale[g, n] · q[n, j] + bias[g, n]` with `g = j / 64`, rounded once to bf16: the weight
// `scratchy_quantizations::affine::affine_dequant_b4_to_dtype` builds on the host, bit for bit.
// `q` is the `[N, K/2]` byte view of the checkpoint's packed u32 codes (code `2j` the low nibble
// of byte `j`); `scale` and `bias` are bf16, group-major `[K/64, N]` (`step::affine_weight`'s
// regions, transposed at load by `weights::group_major`).
//
// The bf16 twin's kernel with a dequant stage in front of its unchanged multiply
// (`gemm_bf16.cu`'s `matmul_template`, itself adapted from ThunderKittens
// `kernels/gemm/bf16_h100` at 67845f5f, MIT, see `thunderkittens/LICENSE`). TK at the pin has no
// int4 or weight-only quantized GEMM, so only the B operand's source changes:
//   - the producer's 4-stage ring loads, per 64-element K step, the twin's two swizzled
//     `st_bf<64, 64>` A tiles, then the step's packed codes as one plain `st_uint8<64·N_BLOCK, 32>`
//     (32 bytes = one 64-code group a row) and its `64·N_BLOCK` scales and biases, one
//     `sv_bf<64·N_BLOCK>` each (row `g` of the group-major arrays);
//   - the 8 consumer warps (256 threads) dequantize them into one swizzled `st_bf<64·N_BLOCK, 64>`,
//     the twin's `b_stack_tile`, through `scratchy_ops.cuh`'s `affine_b4_dequant` gap (TK's
//     element accessor applies the swizzle). At 256 columns each thread owns one B row (its 32
//     bytes, one scale, one bias, 64 outputs); at 64 columns 4 threads share a row (16 each);
//   - `warpgroup::mma_fence()` (TK's `wgmma.fence` + `fence.proxy.async.shared::cta`,
//     `ops/group/mma/warpgroup.cuh:76`) makes those generic-proxy writes visible to WGMMA, and a
//     named barrier over both consumer warpgroups comes before either multiplies;
//   - then the twin's `warpgroup::mma_ABt(accum, a[groupid], b_stack_tile)`, its
//     `mma_async_wait`, and the stage's release.
// The dequant tile is single-buffered, in `lcf`'s scratch block: the same barrier at the top of
// each K step holds every consumer until both warpgroups' WGMMAs of the previous step have read it.
//
// So for the same activations, `C` is `gemm_bf16.cu`'s bits on the host-dequantized weight: the
// same A tiles, the same B values in the same swizzled layout, the same WGMMA calls in the same K
// order (64-element steps, ascending), the same one RNE rounding at the finish (`warpgroup::store`)
// and the same TMA store. No atomics and no split-K, so a repeat run gives the same bits.
//
// The same template is instantiated twice, as the twin is: at `N_BLOCK = 4`
// (`scratchy_cuda_affine_gemm_bf16_abt`, 256 output columns per tile, twin `gemm_bf16_abt`) and
// at `N_BLOCK = 1` (`scratchy_cuda_affine_gemm_col64_bf16_abt`, 64, twin `gemm_col64_bf16_abt`).
//
// Edges: `n` is a multiple of `64·N_BLOCK` and `k` of 64, so no TMA box crosses an edge and every
// bf16 tile stays swizzled (the TK-1 R1 rule). The packed row stride `k/2` bytes and the scales'
// row stride `2n` bytes are then multiples of 32 and 128, as their tensor maps need (16).
//
// Shared memory (dynamic, the twin's budget, set once by `_prepare`), at 256 columns (64): per
// stage the A tiles (16 KiB), the packed codes (8 KiB; 2 KiB) and the scales and biases (1 KiB;
// 256 bytes), 25,600 bytes (18,688), padded to 26 KiB (19 KiB) by `lcf`'s 1,024-byte stage
// alignment: 104 KiB (76 KiB) for 4 stages; the dequant tile 32 KiB (8 KiB); the finish tiles
// the twin's 64 KiB (16 KiB). 200 KiB at most, below the 227 KiB an H100 block may use, and
// `lcf`'s finish block (placed at the top of the budget) overlaps no stage.

#include "kittens.cuh"
#include "prototype.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// One thread block per SM of an H100 SXM, as the twin's grid.
constexpr int H100_SXM_SMS = 132;

// The one affine group CUDA takes (`op_abi::CUDA_AFFINE_FORMS`): 64 codes share a scale and a
// bias, so one group is one K step.
constexpr uint32_t AFFINE_GROUP = uint32_t(scratchy::ops::affine_b4::GROUP);
static_assert(AFFINE_GROUP == 64, "one group is one 64-element K step");

// The named barrier over the two consumer warpgroups, around the dequant (0 is `__syncthreads`;
// `lcf` takes 13 to 15, the twin's finish 4 and 5).
constexpr int DEQUANT_BARRIER = 1;

template <int M_BLOCK, int N_BLOCK>
struct affine_matmul_layout {
    using base_tile = st_bf<64, 64>;
    // One K step's codes: `64·N_BLOCK` rows of 32 bytes. Plain (not swizzled): only the
    // consumers' accessors read it, never WGMMA.
    using packed_tile = st_uint8<64 * N_BLOCK, 32, false>;
    using scale_vec = sv_bf<64 * N_BLOCK>;
    // The dequantized B: the twin's `b_stack_tile`, the `[N, K]` operand `mma_ABt` takes.
    using dequant_tile = st_bf<64 * N_BLOCK, 64>;
    using global_layout = gl<bf16, 1, 1, -1, -1, base_tile>;
    using packed_layout = gl<uint8, 1, 1, -1, -1, packed_tile>;
    using scale_layout = gl<bf16, 1, 1, -1, -1, scale_vec>;
    struct globals {
        global_layout A;
        packed_layout B;
        scale_layout scales, biases;
        global_layout C;
    };
    struct input_block {
        base_tile a[M_BLOCK];
        packed_tile b;
        scale_vec scales, biases;
    };
    struct scratch_block {
        dequant_tile b;
    };
    struct finish_block {
        base_tile c[M_BLOCK][N_BLOCK];
    };
    struct common_state {
        int2 coord;
    };
    struct consumer_state {
        rt_fl<16, N_BLOCK * base_tile::cols> accum;
    };
};

template <int _M_BLOCK = 2, int _N_BLOCK = 4, int _SUPER_M = 12>
struct affine_matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout = affine_matmul_layout<M_BLOCK, N_BLOCK>;
    // The finish block's N_BLOCK column-adjacent 64x64 tiles of one consumer, seen as one
    // [64, 64*N_BLOCK] tile for `warpgroup::store` (as the twin).
    using wide_tile = st_bf<64, 64 * N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS = M_BLOCK * 4, INPUT_PIPE_STAGES = 4,
                         PRODUCER_BARRIER_ARRIVALS = 1;
    using consumers = group<NUM_CONSUMER_WARPS>;
    // The dequant's split: the consumers' threads over the tile's `64·N_BLOCK` rows of 64.
    static constexpr int THREADS_PER_ROW = consumers::GROUP_THREADS / (64 * N_BLOCK);
    static constexpr int DEQUANT_COLS = 64 / THREADS_PER_ROW;
    static_assert(THREADS_PER_ROW * 64 * N_BLOCK == consumers::GROUP_THREADS,
                  "every consumer thread owns whole bytes of one B row");
    // One TMA transaction per stage covers the whole input block: `tma::expect` counts its size.
    static_assert(sizeof(typename layout::input_block) ==
                      M_BLOCK * sizeof(typename layout::base_tile) +
                          size_bytes<typename layout::packed_tile, typename layout::scale_vec,
                                     typename layout::scale_vec>,
                  "the input block is exactly what the producer loads");

    // ThunderKittens template functions: `common_setup` is the twin's.
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows() / (M_BLOCK * 64),
            Cblocks = args.globals.C.cols() / (N_BLOCK * 64);
        int super_rows = (Rblocks / SUPER_M) * SUPER_M, final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M * Cblocks;
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
        if (task_id < super_rows * Cblocks)
            args.common.coord = {SUPER_M * (task_id / super_repeat) + task_id % SUPER_M,
                                 (task_id % super_repeat) / SUPER_M};
        else if (task_id < Rblocks * Cblocks) {
            int remainder_id = task_id - super_rows * Cblocks;
            args.common.coord = {super_rows + (remainder_id % final_rows),
                                 remainder_id / final_rows};
        } else {  // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols() / 64;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS / 4
                     ? 0
                     : warpgroup::groupid();  // producer sets as 0
        args.common.coord = {args.common.coord.x * M_BLOCK + id, args.common.coord.y * N_BLOCK};
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();  // decrease registers for producers
        }
        __device__ static void load(producer_load_args<layout> args) {
            // One thread of the producer warpgroup sets the byte count and issues every TMA load;
            // the `inputs_arrived` semaphore (PRODUCER_BARRIER_ARRIVALS = 1) completes when they
            // land.
            if (warpgroup::laneid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < M_BLOCK; i++)
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x + i, args.iter}, args.inputs_arrived);
                // The codes are `[N, K/2]`: tile row = the `64·N_BLOCK`-row column block, tile
                // column = K step. The scales and biases are `[K/64, N]`: row = K step (one group),
                // vector = the column block.
                const int col_block = args.common.coord.y / N_BLOCK;
                tma::load_async(args.input.b, args.globals.B, {col_block, args.iter},
                                args.inputs_arrived);
                tma::load_async(args.input.scales, args.globals.scales, {args.iter, col_block},
                                args.inputs_arrived);
                tma::load_async(args.input.biases, args.globals.biases, {args.iter, col_block},
                                args.inputs_arrived);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();  // increase registers for consumers
            kittens::warp::zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            // Both warpgroups waited for their previous WGMMA, which read the dequant tile: past
            // this barrier it is free to overwrite.
            consumers::sync(DEQUANT_BARRIER);
            const int t = consumers::laneid();
            scratchy::ops::affine_b4_dequant_row<DEQUANT_COLS>(
                args.scratch.b, args.input.b, args.input.scales, args.input.biases,
                t / THREADS_PER_ROW, (t % THREADS_PER_ROW) * DEQUANT_COLS);
            // Every thread's dequant writes reach the async proxy before any WGMMA reads them.
            warpgroup::mma_fence();
            consumers::sync(DEQUANT_BARRIER);
            warpgroup::mma_ABt(
                args.state.accum,                     // dest registers
                args.input.a[warpgroup::groupid()],  // A: this consumer's 64 rows x 64 K
                args.scratch.b                        // B: 64·N_BLOCK N rows x 64 K, dequantized
            );
            // The WGMMA reads shared memory asynchronously: wait for it before handing the input
            // block back to the producer and the dequant tile to the next K step.
            warpgroup::mma_async_wait();
            if (warp::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(reinterpret_cast<wide_tile&>(args.finish.c[warpgroup::groupid()]),
                             args.state.accum);
            warpgroup::sync(warpgroup::groupid() + 4);
            if (warpgroup::laneid() == 0)
                for (int i = 0; i < N_BLOCK; i++) {
                    tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                     {args.common.coord.x, args.common.coord.y + i});
                    tma::store_async_read_wait();  // wait that store is finished before reusing
                                                   // finish memory
                }
            kittens::warp::zero(args.state.accum);
            if (warp::laneid() == 0) arrive(args.finish_finished);
        }
    };
};

using affine_gemm_bf16_abt_mmt = affine_matmul_template<2, 4, 8>;
// One 64-column B block: `rt_fl<16, 64>` accumulators, `a[2]` and one 64-row packed tile.
using affine_gemm_col64_bf16_abt_mmt = affine_matmul_template<2, 1, 8>;

// The dynamic shared memory every launch asks for, and the attribute `_prepare` raises the
// kernel's limit to: the twin's, the whole H100 budget minus the 1024 bytes `lcf` reserves.
constexpr int AFFINE_GEMM_SMEM_BYTES = kittens::MAX_SHARED_MEMORY - 1024;

// The launch of `mmt`'s instance. The shape rule is the twin's tile's: `m`, `n` and `k` non-zero
// multiples of `M_BLOCK·64`, `N_BLOCK·64` and 64, none above INT_MAX; and `group` 64.
template <typename mmt>
int launch_affine(const void* a, const void* codes, const void* scales, const void* biases,
                  void* c, uint32_t m, uint32_t n, uint32_t k, uint32_t group, void* stream) {
    constexpr uint32_t M_TILE = mmt::M_BLOCK * 64, N_TILE = mmt::N_BLOCK * 64, K_TILE = 64;
    constexpr uint32_t DIM_MAX = static_cast<uint32_t>(INT_MAX);
    if (m == 0 || n == 0 || k == 0 || m % M_TILE != 0 || n % N_TILE != 0 || k % K_TILE != 0 ||
        m > DIM_MAX || n > DIM_MAX || k > DIM_MAX || group != AFFINE_GROUP) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        using global_layout = typename mmt::layout::global_layout;
        using packed_layout = typename mmt::layout::packed_layout;
        using scale_layout = typename mmt::layout::scale_layout;
        using globals = typename mmt::layout::globals;
        // Building each `gl` encodes its TMA tensor map into host memory; nothing is allocated on
        // the device. The codes' rows are `k/2` bytes, the scales' and biases' `2n`.
        global_layout Ag{static_cast<bf16*>(const_cast<void*>(a)), nullptr, nullptr, size_t(m),
                         size_t(k)};
        packed_layout Bg{static_cast<uint8*>(const_cast<void*>(codes)), nullptr, nullptr,
                         size_t(n), size_t(k / 2)};
        scale_layout Sg{static_cast<bf16*>(const_cast<void*>(scales)), nullptr, nullptr,
                        size_t(k / AFFINE_GROUP), size_t(n)};
        scale_layout Zg{static_cast<bf16*>(const_cast<void*>(biases)), nullptr, nullptr,
                        size_t(k / AFFINE_GROUP), size_t(n)};
        global_layout Cg{static_cast<bf16*>(c), nullptr, nullptr, size_t(m), size_t(n)};
        globals G{Ag, Bg, Sg, Zg, Cg};

        // Three warpgroups: two consumers and one producer, as the twin. The kernel's one
        // parameter is the `globals` struct by value (`__grid_constant__`).
        void* args[] = {&G};
        const dim3 grid(H100_SXM_SMS), block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
        return static_cast<int>(cudaLaunchKernel(kittens::prototype::lcf::kernel<mmt>, grid,
                                                 block, args, AFFINE_GEMM_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

}  // namespace

// The kernel's one-time setup: allow it `AFFINE_GEMM_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_affine_gemm_bf16_abt_prepare(void) {
    return static_cast<int>(
        cudaFuncSetAttribute(kittens::prototype::lcf::kernel<affine_gemm_bf16_abt_mmt>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, AFFINE_GEMM_SMEM_BYTES));
}

// `scratchy_cuda_affine_gemm_bf16_abt_prepare`'s, for the 64-column instance.
extern "C" int scratchy_cuda_affine_gemm_col64_bf16_abt_prepare(void) {
    return static_cast<int>(
        cudaFuncSetAttribute(kittens::prototype::lcf::kernel<affine_gemm_col64_bf16_abt_mmt>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, AFFINE_GEMM_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `m`, `n` and `k` must be non-zero multiples of 128, 256 and 64 (`step::GEMM_BF16_TILE`, the
// twin's), none above INT_MAX, and `group` 64; all are checked before any pointer is touched.
// `a` and `c` are 16-byte-aligned device pointers to `m·k` and `m·n` bf16; `codes`, `scales` and
// `biases` are the three regions of one `step::affine_weight(n, k, 64)` buffer: `n·k/2` bytes,
// then `k/64 · n` bf16 each, group-major, each 16-byte aligned. `c` overlaps no input.
// `scratchy_cuda_affine_gemm_bf16_abt_prepare` must have succeeded on the current device. The
// launch is asynchronous on `stream`; the caller owns it and synchronizes.
extern "C" int scratchy_cuda_affine_gemm_bf16_abt(const void* a, const void* codes,
                                                  const void* scales, const void* biases, void* c,
                                                  uint32_t m, uint32_t n, uint32_t k,
                                                  uint32_t group,
                                                  void* stream /* cudaStream_t; null = default */) {
    return launch_affine<affine_gemm_bf16_abt_mmt>(a, codes, scales, biases, c, m, n, k, group,
                                                   stream);
}

// As `scratchy_cuda_affine_gemm_bf16_abt`, on 64-column tiles: `m`, `n` and `k` must be non-zero
// multiples of 128, 64 and 64 (`step::GEMM_COL64_TILE`), none above INT_MAX, and `group` 64,
// checked before any pointer is touched. `scratchy_cuda_affine_gemm_col64_bf16_abt_prepare` must
// have succeeded on the current device.
extern "C" int scratchy_cuda_affine_gemm_col64_bf16_abt(
    const void* a, const void* codes, const void* scales, const void* biases, void* c, uint32_t m,
    uint32_t n, uint32_t k, uint32_t group, void* stream /* cudaStream_t; null = default */) {
    return launch_affine<affine_gemm_col64_bf16_abt_mmt>(a, codes, scales, biases, c, m, n, k,
                                                         group, stream);
}
