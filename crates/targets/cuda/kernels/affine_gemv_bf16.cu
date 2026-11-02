// SPDX-License-Identifier: Apache-2.0
//
// MLX affine 4-bit GEMV for the H100: c[n] = Σ_k a[k] · dequant(W)[n, k], one activation row
// `a [1, k]` bf16 against an MLX affine weight of `n` rows of `k` codes, accumulated in f32 and
// rounded once to bf16. `dequant(W)` is `affine_gemm_bf16.cu`'s: `scale[g, n] · q[n, j] +
// bias[g, n]`, `g = j / 64`, rounded once to bf16, the host dequant's bits. It is the `m = 1`
// case of `affine_gemm_bf16.cu`: a decode step's every affine projection and its lm_head.
//
// `gemv_bf16.cu`'s kernel with a dequant stage in front of its unchanged multiply (its structure
// is ThunderKittens' GEMM at 64×64, from TK's `kernels/gemm/bf16_h100` at 67845f5f, MIT, see
// `thunderkittens/LICENSE`). A producer warp loads, per 64-element K step, into a ring of `STAGES`
// stages, `tma::expect_bytes` before every load:
//   - `a`: the twin's swizzled `st_bf<64, 64>`, whose 63 rows past the array's edge TMA reads as
//     zeros;
//   - the step's packed codes, a plain `st_uint8<64, 32>` (one 64-code group a row), zero past
//     row `n`;
//   - its 64 scales and 64 biases, one `sv_bf<64>` each (row `g` of the group-major arrays),
//     zero past column `n`.
// The consumer warpgroup's 128 threads, two per B row (32 outputs each), dequantize them into one
// swizzled `st_bf<64, 64>` through `scratchy_ops.cuh`'s `affine_b4_dequant` gap; then
// `warpgroup::mma_fence()` (TK's `wgmma.fence` + `fence.proxy.async.shared::cta`,
// `ops/group/mma/warpgroup.cuh:76`), `warpgroup::sync`, and the twin's `warpgroup::mma_ABt` into
// `rt_fl<16, 64>`, waited per stage. The dequant tile is single-buffered: the warpgroup syncs at
// the top of each K step, after every warp's wait, before the tile is overwritten. A B row past
// `n` dequantizes from zero codes, scale and bias to +0, as the twin's swizzled tile reads zeros
// there.
//
// No K tail: `k % 64 == 0` always for an affine weight (`op_abi::CUDA_AFFINE_FORMS`), so the
// twin's plain tail tiles are not instantiated. The `c` store is the twin's: the accumulator goes
// to a plain shared tile (`warpgroup::store`, the one RNE rounding to bf16), then row 0's
// `min(64, n − c0)` columns are copied through TK's accessors. Nothing else of `c` is written.
//
// So for the same `a`, `c` is `gemv_bf16.cu`'s bits on the host-dequantized weight: the same `a`
// tile, the same B values in the same swizzled layout, the same WGMMA in the same K order, the
// same one rounding. No atomics, so a repeat run gives the same bits. Memory traffic is 4.5 bits a
// weight (codes, scale and bias), not 16.
//
// Shared memory: `STAGES` × (`a` 8 KiB + codes 2 KiB), the scales' and the biases' rings (128 bytes
// a stage, each ring in its own 1 KiB allocator slot), the dequant tile and the plain `c` tile
// (8 KiB each), plus 1,024 bytes of alignment slack = 60,416 bytes of dynamic shared memory, set
// once by
// `scratchy_cuda_affine_gemv_bf16_abt_prepare` (in `PREPARE`). The semaphores are static.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// The one affine group CUDA takes (`op_abi::CUDA_AFFINE_FORMS`): one group is one K tile.
constexpr uint32_t AFFINE_GROUP = uint32_t(scratchy::ops::affine_b4::GROUP);
// `n` is a multiple of it: the scales' and biases' rows are `2n` bytes, a multiple of 16, as their
// tensor maps need.
constexpr uint32_t N_MULTIPLE = 8;
// Output columns per CTA (rows of `W`), and K per tile.
constexpr int TILE = 64;
static_assert(uint32_t(TILE) == AFFINE_GROUP, "one group is one K tile");
constexpr int STAGES = 4;
// One consumer warpgroup, then one producer warpgroup whose first warp issues every load.
constexpr int CONSUMER_WARPGROUP = 0;
constexpr int PRODUCER_WARPGROUP = 1;
constexpr int NUM_THREADS = 2 * kittens::WARPGROUP_THREADS;
// The consumer warpgroup's named barrier (0 is `__syncthreads`).
constexpr int CONSUMER_BARRIER = 1;
// The dequant's split: two threads a B row, 32 outputs each.
constexpr int THREADS_PER_ROW = kittens::WARPGROUP_THREADS / TILE;
constexpr int DEQUANT_COLS = TILE / THREADS_PER_ROW;

using tile = st_bf<TILE, TILE>;
using plain_tile = st_bf<TILE, TILE, false>;
// One K step's codes: 64 rows of 32 bytes. Plain: only the consumers' accessors read it.
using packed_tile = st_uint8<TILE, TILE / 2, false>;
using scale_vec = sv_bf<TILE>;
// `a [1, k]`: the swizzled tensor map only (no K tail).
using operand_gl = gl<bf16, 1, 1, -1, -1, tile>;
// The codes `[n, k/2]` and the group-major scales and biases `[k/64, n]`.
using packed_gl = gl<uint8, 1, 1, -1, -1, packed_tile>;
using scale_gl = gl<bf16, 1, 1, -1, -1, scale_vec>;
// `c [1, n]`: no tensor map, written element by element.
using output_gl = gl<bf16, 1, 1, -1, -1>;

struct affine_gemv_globals {
    operand_gl a;
    packed_gl codes;
    scale_gl scales;
    scale_gl biases;
    output_gl c;
    // The columns of `c`; the K tiles, `k / 64`.
    int n;
    int k_tiles;
};

// The bytes one stage's four loads bring.
constexpr uint32_t STAGE_BYTES = size_bytes<tile, packed_tile, scale_vec, scale_vec>;
// Each `allocate` starts at a 1,024-byte boundary (`tma_swizzle_allocator`), so the two vector
// arrays take a whole slot each.
constexpr size_t VEC_SLOT = 1024;
static_assert(STAGES * sizeof(scale_vec) <= VEC_SLOT, "one slot holds a stage ring of vectors");
constexpr size_t SMEM_TILES = STAGES * (sizeof(tile) + sizeof(packed_tile)) + 2 * VEC_SLOT +
                              sizeof(tile) + sizeof(plain_tile);
static_assert(SMEM_TILES == 59392, "4 stages of a and codes, two vector slots, B and c");
// The tiles plus the slack `tma_swizzle_allocator` may spend aligning the first to 1024 bytes.
constexpr int AFFINE_GEMV_SMEM_BYTES = int(SMEM_TILES) + 1024;
static_assert(AFFINE_GEMV_SMEM_BYTES <= kittens::MAX_SHARED_MEMORY, "fits the H100's 227 KiB");

__global__ __launch_bounds__(NUM_THREADS, 1) void affine_gemv_bf16_ker(
    const __grid_constant__ affine_gemv_globals g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    tile(&a_smem)[STAGES] = al.allocate<tile, STAGES>();
    packed_tile(&codes_smem)[STAGES] = al.allocate<packed_tile, STAGES>();
    scale_vec(&scales_smem)[STAGES] = al.allocate<scale_vec, STAGES>();
    scale_vec(&biases_smem)[STAGES] = al.allocate<scale_vec, STAGES>();
    tile& b_smem = al.allocate<tile>();
    plain_tile& c_smem = al.allocate<plain_tile>();

    // This CTA's 64 output columns: rows `64 · col_tile ..` of `W`.
    const int col_tile = blockIdx.x;
    const int warpgroupid = threadIdx.x / kittens::WARPGROUP_THREADS;

    __shared__ kittens::semaphore arrived[STAGES], done[STAGES];
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(done[s], 1, 0);
        }
    }
    __syncthreads();

    if (warpgroupid == PRODUCER_WARPGROUP) {
        // The producer: one warp loads K tile `j` into stage `j % STAGES`, after the consumer
        // released that stage's previous tile.
        if (warpgroup::warpid() == 0) {
            for (int j = 0; j < g.k_tiles; j++) {
                const int stage = j % STAGES;
                if (j >= STAGES) {
                    wait(done[stage], ((j - STAGES) / STAGES) % 2);
                }
                coord<tile> at_a = {0, 0, 0, j};
                coord<packed_tile> at_codes = {0, 0, col_tile, j};
                // Row `j` of the group-major arrays, vector `col_tile` of it.
                coord<scale_vec> at_scales = {0, 0, j, col_tile};
                warp::tma::expect_bytes(arrived[stage], STAGE_BYTES);
                warp::tma::load_async(a_smem[stage], g.a, at_a, arrived[stage]);
                warp::tma::load_async(codes_smem[stage], g.codes, at_codes, arrived[stage]);
                warp::tma::load_async(scales_smem[stage], g.scales, at_scales, arrived[stage]);
                warp::tma::load_async(biases_smem[stage], g.biases, at_scales, arrived[stage]);
            }
        }
    } else if (warpgroupid == CONSUMER_WARPGROUP) {
        // The consumer: each warp holds 16 rows of the 64-row accumulator; only row 0 is real.
        rt_fl<16, TILE> acc;
        warp::zero(acc);
        const int t = warpgroup::laneid();
        for (int j = 0; j < g.k_tiles; j++) {
            const int stage = j % STAGES;
            wait(arrived[stage], (j / STAGES) % 2);
            // Every warp waited for its previous WGMMA, which read `b_smem`: past this barrier it
            // is free to overwrite.
            warpgroup::sync(CONSUMER_BARRIER);
            scratchy::ops::affine_b4_dequant_row<DEQUANT_COLS>(
                b_smem, codes_smem[stage], scales_smem[stage], biases_smem[stage],
                t / THREADS_PER_ROW, (t % THREADS_PER_ROW) * DEQUANT_COLS);
            // Every thread's dequant writes reach the async proxy before the WGMMA reads them.
            warpgroup::mma_fence();
            warpgroup::sync(CONSUMER_BARRIER);
            warpgroup::mma_ABt(acc, a_smem[stage], b_smem);
            // The WGMMA reads shared memory and writes `acc` asynchronously: wait for it before
            // handing the stage back to the producer, and before anything else reads `acc`.
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) {
                arrive(done[stage], 1);
            }
        }
        warpgroup::store(c_smem, acc);
        warpgroup::sync(CONSUMER_BARRIER);
        // Row 0's real columns, `c0 .. c0 + valid`, one per thread, through TK's accessors.
        const int c0 = col_tile * TILE;
        const int valid = min(TILE, g.n - c0);
        if (t < valid) {
            g.c[{0, 0, 0, c0 + t}] = c_smem[{0, t}];
        }
    }
}

}  // namespace

// The kernel's one-time setup: allow it `AFFINE_GEMV_SMEM_BYTES` of dynamic shared memory on the
// current device (the default limit is 48 KB). Run once per device before any launch
// (`kernels::prepare_all`, called by `CudaDevice::open`), never per launch.
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_affine_gemv_bf16_abt_prepare(void) {
    return static_cast<int>(cudaFuncSetAttribute(
        affine_gemv_bf16_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, AFFINE_GEMV_SMEM_BYTES));
}

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught (e.g. tensor-map encode); any other value
// = the cudaError_t cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the
// runtime's last-error state.
//
// `n` must be a non-zero multiple of 8, `k` a non-zero multiple of 64, both at most INT_MAX, and
// `group` 64; all are checked before any pointer is touched. `a` and `c` are 16-byte-aligned
// device pointers to `k` and `n` bf16; `codes`, `scales` and `biases` are the three regions of one
// `step::affine_weight(n, k, 64)` buffer: `n·k/2` bytes, then `k/64 · n` bf16 each, group-major,
// each 16-byte aligned. `c` overlaps no input. `scratchy_cuda_affine_gemv_bf16_abt_prepare` must
// have succeeded on the current device. The launch is asynchronous on `stream`.
extern "C" int scratchy_cuda_affine_gemv_bf16_abt(const void* a, const void* codes,
                                                  const void* scales, const void* biases, void* c,
                                                  uint32_t n, uint32_t k, uint32_t group,
                                                  void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || n % N_MULTIPLE != 0 || n > uint32_t(INT_MAX) || k == 0 ||
        k % AFFINE_GROUP != 0 || k > uint32_t(INT_MAX) || group != AFFINE_GROUP) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        bf16* ap = static_cast<bf16*>(const_cast<void*>(a));
        uint8* qp = static_cast<uint8*>(const_cast<void*>(codes));
        bf16* sp = static_cast<bf16*>(const_cast<void*>(scales));
        bf16* zp = static_cast<bf16*>(const_cast<void*>(biases));
        bf16* cp = static_cast<bf16*>(c);
        // Building each view encodes its TMA tensor map into host memory; nothing is allocated on
        // the device. The row strides are `2k`, `k/2` and `2n` bytes, each a multiple of 16.
        operand_gl ag{ap, nullptr, nullptr, size_t(1), size_t(k)};
        packed_gl qg{qp, nullptr, nullptr, size_t(n), size_t(k / 2)};
        scale_gl sg{sp, nullptr, nullptr, size_t(k / AFFINE_GROUP), size_t(n)};
        scale_gl zg{zp, nullptr, nullptr, size_t(k / AFFINE_GROUP), size_t(n)};
        output_gl cg{cp, nullptr, nullptr, size_t(1), size_t(n)};
        affine_gemv_globals G{ag, qg, sg, zg, cg, int(n), int(k / AFFINE_GROUP)};
        void* args[] = {&G};
        const dim3 grid((n + uint32_t(TILE) - 1) / uint32_t(TILE)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(affine_gemv_bf16_ker, grid, block, args,
                                                 AFFINE_GEMV_SMEM_BYTES,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
