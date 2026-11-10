// SPDX-License-Identifier: Apache-2.0
//
// Test support only, for `tests/gemv_fp8_ring_probe.rs` (FP8 S8 W2c, /tmp/d-w2c.md §3): no
// `CudaStep` and no lowering use it. It asks whether the fp8 decode GEMV is bound by the rate of
// its TMA operations rather than by its arithmetic: the GEMV's grid (8 warps, 2 CTAs per SM, a
// ring of stages per warp or per 4 warps) streams an e4m3 weight `B [n, k]` through shared memory
// and does no dot product. Lane `l` folds byte `l` of each row's every 256 columns into an f32
// (TK's element convertor, then `base_ops::sum`), and every thread stores its sum to `out`, so no
// load is dead. With every weight byte 1.0 the folds count the bytes read, exactly.
//
// The variants differ only in the bytes one TMA operation carries:
//   0 `vec256_r3`: one `sv_fp8e4m3<256>` (256 B) per row per chunk, 3 rows per warp, the GEMV's
//     `rows_per_warp` at the 27B's gate+up and down, its ring of 4 stages per warp;
//   1 `vec256`: the same at 4 rows per warp (every variant below has 4);
//   2 `pair512`: one plain `st_fp8e4m3<2, 256>` (512 B) per 2 rows per chunk. TK splits a vector
//     longer than 256 elements into several TMA operations (`find_vector_divider`), so 512 bytes in
//     one operation is a 2-row tile;
//   3 `tile4k`: one `st_fp8e4m3<16, 256>` (4 KB) per 4 warps' 16 rows per chunk, 4 stages;
//   4 `page16k`: one `st_fp8e4m3<16, 1024>` (16 KB) per 4 warps per 1,024 columns, 3 stages, so
//     two CTAs still fit an SM.
// A ring shared by 4 warps has an `empty` semaphore per stage: each warp arrives once it has read
// the stage, and the group's first warp waits for all 4 before it refills it.

#include "kittens.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

constexpr int WARPS = 8;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int CHUNK = 256;
// `gemv_fp8.cu`'s `GEMV_SMEM_BYTES`: every variant takes at least this much dynamic shared memory,
// so the SM holds the GEMV's 2 CTAs and no more, whatever its ring needs.
constexpr int GEMV_SMEM_BYTES = 68608;

using vec_box = sv_fp8e4m3<CHUNK>;
using pair_box = st_fp8e4m3<2, CHUNK, false>;
using tile_box = st_fp8e4m3<16, CHUNK>;
using page_box = st_fp8e4m3<16, 4 * CHUNK>;
static_assert(sizeof(vec_box) == 256 && sizeof(pair_box) == 512, "256 and 512 bytes an operation");
static_assert(sizeof(tile_box) == 4096 && sizeof(page_box) == 16384, "4 and 16 KB an operation");
// `B [n, k]` e4m3 with every variant's tensor map; `out [ctas, 256]` f32, no map.
using weight_gl = gl<fp8e4m3, 1, 1, -1, -1, vec_box, pair_box, tile_box, page_box>;
using out_gl = gl<float, 1, 1, -1, -1>;

struct probe_globals {
    weight_gl b;
    out_gl out;
    int n;
    int k;
};

// A variant: its box, the rows one box holds, the warps that share a ring, each warp's rows, the
// ring's stages, and the columns one box covers.
template <typename BOX, int BOX_ROWS, int GROUP, int ROWS, int STAGES, int K_STEP>
struct variant {
    using box = BOX;
    static constexpr int box_rows = BOX_ROWS;
    static constexpr int group = GROUP;
    static constexpr int rows = ROWS;
    static constexpr int stages = STAGES;
    static constexpr int k_step = K_STEP;
    // The boxes of one stage, and the rings of a CTA.
    static constexpr int ops = GROUP * ROWS / BOX_ROWS;
    static constexpr int rings = WARPS / GROUP;
    static constexpr int ring_bytes = rings * STAGES * ops * int(sizeof(BOX)) + 1024;
    static constexpr int smem = ring_bytes > GEMV_SMEM_BYTES ? ring_bytes : GEMV_SMEM_BYTES;
    static_assert(GROUP * ROWS % BOX_ROWS == 0 && WARPS % GROUP == 0, "whole boxes, whole groups");
    static_assert(2 * smem <= kittens::MAX_SHARED_MEMORY, "two CTAs per SM, as the GEMV");
};

using vec256_r3 = variant<vec_box, 1, 1, 3, 4, CHUNK>;
using vec256 = variant<vec_box, 1, 1, 4, 4, CHUNK>;
using pair512 = variant<pair_box, 2, 1, 4, 4, CHUNK>;
using tile4k = variant<tile_box, 16, 4, 4, 4, CHUNK>;
using page16k = variant<page_box, 16, 4, 4, 3, 4 * CHUNK>;

template <typename V>
__global__ __launch_bounds__(NUM_THREADS, 2) void tma_ring_probe_ker(
    const __grid_constant__ probe_globals g) {
    using box = typename V::box;
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    box(&ring)[V::rings][V::stages][V::ops] = al.allocate<box, V::rings, V::stages, V::ops>();
    __shared__ kittens::semaphore full[V::rings][V::stages];
    __shared__ kittens::semaphore empty[V::rings][V::stages];
    const int w = kittens::warpid();
    const int lane = warp::laneid();
    const int grp = w / V::group;
    const int wg = w % V::group;
    const bool producer = wg == 0;
    // The ring's first row, and this warp's rows (a vector variant's last block may hold fewer).
    const int row0 = (blockIdx.x * V::rings + grp) * V::group * V::rows;
    const int rows = max(0, min(V::rows, g.n - (row0 + wg * V::rows)));
    const int chunks = g.k / V::k_step;
    if (producer && lane == 0) {
        for (int s = 0; s < V::stages; s++) {
            init_semaphore(full[grp][s], 0, 1);
            init_semaphore(empty[grp][s], V::group, 0);
        }
    }
    // The semaphores are initialised before any thread waits on them.
    __syncthreads();
    // Chunk `j` of the ring's rows into stage `j % stages`.
    const auto issue = [&](int j) {
        const int s = j % V::stages;
        if constexpr (V::box_rows == 1) {
            warp::tma::expect_bytes(full[grp][s], uint32_t(rows * sizeof(box)));
#pragma unroll
            for (int r = 0; r < V::rows; r++) {
                if (r < rows) {
                    warp::tma::load_async(ring[grp][s][r], g.b, {0, 0, row0 + r, j}, full[grp][s]);
                }
            }
        } else {
            warp::tma::expect_bytes(full[grp][s], uint32_t(V::ops * sizeof(box)));
#pragma unroll
            for (int o = 0; o < V::ops; o++) {
                warp::tma::load_async(ring[grp][s][o], g.b, {0, 0, row0 / V::box_rows + o, j},
                                      full[grp][s]);
            }
        }
    };
    float acc = 0.0f;
    if (rows > 0) {
        if (producer) {
            for (int j = 0; j < min(V::stages, chunks); j++) {
                issue(j);
            }
        }
        for (int j = 0; j < chunks; j++) {
            const int s = j % V::stages;
            wait(full[grp][s], (j / V::stages) % 2);
#pragma unroll
            for (int r = 0; r < V::rows; r++) {
                if (r < rows) {
                    const int rr = wg * V::rows + r;
                    const box& bx = ring[grp][s][rr / V::box_rows];
#pragma unroll
                    for (int q = 0; q < V::k_step / CHUNK; q++) {
                        fp8e4m3 v;
                        if constexpr (V::box_rows == 1) {
                            v = bx[q * CHUNK + lane];
                        } else {
                            v = bx[{rr % V::box_rows, q * CHUNK + lane}];
                        }
                        acc = base_ops::sum::op<float>(
                            acc, base_types::convertor<float, fp8e4m3>::convert(v));
                    }
                }
            }
            // Every lane has read the stage before its warp says so, or refills it.
            __syncwarp();
            if constexpr (V::group > 1) {
                if (lane == 0) {
                    arrive(empty[grp][s]);
                }
            }
            if (producer && j + V::stages < chunks) {
                if constexpr (V::group > 1) {
                    wait(empty[grp][s], (j / V::stages) % 2);
                }
                issue(j + V::stages);
            }
        }
    }
    g.out[{0, 0, int(blockIdx.x), int(threadIdx.x)}] = acc;
}

// Variant `V` over `G`'s weight: the shape rule, then its shared memory and the launch.
template <typename V>
int launch(probe_globals G, uint32_t n, uint32_t k, void* stream) {
    if (k % uint32_t(V::k_step) != 0 || (V::box_rows > 1 && n % uint32_t(WARPS * V::rows) != 0)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    const cudaError_t set = cudaFuncSetAttribute(
        tma_ring_probe_ker<V>, cudaFuncAttributeMaxDynamicSharedMemorySize, V::smem);
    if (set != cudaSuccess) {
        return static_cast<int>(set);
    }
    void* args[] = {&G};
    const uint32_t block_rows = uint32_t(WARPS * V::rows);
    const dim3 grid((n + block_rows - 1) / block_rows), block(NUM_THREADS);
    return static_cast<int>(cudaLaunchKernel(tma_ring_probe_ker<V>, grid, block, args, V::smem,
                                             static_cast<cudaStream_t>(stream)));
}

}  // namespace

// Test support for `tests/gemv_fp8_ring_probe.rs` only: stream `b [n, k]` e4m3 through variant
// `variant` (0 `vec256_r3`, 1 `vec256`, 2 `pair512`, 3 `tile4k`, 4 `page16k`) and store each
// thread's fold into `out` (`⌈n / (8 · rows)⌉ · 256` f32, `rows` 3 for variant 0 and 4 for the
// others; `16 · n + 256` f32 always suffice). `n` and `k` non-zero and at most INT_MAX / 2, `k` a
// multiple of the variant's columns a box (256, or 1,024 for variant 4), `n` a multiple of 32 for
// variants 2 to 4; anything else returns SCRATCHY_CUDA_ERR_SHAPE before any pointer is touched.
// `b` 16-byte aligned. It sets its instance's shared-memory attribute on every call (no worker
// launches it). 0 = launched; asynchronous on `stream`.
extern "C" int scratchy_cuda_tma_ring_probe(const void* b, void* out, uint32_t n, uint32_t k,
                                            uint32_t variant,
                                            void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || k == 0 || n > uint32_t(INT_MAX / 2) || k > uint32_t(INT_MAX / 2) || variant > 4) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t rows = variant == 0 ? 3 : 4;
        const uint32_t ctas = (n + uint32_t(WARPS) * rows - 1) / (uint32_t(WARPS) * rows);
        weight_gl bg{static_cast<fp8e4m3*>(const_cast<void*>(b)), nullptr, nullptr, size_t(n),
                     size_t(k)};
        out_gl og{static_cast<float*>(out), nullptr, nullptr, size_t(ctas), size_t(NUM_THREADS)};
        const probe_globals G{bg, og, int(n), int(k)};
        switch (variant) {
            case 0:
                return launch<vec256_r3>(G, n, k, stream);
            case 1:
                return launch<vec256>(G, n, k, stream);
            case 2:
                return launch<pair512>(G, n, k, stream);
            case 3:
                return launch<tile4k>(G, n, k, stream);
            default:
                return launch<page16k>(G, n, k, stream);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
