// SPDX-License-Identifier: Apache-2.0
//
// MLX affine 4-bit token-embedding gather for the H100: out[r, :] = dequant(table[tokens[r], :])
// for every one of the bucket's `rows`, with tokens [rows] u32, out [rows, hidden] bf16 and the
// table an MLX affine weight of `vocab` rows of `hidden` codes. `dequant` is
// `affine_gemm_bf16.cu`'s, row by row: `scale[r, g] · q[r, j] + bias[r, g]`, `g = j / 64`,
// rounded once to bf16, the host dequant's bits. The table's scales and biases stay row-major
// `[vocab, hidden/64]` (`weights::ScaleLayout::RowMajor`): one row's are contiguous.
//
// `embed_bf16.cu`'s bulk-copy gather with a dequant stage in the middle (PLAN-tk100 §2 S3): TK
// has no gather, so a `gl<int>` index read plus TK's descriptor-free bulk copies is its TK form.
// One warp per output row, 4 warps per CTA. Every lane reads the row's id from the `gl<int>` token
// view (`gl::operator[]`; a u32 above 2³¹ reads as negative). For an id in `[0, vocab)`, per chunk
// of up to 4096 elements, lane 0 bulk-loads the row's packed codes into the warp's
// `sv_uint8<2048>` (`tma::load_async`, descriptor-free, `tma::expect_bytes` on the warp's
// semaphore), and the lanes copy its scales and biases (up to 64 each) into an `sv_bf<64>` each
// through TK's accessors (`gl::operator[]`, `sv::operator[]`): a row's scales are `hidden/32`
// bytes, 8 at a `hidden` of 256, and a bulk copy moves only whole 16-byte units from 16-byte
// boundaries. The warp waits, dequantizes them into its `sv_bf<4096>` through `scratchy_ops.cuh`'s
// `affine_b4_dequant` gap (128 outputs a lane), and lane 0 bulk-stores the chunk to the output
// row (`tma::store_async`, whose async-proxy fence orders the lanes' writes before the store; the
// warp syncs first). The next chunk's loads wait
// until that store has read the buffer (`tma::store_async_read_wait`). GLM's 4096-element rows
// are one chunk.
//
// Every row is written, padding rows included. An id outside `[0, vocab)` writes a +0.0 row
// (`warp::zero` on the buffer, then the bulk store) and reads nothing from the table, as
// `embed_bf16.cu`'s: the worker refuses such an id before any copy. A bulk store writes exactly
// `hidden · 2` bytes, so the output's tail past `rows · hidden` is never written.
//
// Not tile-shaped: one warp-vector dequant per row through shared memory. Its TK form costs a
// round trip through shared memory a row: 4 warps × (8 KiB out + 2 KiB codes + 128 bytes each
// of scales and biases, each vector allocated to 128 bytes) = 41,984 bytes of static shared
// memory and a semaphore each, under the 48 KiB default, so no one-time setup.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// The launcher's return codes that are not a `cudaError_t` (which is non-negative).
// `src/kernels.rs` mirrors them.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// The one affine group CUDA takes (`op_abi::CUDA_AFFINE_FORMS`): `hidden` is a multiple of it, so
// every row's codes are whole 32-byte runs, each starting 16-byte aligned, as the bulk copy needs.
constexpr uint32_t AFFINE_GROUP = uint32_t(scratchy::ops::affine_b4::GROUP);
// The bf16 elements one chunk dequantizes at most: one 8 KiB buffer.
constexpr int CHUNK = 4096;
constexpr int CHUNK_GROUPS = CHUNK / int(AFFINE_GROUP);
constexpr int WARPS = 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;

using token_gl = gl<int, 1, 1, 1, -1>;
using row_gl = gl<bf16, 1, 1, -1, -1>;
using packed_gl = gl<uint8, 1, 1, -1, -1>;

struct affine_embed_globals {
    token_gl tokens;
    // The table's `[vocab, hidden/2]` codes and its `[vocab, hidden/64]` scales and biases.
    packed_gl codes;
    row_gl scales;
    row_gl biases;
    row_gl out;
    int rows;
    int hidden;
    int vocab;
};

__global__ __launch_bounds__(NUM_THREADS, 1) void affine_embed_bf16_ker(
    const __grid_constant__ affine_embed_globals g) {
    __shared__ sv_bf<CHUNK> buf[WARPS];
    __shared__ sv_uint8<CHUNK / 2> codes[WARPS];
    __shared__ sv_bf<CHUNK_GROUPS> scales[WARPS];
    __shared__ sv_bf<CHUNK_GROUPS> biases[WARPS];
    __shared__ kittens::semaphore arrived[WARPS];

    const int w = kittens::warpid();
    const int row = blockIdx.x * WARPS + w;
    if (row >= g.rows) {
        return;
    }
    const int id = g.tokens[{0, 0, 0, row}];
    // `hidden` is at most INT_MAX, so the chunks are counted without forming `c0 + CHUNK`.
    const int chunks = (g.hidden - 1) / CHUNK + 1;
    if (id < 0 || id >= g.vocab) {
        warp::zero(buf[w]);
        __syncwarp();
        for (int k = 0; k < chunks; k++) {
            const int c0 = k * CHUNK;
            const uint32_t bytes = uint32_t(min(CHUNK, g.hidden - c0)) * sizeof(bf16);
            warp::tma::store_async(g.out[{0, 0, row, c0}], buf[w][0], bytes);
        }
        warp::tma::store_async_wait();
        return;
    }
    if (warp::laneid() == 0) {
        init_semaphore(arrived[w], 0, 1);
    }
    __syncwarp();
    for (int k = 0; k < chunks; k++) {
        const int c0 = k * CHUNK;
        const int len = min(CHUNK, g.hidden - c0);
        const int g0 = c0 / int(AFFINE_GROUP);
        const uint32_t code_bytes = uint32_t(len / 2);
        // The previous chunk's store must have read the buffer before this dequant overwrites it,
        // and every lane's dequant must have read the previous chunk's inputs.
        warp::tma::store_async_read_wait();
        __syncwarp();
        warp::tma::expect_bytes(arrived[w], code_bytes);
        warp::tma::load_async(codes[w][0], g.codes[{0, 0, id, c0 / 2}], code_bytes, arrived[w]);
        for (int i = warp::laneid(); i < len / int(AFFINE_GROUP); i += kittens::WARP_THREADS) {
            scales[w][i] = g.scales[{0, 0, id, g0 + i}];
            biases[w][i] = g.biases[{0, 0, id, g0 + i}];
        }
        wait(arrived[w], k % 2);
        __syncwarp();
        scratchy::ops::affine_b4_dequant_sv(buf[w], codes[w], scales[w], biases[w], len);
        __syncwarp();
        warp::tma::store_async(g.out[{0, 0, row, c0}], buf[w][0],
                               uint32_t(len) * uint32_t(sizeof(bf16)));
    }
    warp::tma::store_async_wait();
}

}  // namespace

// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = shape refused, nothing launched;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t
// cudaLaunchKernel returned for THIS launch. Nothing here reads or clears the runtime's last-error
// state.
//
// `rows`, `hidden` and `vocab` must be non-zero and at most INT_MAX, `hidden` a multiple of 64,
// `group` 64, and `rows · hidden` at most INT_MAX; all are checked before any pointer is touched.
// `tokens` must be a 4-byte-aligned device pointer to `rows` u32; `out` a 16-byte-aligned device
// pointer to `rows·hidden` bf16; `codes`, `scales` and `biases` the three regions of one
// `step::affine_weight(vocab, hidden, 64)` buffer: `vocab·hidden/2` bytes, then `vocab ·
// hidden/64` bf16 each, row-major, each 16-byte aligned. `out` overlaps no input. Static shared
// memory only, so no one-time setup. The launch is asynchronous on `stream`; the caller owns it and
// synchronizes.
extern "C" int scratchy_cuda_affine_embed_bf16(const void* tokens, const void* codes,
                                               const void* scales, const void* biases, void* out,
                                               uint32_t rows, uint32_t hidden, uint32_t vocab,
                                               uint32_t group,
                                               void* stream /* cudaStream_t; null = default */) {
    if (rows == 0 || hidden == 0 || vocab == 0 || hidden % AFFINE_GROUP != 0 ||
        group != AFFINE_GROUP || rows > uint32_t(INT_MAX) || hidden > uint32_t(INT_MAX) ||
        vocab > uint32_t(INT_MAX) || uint64_t(rows) * uint64_t(hidden) > uint64_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        int* tp = static_cast<int*>(const_cast<void*>(tokens));
        uint8* qp = static_cast<uint8*>(const_cast<void*>(codes));
        bf16* sp = static_cast<bf16*>(const_cast<void*>(scales));
        bf16* zp = static_cast<bf16*>(const_cast<void*>(biases));
        bf16* op = static_cast<bf16*>(out);
        // No view carries a TMA type, so building them encodes nothing: they are the pointers
        // and the dimensions the kernel indexes with.
        token_gl tg{tp, nullptr, nullptr, nullptr, size_t(rows)};
        packed_gl qg{qp, nullptr, nullptr, size_t(vocab), size_t(hidden / 2)};
        row_gl sg{sp, nullptr, nullptr, size_t(vocab), size_t(hidden / AFFINE_GROUP)};
        row_gl zg{zp, nullptr, nullptr, size_t(vocab), size_t(hidden / AFFINE_GROUP)};
        row_gl og{op, nullptr, nullptr, size_t(rows), size_t(hidden)};
        affine_embed_globals G{tg, qg, sg, zg, og, int(rows), int(hidden), int(vocab)};
        void* args[] = {&G};
        const dim3 grid((rows + uint32_t(WARPS) - 1) / uint32_t(WARPS)), block(NUM_THREADS);
        return static_cast<int>(cudaLaunchKernel(affine_embed_bf16_ker, grid, block, args, 0,
                                                 static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
