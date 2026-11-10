// SPDX-License-Identifier: Apache-2.0
//
// The decode megakernel's interpreter (FP8 S8 M0, /tmp/a-mk-plan.md §6): one persistent kernel
// that plays a per-SM instruction list from global memory. Modelled on HazyResearch's Megakernels
// (`include/`, MIT, 7309cec; credited in `kernels/PROVENANCE.md`), written again here against
// ThunderKittens only: no code is copied (theirs reads its ring through `reinterpret_cast` page
// views, volatile spins and `atomicAdd`, which the TK check refuses).
//
// THE MACHINE.
//   - Grid: one CTA per SM, `SMS` = 132 (the H100's), launched cooperatively so all 132 are
//     resident at once (a cross-SM wait on a CTA that never started would hang). The SM index is
//     `blockIdx.x`: no `%smid`.
//   - Two programs share the interpreter (`control`, `play`). The test program (`mk_test_ker`) is
//     20 warps: two LANES of 8 consumer warps (lane `L` is warps `8L .. 8L + 7`), each playing its
//     own list, warp 16 + L lane L's controller, warps 18 and 19 idle. The MLP program
//     (`mk_mlp_ker`, M0 Day 2) is one lane of 8 consumer warps and its controller (9 warps): its
//     norm holds an `rt_fl<16, 256>` (128 f32 a thread), more than a 20-warp CTA's 102 registers.
//     Either way an instruction is the work of one of today's 8-warp CTAs.
//   - The instruction list: `instructions [1, SMS · LANES, N, 32]` int: worker
//     `w = LANES · blockIdx.x + L` owns row `w`, its `N` instructions of 32 ints (128 bytes), built
//     on the host before the launch (M0: in the test; M1: emitted by `#[forward]`). Word 0 is the
//     opcode, words 1.. its arguments. Every worker runs the same `N`; a shorter list is padded
//     with `OP_NOOP`, so every loop below has the launch's trip count.
//   - The instruction ring: `STAGES` = 2 shared `sv<int, 32>` per lane. Lane L's controller
//     TMA-loads instruction `i` into stage `i % 2` on `arrived[L][i % 2]`, once the consumers have
//     finished instruction `i − 2` there (`finished[L][i % 2]`, 8 warp arrivals). So instruction
//     `i + 1` is in shared memory before instruction `i` ends.
//   - The pages: the dynamic shared memory is `PAGES` = 13 pages of 16 KiB. Lane L owns pages
//     `6L .. 6L + 5`; page 12 is the SM's (Day 2: the normed row, computed once per SM). An opcode
//     carves its own TK types over its lane's pages with a TK allocator (`pages_of<L>`), so a page
//     has a type per opcode and no page is ever reached through a cast.
//   - Cross-SM order: a global `counters [1, C]` u32, zeroed by the host. A producer publishes
//     with `publish` (the async-proxy fence, then `scratchy::ops::gmem_counter_arrive`, a release
//     add at GPU scope); a consumer waits with `scratchy::ops::gmem_counter_wait` (an acquire load
//     at GPU scope, spun with a sleep). Every counter index and every count waited for is a
//     constant of the instruction list.
//
// THE TEST PROGRAM'S OPCODES (M0 Day 1; the MLP program's two are below, at `OP_GATE_UP`):
//   - `OP_NOOP` (0): nothing.
//   - `OP_CHAIN` (1): lane warp 0's lane 0 waits for `counters[a1] ≥ a2` (when `a2` is non-zero),
//     reads `chain[a3]` (0 when it waited for nothing), writes `chain[a4] = that + 1`, and publishes
//     `counters[a5] += 1`. A chain over every lane of every SM checks the ring, the dispatch and the
//     two gaps: a missed acquire or release reads a stale value, a missed publish hangs (the test's
//     watchdog).
//   - `OP_PAGE_ECHO` (2): row `a1` of `x [R, 4096]` f32 TMA-loaded into the lane's page `a2`, then
//     TMA-stored to row `a3` of `out [R, 4096]`, by lane warp 0. It checks that each lane's pages
//     are its own and that a page round-trips through the TMA unit.
//
// One warp's lane 0 issues every TMA copy (TK's `warp::tma` does). The consumers of a lane sync on
// their own named barrier (`LANE_BARRIER + L`, 256 threads), never `__syncthreads`, which would
// tie the two lanes together.

#include "kittens.cuh"
#include "scratchy_ops.cuh"

#include <climits>
#include <cstdint>

using namespace kittens;

// Launcher return codes that are not a `cudaError_t`. `src/kernels.rs` mirrors these two values.
#define SCRATCHY_CUDA_ERR_SHAPE (-1)
#define SCRATCHY_CUDA_ERR_HOST (-2)

namespace {

// The H100's SM count: the grid, and the instruction list's workers per lane.
constexpr int SMS = 132;
constexpr int LANES = 2;
constexpr int LANE_WARPS = 8;
constexpr int CONSUMER_WARPS = LANES * LANE_WARPS;
constexpr int CONTROLLER_WARP = CONSUMER_WARPS;
constexpr int WARPS = CONSUMER_WARPS + 4;
constexpr int NUM_THREADS = WARPS * kittens::WARP_THREADS;
constexpr int STAGES = 2;
constexpr int INSTRUCTION_WIDTH = 32;
// Named barrier 0 is `__syncthreads`; lane L's consumers use `LANE_BARRIER + L`.
constexpr int LANE_BARRIER = 1;

constexpr int PAGE_FLOATS = 4096;
constexpr int PAGES = 13;
constexpr int LANE_PAGES = 6;
static_assert(LANES * LANE_PAGES + 1 == PAGES, "two lanes' pages and the SM's");

// The opcodes, and the argument words they read (`src/kernels.rs`'s test entry mirrors them).
constexpr int OP_NOOP = 0;
constexpr int OP_CHAIN = 1;
constexpr int OP_PAGE_ECHO = 2;

using instruction_sv = sv<int, INSTRUCTION_WIDTH>;
using page_sv = sv_fl<PAGE_FLOATS>;
static_assert(sizeof(instruction_sv) == 128 && sizeof(page_sv) == 16384, "128 B, 16 KiB");
// The pages plus the slack the allocator may spend aligning the first to 1024 bytes.
constexpr int MK_SMEM_BYTES = PAGES * int(sizeof(page_sv)) + 1024;
static_assert(MK_SMEM_BYTES + 4 * 1024 <= kittens::MAX_SHARED_MEMORY, "with the static ring");

using instruction_gl = gl<int, 1, -1, -1, INSTRUCTION_WIDTH, instruction_sv>;
using counter_gl = gl<uint32_t, 1, 1, 1, -1>;
using chain_gl = gl<int, 1, 1, 1, -1>;
using rows_gl = gl<float, 1, 1, -1, PAGE_FLOATS, page_sv>;

struct mk_globals {
    instruction_gl instructions;
    counter_gl counters;
    chain_gl chain;
    rows_gl x;
    rows_gl out;
    int n;
};

// Lane `LANE`'s pages, typed `page_sv`: a TK allocator over the dynamic shared memory, past the
// lanes before it.
template <int LANE>
__device__ inline page_sv (&pages_of())[LANE_PAGES] {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    if constexpr (LANE > 0) {
        al.allocate<page_sv, LANE * LANE_PAGES>();
    }
    return al.allocate<page_sv, LANE_PAGES>();
}

// The async-proxy fence on global memory: orders this thread's generic-proxy accesses with its
// async-proxy (TMA) ones. The one inline PTX of the file (`ASM_ALLOWLIST`): TK at the pin emits a
// proxy fence only for shared memory, inside its own TMA and MMA wrappers.
__device__ inline void async_proxy_fence() {
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
}

// Make this thread's prior writes, generic or TMA (after `store_async_wait`), visible to every
// SM's reads, then add `n` to `counters[idx]` with release order at GPU scope. One thread calls it.
__device__ inline void publish(const counter_gl& counters, int idx, uint32_t n) {
    async_proxy_fence();
    scratchy::ops::gmem_counter_arrive(counters, {0, 0, 0, idx}, n);
}

// Wait until `counters[idx] ≥ want`, then order the acquire before this thread's TMA reads as well
// as its generic ones. One thread calls it.
__device__ inline void await(const counter_gl& counters, int idx, uint32_t want) {
    scratchy::ops::gmem_counter_wait(counters, {0, 0, 0, idx}, want);
    async_proxy_fence();
}

// `OP_CHAIN`, by lane warp 0.
__device__ inline void op_chain(const mk_globals& g, const instruction_sv& ins) {
    if (warp::laneid() == 0) {
        const uint32_t want = uint32_t(ins[2]);
        int prev = 0;
        if (want > 0) {
            await(g.counters, ins[1], want);
            prev = g.chain[{0, 0, 0, ins[3]}];
        }
        g.chain[{0, 0, 0, ins[4]}] = prev + 1;
        publish(g.counters, ins[5], 1);
    }
    __syncwarp();
}

// `OP_PAGE_ECHO`, by lane warp 0: `phase` is the lane's count of echoes so far (its semaphore's
// phase).
template <int LANE>
__device__ inline void op_page_echo(const mk_globals& g, const instruction_sv& ins,
                                    semaphore& echoed, int phase) {
    page_sv(&pages)[LANE_PAGES] = pages_of<LANE>();
    page_sv& page = pages[ins[2]];
    warp::tma::expect_bytes(echoed, uint32_t(sizeof(page_sv)));
    warp::tma::load_async(page, g.x, {0, 0, ins[1], 0}, echoed);
    wait(echoed, phase % 2);
    warp::tma::store_async(g.out, page, {0, 0, ins[3], 0});
    // The page is read before the lane's next instruction may load into it.
    warp::tma::store_async_wait();
    __syncwarp();
}

// THE INTERPRETER, shared by every program (`mk_test_ker`, and M0's `mk_mlp_ker`).

// One lane's instruction ring: its stages and their semaphores (`arrived`: the stage holds its
// instruction; `finished`: the lane's warps are done with it).
struct ring_state {
    instruction_sv (&stage)[STAGES];
    semaphore (&arrived)[STAGES];
    semaphore (&finished)[STAGES];
};

// A lane's controller warp: instruction `i` of row `worker` of `instructions` into stage `i % 2`,
// once the stage's instruction `i − 2` is done. `n` is the launch's list length, every worker's.
__device__ inline void control(const instruction_gl& instructions, int n, int worker,
                               const ring_state& r) {
    for (int i = 0; i < n; i++) {
        const int s = i % STAGES;
        if (i >= STAGES) {
            wait(r.finished[s], (i / STAGES - 1) % 2);
        }
        warp::tma::expect_bytes(r.arrived[s], uint32_t(sizeof(instruction_sv)));
        warp::tma::load_async(r.stage[s], instructions, {0, worker, i, 0}, r.arrived[s]);
    }
}

// A lane's `LANE_W` consumer warps: instruction `i` from stage `i % 2`, `run` on it by every warp
// of the lane; then, once every warp is done (the lane's named barrier `barrier`), `done` on it by
// every warp (a publish runs there, after every store of the instruction), and the stage handed back.
template <int LANE_W, typename RUN, typename DONE>
__device__ inline void play(int n, const ring_state& r, int barrier, RUN&& run, DONE&& done) {
    for (int i = 0; i < n; i++) {
        const int s = i % STAGES;
        wait(r.arrived[s], (i / STAGES) % 2);
        run(r.stage[s]);
        // Every warp of the lane is done with the stage (and its stores) before `done`.
        group<LANE_W>::sync(barrier);
        done(r.stage[s]);
        if (warp::laneid() == 0) {
            arrive(r.finished[s]);
        }
    }
}

// THE INTERPRETER'S OWN TEST PROGRAM (M0 Day 1): two lanes per SM, `OP_CHAIN` and `OP_PAGE_ECHO`.

// Lane `LANE`'s consumers.
template <int LANE>
__device__ inline void consume(const mk_globals& g, const ring_state& r, semaphore& echoed) {
    const int lw = kittens::warpid() % LANE_WARPS;
    int echoes = 0;
    play<LANE_WARPS>(
        g.n, r, LANE_BARRIER + LANE,
        [&](const instruction_sv& ins) {
            const int op = ins[0];
            if (lw == 0 && op != OP_NOOP) {
                if (op == OP_CHAIN) {
                    op_chain(g, ins);
                } else if (op == OP_PAGE_ECHO) {
                    op_page_echo<LANE>(g, ins, echoed, echoes);
                }
            }
            echoes += op == OP_PAGE_ECHO ? 1 : 0;
        },
        [](const instruction_sv&) {});
}

__global__ __launch_bounds__(NUM_THREADS, 1) void mk_test_ker(const __grid_constant__ mk_globals g) {
    __shared__ sv<int, INSTRUCTION_WIDTH> ring0[STAGES];
    __shared__ sv<int, INSTRUCTION_WIDTH> ring1[STAGES];
    __shared__ semaphore arrived0[STAGES], finished0[STAGES], arrived1[STAGES], finished1[STAGES];
    __shared__ semaphore echoed0, echoed1;
    const int w = kittens::warpid();
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived0[s], 0, 1);
            init_semaphore(arrived1[s], 0, 1);
            init_semaphore(finished0[s], LANE_WARPS, 0);
            init_semaphore(finished1[s], LANE_WARPS, 0);
        }
        init_semaphore(echoed0, 0, 1);
        init_semaphore(echoed1, 0, 1);
    }
    // Every semaphore is initialised before any thread waits on it or a TMA arrives on it.
    __syncthreads();
    const ring_state r0{ring0, arrived0, finished0}, r1{ring1, arrived1, finished1};
    if (w < LANE_WARPS) {
        consume<0>(g, r0, echoed0);
    } else if (w < CONSUMER_WARPS) {
        consume<1>(g, r1, echoed1);
    } else if (w == CONTROLLER_WARP) {
        control(g.instructions, g.n, LANES * int(blockIdx.x), r0);
    } else if (w == CONTROLLER_WARP + 1) {
        control(g.instructions, g.n, LANES * int(blockIdx.x) + 1, r1);
    }
    // No warp leaves while another may still arrive on a semaphore it waits on.
    __syncthreads();
}

// THE M0 MLP PROGRAM (`mk_mlp_ker`): Gemma's dense-MLP block of one layer, `y = x +
// down(gelu_tanh_mul(gate(n), up(n)))`, `n = RMSNorm(x)`, as two opcodes on one lane of 8 warps per
// SM (132 workers). One instruction is the work of one of today's 8-warp CTAs, with today's code
// in today's order, so its bits are today's:
// Every instruction names its order: `a3` a counter to wait for and `a4` the count (0: none),
// before its reads; `a5` the counter it adds 1 to after its stores. The layers chain through
// `h [layers + 1, 2,816]`: layer `L` reads `h[L]` and writes `h[L + 1]`, so the test's lists make
// layer L + 1's gate/up wait for layer L's 176 down blocks, and layer L's down for its 66 gate/up
// blocks.
//   - `OP_GATE_UP` (3): `a1` the layer, `a2` the block. The one-row norm of `h[layer]` (2,816 bf16,
//     the pre-feedforward norm, `rms_norm_one_row_ker<NORM>`'s pass 1 by warps 0..3 and its pass 2,
//     every tile, by warps 0..3 in turn as their tiles' owners), stored in the lane's row; then
//     `gemv_fp8.cu`'s `quantize_row` of it and its `gate_up_gemv_fp8_ker<false, gelu_tanh_mul>`
//     body on block `a2` (32 rows of 4 per warp, 2,112 of them over 66 blocks, one split), stored to
//     `act[layer]`.
//   - `OP_DOWN` (4): `a1` the layer, `a2` the block. `gemv_fp8_ker<false, 2>`'s body on block `a2`
//     (16 rows of 2 per warp, 2,816 over 176 blocks, one split) over `act[layer]` with W4a's
//     residual epilogue (`gemv_fp8_residual`), the residual `h[layer]`, stored to `h[layer + 1]`.
// Its weights are TMA-loaded before the wait, so their DRAM round trip overlaps the gate/up's tail.
// The norm computed once per instruction is one CTA's worth, as today's GEMV recomputes its
// prologue in each CTA (the norm is the one-row kernel's own code, so its bits are that kernel's).
//
// Shared memory, carved from the pages: every buffer is written by one proxy only, the TMA unit or
// the threads, never both, so no shared-memory proxy fence is needed (the norm's tiles, the gain,
// the down's input and the weight ring by TMA; the normed or quantized row, the sums and the
// stored tile by the threads). The lane's semaphores and the amax vector are static. A register tile `rt_fl<16, 256>` is 128 f32 a thread,
// so the program runs 9 warps (8 consumers and the controller): 224 registers a thread at most.

constexpr int OP_GATE_UP = 3;
constexpr int OP_DOWN = 4;

// Gemma 4 26B-A4B's dense MLP: hidden 2,816 (11 norm tiles of 256), intermediate 2,112.
constexpr int HIDDEN = 2816;
constexpr int INTER = 2112;
constexpr int NORM_TILES = HIDDEN / 256;
constexpr int MLP_WARPS = LANE_WARPS + 1;
constexpr int MLP_THREADS = MLP_WARPS * kittens::WARP_THREADS;
// `gemv_fp8.cu`'s geometry: 8 warps of 4 rows (gate/up), of 2 rows (the down at 176 blocks).
constexpr int GEMV_CHUNK = 256;
constexpr int GEMV_STAGES = 4;
constexpr int GU_ROWS = 4;
constexpr int DOWN_ROWS = 2;
constexpr int GU_BLOCKS = (INTER + LANE_WARPS * GU_ROWS - 1) / (LANE_WARPS * GU_ROWS);
constexpr int DOWN_BLOCKS = (HIDDEN + LANE_WARPS * DOWN_ROWS - 1) / (LANE_WARPS * DOWN_ROWS);
static_assert(GU_BLOCKS == 66 && DOWN_BLOCKS == 176, "today's grids at these shapes");
// The rows' chunks: the norm's output (11 of 256) and the down's input (2,112 / 256 = 8.25: 9).
constexpr int ROW_CHUNKS = 11;
constexpr int GU_CHUNKS = HIDDEN / GEMV_CHUNK;
constexpr int DOWN_CHUNKS = (INTER + GEMV_CHUNK - 1) / GEMV_CHUNK;
static_assert(GU_CHUNKS == 11 && DOWN_CHUNKS == 9 && DOWN_CHUNKS <= ROW_CHUNKS, "the rows fit");

using norm_tile = st_bf<16, 256, false>;
using norm_rt = rt_fl<16, 256>;
using norm_rows = norm_rt::col_vec;
using norm_cols = norm_rt::row_vec;
using gain_sv = sv_bf<256>;
using sums_sv = sv_fl<16>;
using row_sv = sv_bf<GEMV_CHUNK>;
using w_sv = sv_fp8e4m3<GEMV_CHUNK>;
using chunk_rv = rv_fl<GEMV_CHUNK>;
using amax_rv = rv_fl<16>;

using x_gl = gl<bf16, 1, -1, 1, HIDDEN, norm_tile, row_sv>;
using gain_gl = gl<bf16, 1, 1, -1, HIDDEN, gain_sv>;
using act_gl = gl<bf16, 1, -1, 1, INTER, row_sv>;
using weight_gl = gl<fp8e4m3, 1, -1, -1, -1, w_sv>;
using scale_gl = gl<float, 1, 1, -1, -1>;
using out_gl = gl<bf16, 1, -1, 1, HIDDEN>;

struct mlp_globals {
    instruction_gl instructions;
    counter_gl counters;
    // `h[L]` is layer L's input (TMA view `x`, element view `y`); per layer the norm's gain, the
    // gate/up/down weights and scales, and the activation between the two opcodes.
    x_gl x;
    out_gl y;
    gain_gl gain;
    weight_gl gate;
    weight_gl up;
    weight_gl down;
    scale_gl gate_scale;
    scale_gl up_scale;
    scale_gl down_scale;
    act_gl act;
    float eps;
    float gain_offset;
    int n;
};

// The lane's shared memory, carved from the pages by a TK allocator: the norm's tiles (pass 1's
// two stages per warp and pass 2's owned tile), its gains and sums, the row, and the weight ring.
struct mlp_smem {
    norm_tile (&tiles)[4][2];
    norm_tile (&owned)[2];
    gain_sv& gain;
    sums_sv (&sums)[4];
    row_sv (&row)[ROW_CHUNKS];
    row_sv (&act)[DOWN_CHUNKS];
    w_sv (&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS];
};

__device__ inline mlp_smem mlp_pages() {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    norm_tile(&tiles)[4][2] = al.allocate<norm_tile, 4, 2>();
    norm_tile(&owned)[2] = al.allocate<norm_tile, 2>();
    gain_sv& gain = al.allocate<gain_sv>();
    sums_sv(&sums)[4] = al.allocate<sums_sv, 4>();
    row_sv(&row)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    row_sv(&act)[DOWN_CHUNKS] = al.allocate<row_sv, DOWN_CHUNKS>();
    w_sv(&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS] = al.allocate<w_sv, LANE_WARPS, GEMV_STAGES, GU_ROWS>();
    return mlp_smem{tiles, owned, gain, sums, row, act, b};
}
// 10 norm tiles of 8 KiB, the gain and sums, the two rows and the ring of 8 × 4 × 4 × 256 B, each
// allocation aligned to 1,024 bytes: inside the 13 pages.
static_assert(10 * sizeof(norm_tile) + (ROW_CHUNKS + DOWN_CHUNKS) * sizeof(row_sv) +
                      LANE_WARPS * GEMV_STAGES * GU_ROWS * sizeof(w_sv) + 7 * 1024 <=
                  size_t(MK_SMEM_BYTES),
              "the lane's buffers fit the pages");

// The lane's semaphores, each with the count of times it was waited on so far (its phase).
struct mlp_sems {
    semaphore (&arrived)[4][2];
    semaphore& owned;
    semaphore& act;
    semaphore (&ring)[LANE_WARPS][GEMV_STAGES];
    sums_sv& amax;
};

// The lane's 8 consumer warps sync on named barrier 1 (288 threads would include the controller).
__device__ inline void lane_sync() { group<LANE_WARPS>::sync(LANE_BARRIER); }

// `rms_norm_one_row_ker<NORM>`'s pass 1 on `x[layer]` by warps 0..3 (the other four idle at the
// barriers), then each warp's `inv`; then its pass 2 for every tile `t`, by warp `t % 4` (that
// tile's owner in today's grid of 11 CTAs, each warp 0 there), stored to the lane's row. Returns
// once every chunk of the row is in shared memory, in every warp. `uses` counts this lane's norms
// (the semaphores' phase).
__device__ inline void norm_row(const mlp_globals& g, const mlp_smem& sm, const mlp_sems& se,
                                int layer, int uses) {
    const int w = kittens::warpid();
    norm_rows inv;
    if (w < 4) {
        const int mine = (NORM_TILES - w + 3) / 4;
        // Load `i` of this warp's pass 1 lands in stage `i % 2`; the phase counts every load of
        // the stage over the lane's norms so far.
        const auto issue = [&](int i) {
            const int stage = i % 2, t = w + i * 4;
            warp::tma::expect_bytes(se.arrived[w][stage], uint32_t(sizeof(norm_tile)));
            warp::tma::load_async(sm.tiles[w][stage], g.x, {0, layer, 0, t}, se.arrived[w][stage]);
        };
        const int per_stage = (mine + 1) / 2;
        norm_rows part;
        warp::zero(part);
        issue(0);
        for (int i = 0; i < mine; i++) {
            const int stage = i % 2;
            __syncwarp();
            if (i + 1 < mine) {
                issue(i + 1);
            }
            const int k = uses * (stage == 0 ? per_stage : mine / 2) + i / 2;
            wait(se.arrived[w][stage], k % 2);
            norm_rt x;
            warp::load(x, sm.tiles[w][stage]);
            warp::mul(x, x, x);
            warp::row_sum(part, x, part);
        }
        warp::store(sm.sums[w], part);
    }
    lane_sync();
    if (w < 4) {
        norm_rows total, other;
        warp::load(total, sm.sums[0]);
        for (int v = 1; v < 4; v++) {
            warp::load(other, sm.sums[v]);
            warp::add(total, total, other);
        }
        norm_rows rcp;
        warp::one(rcp);
        warp::div(rcp, rcp, float(HIDDEN));
        scratchy::ops::rv_ternary_map<scratchy::ops::rms_inv_fused>(inv, total, rcp, g.eps);
    }
    // Pass 2, one tile at a time by its owner warp, into the row (its chunk `t` is the tile's row 0).
    for (int t = 0; t < NORM_TILES; t++) {
        if (w == t % 4) {
            const int k = uses * NORM_TILES + t;
            warp::tma::expect_bytes(se.owned, uint32_t(sizeof(norm_tile) + sizeof(gain_sv)));
            warp::tma::load_async(sm.owned[0], g.x, {0, layer, 0, t}, se.owned);
            warp::tma::load_async(sm.gain, g.gain, {0, 0, layer, t}, se.owned);
            wait(se.owned, k % 2);
            norm_rt x;
            warp::load(x, sm.owned[0]);
            warp::mul_row(x, x, inv);
            norm_cols gain;
            warp::load(gain, sm.gain);
            warp::add(gain, gain, g.gain_offset);
            warp::mul_col(x, x, gain);
            warp::store(sm.owned[1], x);
            __syncwarp();
            for (int e = warp::laneid(); e < GEMV_CHUNK; e += kittens::WARP_THREADS) {
                sm.row[t][e] = sm.owned[1][{0, e}];
            }
            __syncwarp();
        }
        // The owned tiles are free for the next owner.
        lane_sync();
    }
}

// `gemv_fp8.cu`'s `quantize_row` of `src`'s `chunks` (one split) into the lane's row: its amax by
// the 8 warps (`warp::abs`, `warp::max`, the 16-slot vector), its scale, then each chunk quantized
// (`src` may be the row itself). Every warp calls it; returns the scale.
template <int N>
__device__ inline float quantize_row(const row_sv (&src)[N], const mlp_smem& sm,
                                     const mlp_sems& se, int chunks) {
    const int w = kittens::warpid();
    chunk_rv x, m;
    warp::zero(m);
    for (int c = w; c < chunks; c += LANE_WARPS) {
        warp::load(x, src[c]);
        warp::abs(x, x);
        warp::max(m, m, x);
    }
    const float part = warp::max(m);
    if (warp::laneid() == 0) {
        se.amax[w] = part;
        se.amax[w + LANE_WARPS] = part;
    }
    lane_sync();
    amax_rv all;
    warp::zero(all);
    warp::load(all, se.amax);
    const float scale = scratchy::ops::fp8_scale::op<float>(warp::max(all));
    for (int c = w; c < chunks; c += LANE_WARPS) {
        warp::load(x, src[c]);
        warp::bin_op<scratchy::ops::fp8_quant_pre>(x, x, scale);
        warp::unary_op<scratchy::ops::fp8_round>(x, x);
        warp::store(sm.row[c], x);
    }
    lane_sync();
    return scale;
}

// `gemv_fp8.cu`'s `gemv_dots` for one warp's `ROWS` rows from `row0` of weight `b` (`depth` the
// layer) over the row's `chunks`, the ring continuing at load `j0`; `ring_uses` counts the warp's
// loads into each stage so far (the phase). `before` runs after the ring is primed. Returns each
// row's unscaled sum in every lane, rows past `n` +0.
template <int ROWS, typename BEFORE>
__device__ inline void dots(const weight_gl& b, const mlp_smem& sm, const mlp_sems& se, int depth,
                            int row0, int n, int chunks, int j0, BEFORE&& before,
                            float (&dot)[ROWS]) {
    const int w = kittens::warpid();
    const int rows = max(0, min(ROWS, n - row0));
    const bool live = rows > 0;
    const auto issue = [&](int j) {
        const int s = (j0 + j) % GEMV_STAGES;
        warp::tma::expect_bytes(se.ring[w][s], uint32_t(rows * sizeof(w_sv)));
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            if (r < rows) {
                warp::tma::load_async(sm.b[w][s][r], b, {0, depth, row0 + r, j}, se.ring[w][s]);
            }
        }
    };
    if (live) {
        for (int j = 0; j < min(GEMV_STAGES, chunks); j++) {
            issue(j);
        }
    }
    before();
    chunk_rv acc[ROWS];
    chunk_rv x, y;
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        warp::zero(acc[r]);
    }
    if (live) {
        for (int j = 0; j < chunks; j++) {
            const int s = (j0 + j) % GEMV_STAGES;
            warp::load(x, sm.row[j]);
            wait(se.ring[w][s], ((j0 + j) / GEMV_STAGES) % 2);
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (r < rows) {
                    scratchy::ops::sv_fp8_to_rv_fl(y, sm.b[w][s][r]);
                    scratchy::ops::rv_fma<scratchy::ops::fma_acc>(acc[r], y, x, acc[r]);
                }
            }
            __syncwarp();
            if (j + GEMV_STAGES < chunks) {
                issue(j + GEMV_STAGES);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        dot[r] = warp::sum(acc[r]);
    }
}

// `(dot · w_scale) · a_scale`, `gemv_fp8.cu`'s `fp8_scaled`.
__device__ inline float scaled(const scale_gl& s, int depth, int row, float dot, float a_scale) {
    return base_ops::mul::op<float>(base_ops::mul::op<float>(dot, s[{0, 0, depth, row}]), a_scale);
}

// The instruction's wait (`a3`, `a4`) by the lane's thread 0, then the lane's barrier, then every
// thread's async-proxy fence: every read after it, generic or TMA by any warp of the lane, sees
// the stores the counted publishes ordered. Every consumer warp calls it.
__device__ inline void lane_await(const mlp_globals& g, const instruction_sv& ins) {
    if (ins[4] > 0) {
        if (kittens::warpid() == 0 && warp::laneid() == 0) {
            scratchy::ops::gmem_counter_wait(g.counters, {0, 0, 0, ins[3]}, uint32_t(ins[4]));
        }
        lane_sync();
        async_proxy_fence();
    }
}

// The lane's 8 consumer warps.
__device__ inline void mlp_consume(const mlp_globals& g, const ring_state& r, const mlp_sems& se) {
    const mlp_smem sm = mlp_pages();
    const int w = kittens::warpid();
    // Each warp's weight-ring loads so far (the ring's running index), the lane's norms and its
    // down rows loaded so far (the semaphores' phases).
    int j0 = 0;
    int norms = 0;
    int downs = 0;
    play<LANE_WARPS>(
        g.n, r, LANE_BARRIER,
        [&](const instruction_sv& ins) {
            const int op = ins[0], layer = ins[1], block = ins[2];
            if (op == OP_GATE_UP) {
                const int row0 = (block * LANE_WARPS + w) * GU_ROWS;
                float a_scale = 0.0f;
                float gate[GU_ROWS], up[GU_ROWS];
                dots<GU_ROWS>(
                    g.gate, sm, se, layer, row0, INTER, GU_CHUNKS, j0,
                    [&] {
                        lane_await(g, ins);
                        norm_row(g, sm, se, layer, norms);
                        a_scale = quantize_row(sm.row, sm, se, GU_CHUNKS);
                    },
                    gate);
                dots<GU_ROWS>(g.up, sm, se, layer, row0, INTER, GU_CHUNKS, j0 + GU_CHUNKS, [] {},
                              up);
                const int rows = max(0, min(GU_ROWS, INTER - row0));
#pragma unroll
                for (int rr = 0; rr < GU_ROWS; rr++) {
                    if (warp::laneid() == 0 && rr < rows) {
                        const float gv = scaled(g.gate_scale, layer, row0 + rr, gate[rr], a_scale);
                        const float uv = scaled(g.up_scale, layer, row0 + rr, up[rr], a_scale);
                        g.act[{0, layer, 0, row0 + rr}] = base_types::convertor<bf16, float>::convert(
                            scratchy::ops::gelu_tanh_mul::op<float>(gv, uv));
                    }
                }
                j0 += rows > 0 ? 2 * GU_CHUNKS : 0;
                norms += 1;
            } else if (op == OP_DOWN) {
                const int row0 = (block * LANE_WARPS + w) * DOWN_ROWS;
                float a_scale = 0.0f;
                float dot[DOWN_ROWS];
                dots<DOWN_ROWS>(
                    g.down, sm, se, layer, row0, HIDDEN, DOWN_CHUNKS, j0,
                    [&] {
                        // Every gate/up block of the layer is stored before its row is read.
                        lane_await(g, ins);
                        if (w == 0) {
                            warp::tma::expect_bytes(se.act, uint32_t(DOWN_CHUNKS * sizeof(row_sv)));
                            for (int c = 0; c < DOWN_CHUNKS; c++) {
                                warp::tma::load_async(sm.act[c], g.act, {0, layer, 0, c}, se.act);
                            }
                        }
                        wait(se.act, downs % 2);
                        a_scale = quantize_row(sm.act, sm, se, DOWN_CHUNKS);
                    },
                    dot);
                const int rows = max(0, min(DOWN_ROWS, HIDDEN - row0));
#pragma unroll
                for (int rr = 0; rr < DOWN_ROWS; rr++) {
                    if (warp::laneid() == 0 && rr < rows) {
                        const float v = scaled(g.down_scale, layer, row0 + rr, dot[rr], a_scale);
                        g.y[{0, layer + 1, 0, row0 + rr}] =
                            scratchy::ops::with_residual(v, g.y[{0, layer, 0, row0 + rr}]);
                    }
                }
                j0 += rows > 0 ? DOWN_CHUNKS : 0;
                downs += 1;
            }
        },
        [&](const instruction_sv& ins) {
            // The instruction's stores are done (the barrier): publish them once.
            if (ins[0] != OP_NOOP && w == 0 && warp::laneid() == 0) {
                publish(g.counters, ins[5], 1);
            }
        });
}

__global__ __launch_bounds__(MLP_THREADS, 1) void mk_mlp_ker(const __grid_constant__ mlp_globals g) {
    __shared__ sv<int, INSTRUCTION_WIDTH> ring[STAGES];
    __shared__ semaphore arrived[STAGES], finished[STAGES];
    __shared__ semaphore norm_arrived[4][2], owned, act, weights[LANE_WARPS][GEMV_STAGES];
    __shared__ sv_fl<16> amax;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) {
            init_semaphore(arrived[s], 0, 1);
            init_semaphore(finished[s], LANE_WARPS, 0);
        }
        for (int w = 0; w < 4; w++) {
            init_semaphore(norm_arrived[w][0], 0, 1);
            init_semaphore(norm_arrived[w][1], 0, 1);
        }
        init_semaphore(owned, 0, 1);
        init_semaphore(act, 0, 1);
        for (int w = 0; w < LANE_WARPS; w++) {
            for (int s = 0; s < GEMV_STAGES; s++) {
                init_semaphore(weights[w][s], 0, 1);
            }
        }
    }
    // Every semaphore is initialised before any thread waits on it or a TMA arrives on it.
    __syncthreads();
    const ring_state r{ring, arrived, finished};
    if (kittens::warpid() < LANE_WARPS) {
        const mlp_sems se{norm_arrived, owned, act, weights, amax};
        mlp_consume(g, r, se);
    } else {
        control(g.instructions, g.n, int(blockIdx.x), r);
    }
    // No warp leaves while another may still arrive on a semaphore it waits on.
    __syncthreads();
}

// The MLP program's operands, as the launcher was given them.
struct mlp_args {
    const void* instructions = nullptr;
    void* counters = nullptr;
    void* h = nullptr;
    const void* gain = nullptr;
    const void* gate = nullptr;
    const void* gate_scale = nullptr;
    const void* up = nullptr;
    const void* up_scale = nullptr;
    const void* down = nullptr;
    const void* down_scale = nullptr;
    void* act = nullptr;
};

// The launch's operands, as the launcher was given them.
struct mk_args {
    const void* instructions = nullptr;
    void* counters = nullptr;
    void* chain = nullptr;
    const void* x = nullptr;
    void* out = nullptr;
};

}  // namespace

// The kernel's one-time setup, run once per device by `kernels::prepare_all`: allow it
// `MK_SMEM_BYTES` of dynamic shared memory (which also loads its module).
//
// 0 = done; any other value = the cudaError_t of setting the attribute.
extern "C" int scratchy_cuda_mk_prepare(void) {
    for (const cudaError_t e : {
             cudaFuncSetAttribute(mk_test_ker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
         }) {
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }
    return 0;
}

// Whether the current device has the 132 SMs the grid assumes: 0 yes, SCRATCHY_CUDA_ERR_SHAPE no,
// else the cudaError_t of the query.
static int check_sms() {
    int device = 0, sms = 0;
    if (const cudaError_t e = cudaGetDevice(&device); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    return sms == SMS ? 0 : SCRATCHY_CUDA_ERR_SHAPE;
}

// Test support (`tests/megakernel.rs`), M0's go/no-go: play the MLP program's
// `instructions [SMS, n, 32]` int over `layers` of Gemma's dense MLP (hidden 2,816, intermediate
// 2,112), in one cooperative launch of 132 CTAs of 288 threads:
//   - `h [layers + 1, 2816]` bf16: `h[0]` the input, `h[L + 1]` layer L's output;
//   - `gain [layers, 2816]` bf16, the norm's weight, `eps` and `gain_offset` its scalars;
//   - `gate`, `up [layers, 2112, 2816]` and `down [layers, 2816, 2112]` e4m3, their scales
//     `[layers, 2112]`, `[layers, 2112]`, `[layers, 2816]` f32;
//   - `act [layers, 2112]` bf16, each layer's activation;
//   - `counters [counter_count]` u32, zeroed by the caller.
//
// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched: `n`, `layers` or
// `counter_count` zero or above INT_MAX, `eps` not finite and positive, or the device not 132 SMs;
// SCRATCHY_CUDA_ERR_HOST = a C++ exception was caught; any other value = the cudaError_t of the
// launch or the SM-count query. Asynchronous on `stream`.
extern "C" int scratchy_cuda_mk_mlp_test(const void* instructions, uint32_t n, void* counters,
                                         uint32_t counter_count, void* h, const void* gain,
                                         const void* gate, const void* gate_scale, const void* up,
                                         const void* up_scale, const void* down,
                                         const void* down_scale, void* act, uint32_t layers,
                                         float eps, float gain_offset,
                                         void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || n > uint32_t(INT_MAX) || layers == 0 || layers >= uint32_t(INT_MAX) ||
        counter_count == 0 || counter_count > uint32_t(INT_MAX) || !(eps > 0.0f && eps <= 3.0e38f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (const int e = check_sms(); e != 0) {
        return e;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const mlp_args a{instructions, counters, h, gain, gate, gate_scale, up, up_scale,
                         down, down_scale, act};
        const auto bf = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
        const auto e4 = [](const void* p) { return static_cast<fp8e4m3*>(const_cast<void*>(p)); };
        const auto f = [](const void* p) { return static_cast<float*>(const_cast<void*>(p)); };
        const size_t L = layers;
        instruction_gl ig{static_cast<int*>(const_cast<void*>(a.instructions)), nullptr,
                          size_t(SMS), size_t(n), nullptr};
        counter_gl cg{static_cast<uint32_t*>(a.counters), nullptr, nullptr, nullptr,
                      size_t(counter_count)};
        x_gl xg{bf(a.h), nullptr, L + 1, nullptr, nullptr};
        out_gl yg{bf(a.h), nullptr, L + 1, nullptr, nullptr};
        gain_gl gg{bf(a.gain), nullptr, nullptr, L, nullptr};
        weight_gl gate_g{e4(a.gate), nullptr, L, size_t(INTER), size_t(HIDDEN)};
        weight_gl up_g{e4(a.up), nullptr, L, size_t(INTER), size_t(HIDDEN)};
        weight_gl down_g{e4(a.down), nullptr, L, size_t(HIDDEN), size_t(INTER)};
        scale_gl gs{f(a.gate_scale), nullptr, nullptr, L, size_t(INTER)};
        scale_gl us{f(a.up_scale), nullptr, nullptr, L, size_t(INTER)};
        scale_gl ds{f(a.down_scale), nullptr, nullptr, L, size_t(HIDDEN)};
        act_gl ag{bf(a.act), nullptr, L, nullptr, nullptr};
        mlp_globals G{ig, cg, xg, yg, gg, gate_g, up_g, down_g, gs, us, ds, ag, eps, gain_offset,
                      int(n)};
        void* args[] = {&G};
        const dim3 grid{uint32_t(SMS)}, block{uint32_t(MLP_THREADS)};
        return static_cast<int>(cudaLaunchCooperativeKernel(mk_mlp_ker, grid, block, args,
                                                            MK_SMEM_BYTES,
                                                            static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/megakernel.rs`), M0 Day 1: play `instructions [SMS · LANES, n, 32]` int,
// the counters `[counters]` u32 (zeroed by the caller), `chain [chain]` int, and `x`, `out`
// `[rows, 4096]` f32, in one cooperative launch of 132 CTAs.
//
// 0 = launched; SCRATCHY_CUDA_ERR_SHAPE = refused, nothing launched: `n` zero or above INT_MAX,
// `counters`, `chain` or `rows` zero, or the device not 132 SMs; SCRATCHY_CUDA_ERR_HOST = a C++
// exception was caught (a tensor-map encode); any other value = the cudaError_t of the launch or
// of the SM-count query. Asynchronous on `stream`.
extern "C" int scratchy_cuda_mk_test(const void* instructions, uint32_t n, void* counters,
                                     uint32_t counter_count, void* chain, uint32_t chain_count,
                                     const void* x, void* out, uint32_t rows,
                                     void* stream /* cudaStream_t; null = default */) {
    if (n == 0 || n > uint32_t(INT_MAX) || counter_count == 0 || chain_count == 0 || rows == 0 ||
        rows > uint32_t(INT_MAX)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (const int e = check_sms(); e != 0) {
        return e;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const mk_args a{instructions, counters, chain, x, out};
        instruction_gl ig{static_cast<int*>(const_cast<void*>(a.instructions)), nullptr,
                          size_t(SMS * LANES), size_t(n), nullptr};
        counter_gl cg{static_cast<uint32_t*>(a.counters), nullptr, nullptr, nullptr,
                      size_t(counter_count)};
        chain_gl chg{static_cast<int*>(a.chain), nullptr, nullptr, nullptr, size_t(chain_count)};
        rows_gl xg{static_cast<float*>(const_cast<void*>(a.x)), nullptr, nullptr, size_t(rows),
                   nullptr};
        rows_gl og{static_cast<float*>(a.out), nullptr, nullptr, size_t(rows), nullptr};
        mk_globals G{ig, cg, chg, xg, og, int(n)};
        void* args[] = {&G};
        const dim3 grid{uint32_t(SMS)}, block{uint32_t(NUM_THREADS)};
        return static_cast<int>(cudaLaunchCooperativeKernel(mk_test_ker, grid, block, args,
                                                            MK_SMEM_BYTES,
                                                            static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
