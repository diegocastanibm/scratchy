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

// M0-A (the ablation, test support only; `V = 0` is the program above, unchanged): each flag
// removes one suspected cost, so its time can be measured. Only `V = 0` and `FULL_WAVE` alone keep
// the bits.
//   - `NORM_ONCE`: the gate/up reads the layer's normed row `nrow[layer]` (computed by the host's
//     launch of today's norm) instead of norming `h[layer]` itself.
//   - `QUANT_ONCE`: both opcodes read a pre-quantized row (`qrow[layer]` / `qact[layer]`, e4m3 values
//     as bf16, and their scales `qscale`) instead of running `quantize_row`. Implies `NORM_ONCE`
//     for the gate/up.
//   - `NO_WAIT`: no counter wait (the lists name none, the data is already there).
//   - `FULL_WAVE`: the gate/up at 2 rows a warp (132 blocks of 16 rows) instead of 4 (66 of 32).
//   - `TWO_LANES`: two lanes of 8 consumer warps per SM (264 workers), with `NORM_ONCE` and
//     `QUANT_ONCE` (no norm tile: the registers fit).
constexpr int NORM_ONCE = 1;
constexpr int QUANT_ONCE = 2;
constexpr int NO_WAIT = 4;
constexpr int FULL_WAVE = 8;
constexpr int TWO_LANES = 16;
// A trace record: 8 `clock64` stamps per instruction (`T_*`), `[workers, n, 8]` u64, when the
// launcher is given a trace buffer.
constexpr int T_START = 0;
constexpr int T_PRIMED = 1;
constexpr int T_WAITED = 2;
constexpr int T_NORMED = 3;
constexpr int T_QUANTIZED = 4;
constexpr int T_DOTS = 5;
constexpr int T_STORED = 6;
constexpr int T_ARRIVED = 7;
constexpr int TRACE_WIDTH = 8;

// Gemma 4 26B-A4B's dense MLP: hidden 2,816 (11 norm tiles of 256), intermediate 2,112.
constexpr int HIDDEN = 2816;
constexpr int INTER = 2112;
constexpr int NORM_TILES = HIDDEN / 256;
constexpr int MLP_WARPS = LANE_WARPS + 1;
// `gemv_fp8.cu`'s geometry: 8 warps of 4 rows (gate/up), of 2 rows (the down at 176 blocks).
constexpr int GEMV_CHUNK = 256;
constexpr int GEMV_STAGES = 4;
constexpr int GU_ROWS = 4;
constexpr int GU_ROWS_FULL = 2;
constexpr int DOWN_ROWS = 2;
constexpr int GU_BLOCKS = (INTER + LANE_WARPS * GU_ROWS - 1) / (LANE_WARPS * GU_ROWS);
constexpr int DOWN_BLOCKS = (HIDDEN + LANE_WARPS * DOWN_ROWS - 1) / (LANE_WARPS * DOWN_ROWS);
constexpr int GU_BLOCKS_FULL = (INTER + LANE_WARPS * GU_ROWS_FULL - 1) / (LANE_WARPS * GU_ROWS_FULL);
static_assert(GU_BLOCKS == 66 && DOWN_BLOCKS == 176 && GU_BLOCKS_FULL == 132,
              "today's grids at these shapes, and the full wave's");
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
using trace_gl = gl<uint64_t, 1, -1, -1, TRACE_WIDTH>;
using qscale_gl = gl<float, 1, 1, -1, 2>;

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
    // M0-A only: the precomputed rows (`NORM_ONCE`: `nrow`; `QUANT_ONCE`: `qrow`, `qact` and the
    // scales `qscale[layer] = {gate/up's, down's}`) and the trace (`trace_on` zero: none).
    x_gl nrow;
    x_gl qrow;
    act_gl qact;
    qscale_gl qscale;
    trace_gl trace;
    int trace_on;
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
    // M0-A: a precomputed row, TMA-written only.
    row_sv (&pre)[ROW_CHUNKS];
    w_sv (&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS];
};

// The norm-free form's lane buffers (`TWO_LANES`): its two rows and its weight ring, lane `LANE`'s
// past lane 0's.
struct lane_smem {
    row_sv (&row)[ROW_CHUNKS];
    row_sv (&act)[DOWN_CHUNKS];
    // M0-A: a precomputed row, TMA-written only.
    row_sv (&pre)[ROW_CHUNKS];
    w_sv (&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS];
};

template <int LANE>
__device__ inline lane_smem lane_pages() {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    if constexpr (LANE > 0) {
        al.allocate<row_sv, ROW_CHUNKS>();
        al.allocate<row_sv, DOWN_CHUNKS>();
        al.allocate<row_sv, ROW_CHUNKS>();
        al.allocate<w_sv, LANE_WARPS, GEMV_STAGES, GU_ROWS>();
    }
    row_sv(&row)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    row_sv(&act)[DOWN_CHUNKS] = al.allocate<row_sv, DOWN_CHUNKS>();
    row_sv(&pre)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS] = al.allocate<w_sv, LANE_WARPS, GEMV_STAGES, GU_ROWS>();
    return lane_smem{row, act, pre, b};
}
static_assert(2 * ((2 * ROW_CHUNKS + DOWN_CHUNKS) * sizeof(row_sv) +
                   LANE_WARPS * GEMV_STAGES * GU_ROWS * sizeof(w_sv) + 4 * 1024) <=
                  size_t(MK_SMEM_BYTES),
              "two lanes' rows and rings fit the pages");

__device__ inline mlp_smem mlp_pages() {
    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]);
    norm_tile(&tiles)[4][2] = al.allocate<norm_tile, 4, 2>();
    norm_tile(&owned)[2] = al.allocate<norm_tile, 2>();
    gain_sv& gain = al.allocate<gain_sv>();
    sums_sv(&sums)[4] = al.allocate<sums_sv, 4>();
    row_sv(&row)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    row_sv(&act)[DOWN_CHUNKS] = al.allocate<row_sv, DOWN_CHUNKS>();
    row_sv(&pre)[ROW_CHUNKS] = al.allocate<row_sv, ROW_CHUNKS>();
    w_sv(&b)[LANE_WARPS][GEMV_STAGES][GU_ROWS] = al.allocate<w_sv, LANE_WARPS, GEMV_STAGES, GU_ROWS>();
    return mlp_smem{tiles, owned, gain, sums, row, act, pre, b};
}
// 10 norm tiles of 8 KiB, the gain and sums, the two rows and the ring of 8 × 4 × 4 × 256 B, each
// allocation aligned to 1,024 bytes: inside the 13 pages.
static_assert(10 * sizeof(norm_tile) + (2 * ROW_CHUNKS + DOWN_CHUNKS) * sizeof(row_sv) +
                      LANE_WARPS * GEMV_STAGES * GU_ROWS * sizeof(w_sv) + 8 * 1024 <=
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

// The lane's 8 consumer warps sync on their named barrier: 1 (288 threads would include the
// controller), lane 1's 2 under `TWO_LANES`. A lane's warps are `8L .. 8L + 7`.
__device__ inline int lane_of() { return kittens::warpid() / LANE_WARPS; }
__device__ inline void lane_sync() { group<LANE_WARPS>::sync(LANE_BARRIER + lane_of()); }

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
template <int N, typename SM>
__device__ inline float quantize_row(const row_sv (&src)[N], const SM& sm, const mlp_sems& se,
                                     int chunks) {
    const int w = kittens::warpid() % LANE_WARPS;
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
// layer) over `xrow`'s `chunks` (the quantized row), the ring continuing at load `j0` (the warp's
// loads so far, the stages' phase). `before` runs after the ring is primed, and fills `xrow`.
// Returns each row's unscaled sum in every lane, rows past `n` +0.
template <int ROWS, typename SM, int N, typename BEFORE>
__device__ inline void dots(const weight_gl& b, const SM& sm, const mlp_sems& se, int depth,
                            int row0, int n, const row_sv (&xrow)[N], int chunks, int j0,
                            BEFORE&& before, float (&dot)[ROWS]) {
    const int w = kittens::warpid() % LANE_WARPS;
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
            warp::load(x, xrow[j]);
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
        if (kittens::warpid() % LANE_WARPS == 0 && warp::laneid() == 0) {
            scratchy::ops::gmem_counter_wait(g.counters, {0, 0, 0, ins[3]}, uint32_t(ins[4]));
        }
        lane_sync();
        async_proxy_fence();
    }
}

// M0-A's trace: stamp `k` of instruction `i` of `worker` (its SM's `clock64`), by the lane's warp
// 0 lane 0. `trace_sync` ends a phase on every warp of the lane first; both are nothing when the
// launch has no trace, so an untraced play is the program's own timing.
__device__ inline void stamp(const mlp_globals& g, int worker, int i, int k) {
    if (g.trace_on && kittens::warpid() % LANE_WARPS == 0 && warp::laneid() == 0) {
        g.trace[{0, worker, i, k}] = uint64_t(clock64());
    }
}
__device__ inline void trace_sync(const mlp_globals& g) {
    if (g.trace_on) {
        lane_sync();
    }
}

// The row `src[layer]`'s `chunks` TMA-loaded into `dst` by the lane's warp 0, on `sem` (its
// `uses`-th use), and waited for by every warp: the down's activation, and M0-A's precomputed rows.
template <typename GL, int N>
__device__ inline void load_row(const GL& src, int layer, row_sv (&dst)[N], int chunks,
                                semaphore& sem, int uses) {
    if (kittens::warpid() % LANE_WARPS == 0) {
        warp::tma::expect_bytes(sem, uint32_t(chunks * sizeof(row_sv)));
        for (int c = 0; c < chunks; c++) {
            warp::tma::load_async(dst[c], src, {0, layer, 0, c}, sem);
        }
    }
    wait(sem, uses % 2);
}

// A lane's 8 consumer warps at variant `V` (0: the M0 program), over its buffers `sm` (`mlp_smem`,
// or `lane_smem` under `TWO_LANES`) and its worker index.
template <int V, typename SM>
__device__ inline void mlp_consume(const mlp_globals& g, const ring_state& r, const mlp_sems& se,
                                   const SM& sm, int worker) {
    constexpr bool PRE_NORM = (V & (NORM_ONCE | QUANT_ONCE)) != 0;
    constexpr bool PRE_QUANT = (V & QUANT_ONCE) != 0;
    constexpr bool WAITS = (V & NO_WAIT) == 0;
    constexpr int GUR = (V & FULL_WAVE) ? GU_ROWS_FULL : GU_ROWS;
    const int w = kittens::warpid() % LANE_WARPS;
    // Each warp's weight-ring loads so far (the ring's running index), the lane's norms and its
    // row loads so far (the semaphores' phases), and its instruction index (the trace's row).
    int j0 = 0;
    int norms = 0;
    int rows_loaded = 0;
    int i = 0;
    play<LANE_WARPS>(
        g.n, r, LANE_BARRIER + lane_of(),
        [&](const instruction_sv& ins) {
            const int op = ins[0], layer = ins[1], block = ins[2];
            stamp(g, worker, i, T_START);
            if (op == OP_GATE_UP) {
                const int row0 = (block * LANE_WARPS + w) * GUR;
                float a_scale = 0.0f;
                float gate[GUR], up[GUR];
                dots<GUR>(
                    g.gate, sm, se, layer, row0, INTER, PRE_QUANT ? sm.pre : sm.row, GU_CHUNKS, j0,
                    [&] {
                        trace_sync(g);
                        stamp(g, worker, i, T_PRIMED);
                        if constexpr (WAITS) {
                            lane_await(g, ins);
                        }
                        trace_sync(g);
                        stamp(g, worker, i, T_WAITED);
                        if constexpr (PRE_QUANT) {
                            load_row(g.qrow, layer, sm.pre, GU_CHUNKS, se.act, rows_loaded);
                            rows_loaded += 1;
                            a_scale = g.qscale[{0, 0, layer, 0}];
                            stamp(g, worker, i, T_NORMED);
                        } else {
                            if constexpr (PRE_NORM) {
                                load_row(g.nrow, layer, sm.pre, GU_CHUNKS, se.act, rows_loaded);
                                rows_loaded += 1;
                                trace_sync(g);
                                stamp(g, worker, i, T_NORMED);
                                a_scale = quantize_row(sm.pre, sm, se, GU_CHUNKS);
                            } else {
                                norm_row(g, sm, se, layer, norms);
                                stamp(g, worker, i, T_NORMED);
                                a_scale = quantize_row(sm.row, sm, se, GU_CHUNKS);
                            }
                        }
                        stamp(g, worker, i, T_QUANTIZED);
                    },
                    gate);
                dots<GUR>(g.up, sm, se, layer, row0, INTER, PRE_QUANT ? sm.pre : sm.row, GU_CHUNKS,
                          j0 + GU_CHUNKS, [] {}, up);
                trace_sync(g);
                stamp(g, worker, i, T_DOTS);
                const int rows = max(0, min(GUR, INTER - row0));
#pragma unroll
                for (int rr = 0; rr < GUR; rr++) {
                    if (warp::laneid() == 0 && rr < rows) {
                        const float gv = scaled(g.gate_scale, layer, row0 + rr, gate[rr], a_scale);
                        const float uv = scaled(g.up_scale, layer, row0 + rr, up[rr], a_scale);
                        g.act[{0, layer, 0, row0 + rr}] = base_types::convertor<bf16, float>::convert(
                            scratchy::ops::gelu_tanh_mul::op<float>(gv, uv));
                    }
                }
                j0 += rows > 0 ? 2 * GU_CHUNKS : 0;
                norms += PRE_NORM ? 0 : 1;
            } else if (op == OP_DOWN) {
                const int row0 = (block * LANE_WARPS + w) * DOWN_ROWS;
                float a_scale = 0.0f;
                float dot[DOWN_ROWS];
                dots<DOWN_ROWS>(
                    g.down, sm, se, layer, row0, HIDDEN, PRE_QUANT ? sm.pre : sm.row, DOWN_CHUNKS,
                    j0,
                    [&] {
                        trace_sync(g);
                        stamp(g, worker, i, T_PRIMED);
                        // Every gate/up block of the layer is stored before its row is read.
                        if constexpr (WAITS) {
                            lane_await(g, ins);
                        }
                        trace_sync(g);
                        stamp(g, worker, i, T_WAITED);
                        if constexpr (PRE_QUANT) {
                            load_row(g.qact, layer, sm.pre, DOWN_CHUNKS, se.act, rows_loaded);
                            rows_loaded += 1;
                            a_scale = g.qscale[{0, 0, layer, 1}];
                        } else {
                            load_row(g.act, layer, sm.act, DOWN_CHUNKS, se.act, rows_loaded);
                            rows_loaded += 1;
                            a_scale = quantize_row(sm.act, sm, se, DOWN_CHUNKS);
                        }
                        stamp(g, worker, i, T_NORMED);
                        stamp(g, worker, i, T_QUANTIZED);
                    },
                    dot);
                trace_sync(g);
                stamp(g, worker, i, T_DOTS);
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
            }
        },
        [&](const instruction_sv& ins) {
            stamp(g, worker, i, T_STORED);
            // The instruction's stores are done (the barrier): publish them once.
            if (ins[0] != OP_NOOP && w == 0 && warp::laneid() == 0) {
                publish(g.counters, ins[5], 1);
            }
            stamp(g, worker, i, T_ARRIVED);
            i += 1;
        });
}

// The variant's lanes per SM and threads per CTA (8 consumer warps and a controller per lane).
template <int V>
constexpr int mlp_lanes() {
    return (V & TWO_LANES) ? 2 : 1;
}
template <int V>
constexpr int mlp_threads() {
    return mlp_lanes<V>() * MLP_WARPS * kittens::WARP_THREADS;
}

template <int V>
__global__ __launch_bounds__(mlp_threads<V>(), 1) void mk_mlp_ker(const __grid_constant__ mlp_globals g) {
    constexpr int LN = mlp_lanes<V>();
    static_assert(!(V & TWO_LANES) || (V & QUANT_ONCE), "two lanes only without the norm's tile");
    __shared__ sv<int, INSTRUCTION_WIDTH> ring[LN][STAGES];
    __shared__ semaphore arrived[LN][STAGES], finished[LN][STAGES];
    __shared__ semaphore norm_arrived[4][2], owned, act[LN], weights[LN][LANE_WARPS][GEMV_STAGES];
    __shared__ sv_fl<16> amax[LN];
    if (threadIdx.x == 0) {
        for (int l = 0; l < LN; l++) {
            for (int s = 0; s < STAGES; s++) {
                init_semaphore(arrived[l][s], 0, 1);
                init_semaphore(finished[l][s], LANE_WARPS, 0);
            }
            init_semaphore(act[l], 0, 1);
            for (int w = 0; w < LANE_WARPS; w++) {
                for (int s = 0; s < GEMV_STAGES; s++) {
                    init_semaphore(weights[l][w][s], 0, 1);
                }
            }
        }
        for (int w = 0; w < 4; w++) {
            init_semaphore(norm_arrived[w][0], 0, 1);
            init_semaphore(norm_arrived[w][1], 0, 1);
        }
        init_semaphore(owned, 0, 1);
    }
    // Every semaphore is initialised before any thread waits on it or a TMA arrives on it.
    __syncthreads();
    const int w = kittens::warpid();
    if (w < LN * LANE_WARPS) {
        const int lane = w / LANE_WARPS;
        const ring_state r{ring[lane], arrived[lane], finished[lane]};
        const mlp_sems se{norm_arrived, owned, act[lane], weights[lane], amax[lane]};
        const int worker = LN * int(blockIdx.x) + lane;
        if constexpr (LN == 1) {
            mlp_consume<V>(g, r, se, mlp_pages(), worker);
        } else if (lane == 0) {
            mlp_consume<V>(g, r, se, lane_pages<0>(), worker);
        } else {
            mlp_consume<V>(g, r, se, lane_pages<1>(), worker);
        }
    } else if (w < LN * LANE_WARPS + LN) {
        const int lane = w - LN * LANE_WARPS;
        const ring_state r{ring[lane], arrived[lane], finished[lane]};
        control(g.instructions, g.n, LN * int(blockIdx.x) + lane, r);
    }
    // No warp leaves while another may still arrive on a semaphore it waits on.
    __syncthreads();
}

// M0-A's chain outside the megakernel (test support): one plain launch of the gate/up GEMV + GeLU
// over a normed row, `GUR` rows a warp (4: today's 66 CTAs; 2: the full wave's 132), each CTA the
// body of `OP_GATE_UP` under `NORM_ONCE` (the row TMA-loaded, `quantize_row`, the gate's and the
// up's dots, the stores): today's `gate_up_gemv_fp8_ker<false, gelu_tanh_mul>` per CTA. `layer` is
// the layer of `g`'s weights, `nrow` and `act` it reads and writes.
constexpr int PLAIN_SMEM_BYTES =
    int((2 * ROW_CHUNKS + DOWN_CHUNKS) * sizeof(row_sv) +
        LANE_WARPS * GEMV_STAGES * GU_ROWS * sizeof(w_sv)) +
    4 * 1024;

constexpr int PLAIN_THREADS = LANE_WARPS * kittens::WARP_THREADS;

// The plain launch's layer rides in `g.n` (its one parameter is the globals, as every kernel's).
template <int GUR>
__global__ __launch_bounds__(PLAIN_THREADS, 1) void mk_gate_up_plain_ker(
    const __grid_constant__ mlp_globals g) {
    const int layer = g.n;
    __shared__ semaphore norm_arrived[4][2], owned, act, weights[LANE_WARPS][GEMV_STAGES];
    __shared__ sv_fl<16> amax;
    if (threadIdx.x == 0) {
        init_semaphore(act, 0, 1);
        for (int w = 0; w < LANE_WARPS; w++) {
            for (int s = 0; s < GEMV_STAGES; s++) {
                init_semaphore(weights[w][s], 0, 1);
            }
        }
    }
    __syncthreads();
    const lane_smem sm = lane_pages<0>();
    const mlp_sems se{norm_arrived, owned, act, weights, amax};
    const int w = kittens::warpid();
    const int row0 = (int(blockIdx.x) * LANE_WARPS + w) * GUR;
    float a_scale = 0.0f;
    float gate[GUR], up[GUR];
    dots<GUR>(
        g.gate, sm, se, layer, row0, INTER, sm.row, GU_CHUNKS, 0,
        [&] {
            load_row(g.nrow, layer, sm.pre, GU_CHUNKS, se.act, 0);
            a_scale = quantize_row(sm.pre, sm, se, GU_CHUNKS);
        },
        gate);
    dots<GUR>(g.up, sm, se, layer, row0, INTER, sm.row, GU_CHUNKS, GU_CHUNKS, [] {}, up);
    const int rows = max(0, min(GUR, INTER - row0));
#pragma unroll
    for (int rr = 0; rr < GUR; rr++) {
        if (warp::laneid() == 0 && rr < rows) {
            const float gv = scaled(g.gate_scale, layer, row0 + rr, gate[rr], a_scale);
            const float uv = scaled(g.up_scale, layer, row0 + rr, up[rr], a_scale);
            g.act[{0, layer, 0, row0 + rr}] = base_types::convertor<bf16, float>::convert(
                scratchy::ops::gelu_tanh_mul::op<float>(gv, uv));
        }
    }
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
    // M0-A only (null in the M0 entry).
    const void* nrow = nullptr;
    const void* qrow = nullptr;
    const void* qact = nullptr;
    const void* qscale = nullptr;
    void* trace = nullptr;
};

// The variants the ablation entry plays (M0-A), by the flags' value.
constexpr int V1 = NORM_ONCE;
constexpr int V2 = NORM_ONCE | QUANT_ONCE;
constexpr int V3 = NO_WAIT;
constexpr int V4 = FULL_WAVE;
constexpr int V5 = NORM_ONCE | QUANT_ONCE | TWO_LANES;
constexpr int V6 = NORM_ONCE | QUANT_ONCE | FULL_WAVE | TWO_LANES;

// The MLP program's views over `a`, `layers` and `workers` instruction rows. Can throw.
inline mlp_globals mlp_views(const mlp_args& a, uint32_t n, uint32_t counter_count,
                             uint32_t layers, uint32_t workers, float eps, float gain_offset) {
    const auto bf = [](const void* p) { return static_cast<bf16*>(const_cast<void*>(p)); };
    const auto e4 = [](const void* p) { return static_cast<fp8e4m3*>(const_cast<void*>(p)); };
    const auto f = [](const void* p) { return static_cast<float*>(const_cast<void*>(p)); };
    const size_t L = layers;
    // The views of absent M0-A buffers point at `h` (never read): a view needs an address.
    const auto or_h = [&](const void* p) { return p != nullptr ? p : a.h; };
    instruction_gl ig{static_cast<int*>(const_cast<void*>(a.instructions)), nullptr,
                      size_t(workers), size_t(n), nullptr};
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
    x_gl ng{bf(or_h(a.nrow)), nullptr, L, nullptr, nullptr};
    x_gl qg{bf(or_h(a.qrow)), nullptr, L, nullptr, nullptr};
    act_gl qa{bf(or_h(a.qact)), nullptr, L, nullptr, nullptr};
    qscale_gl qs{f(or_h(a.qscale)), nullptr, nullptr, L, nullptr};
    trace_gl tg{static_cast<uint64_t*>(a.trace != nullptr ? a.trace : a.h), nullptr,
                size_t(workers), size_t(n), nullptr};
    return mlp_globals{ig, cg, xg, yg, gg, gate_g, up_g, down_g, gs, us, ds, ag,
                       ng, qg, qa, qs, tg, a.trace != nullptr ? 1 : 0, eps, gain_offset, int(n)};
}

// Launch variant `V`'s cooperative grid.
template <int V>
int launch_mlp(const mlp_globals& G, void* stream) {
    mlp_globals g = G;
    void* args[] = {&g};
    const dim3 grid{uint32_t(SMS)}, block{uint32_t(mlp_threads<V>())};
    return static_cast<int>(cudaLaunchCooperativeKernel(mk_mlp_ker<V>, grid, block, args,
                                                        MK_SMEM_BYTES,
                                                        static_cast<cudaStream_t>(stream)));
}

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
             cudaFuncSetAttribute(mk_mlp_ker<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V5>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_mlp_ker<V6>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  MK_SMEM_BYTES),
             cudaFuncSetAttribute(mk_gate_up_plain_ker<GU_ROWS>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, PLAIN_SMEM_BYTES),
             cudaFuncSetAttribute(mk_gate_up_plain_ker<GU_ROWS_FULL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, PLAIN_SMEM_BYTES),
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
        const mlp_args a{instructions, counters, h,  gain, gate, gate_scale,
                         up,           up_scale, down, down_scale, act};
        return launch_mlp<0>(mlp_views(a, n, counter_count, layers, SMS, eps, gain_offset), stream);
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/megakernel.rs`), M0-A's ablation: `scratchy_cuda_mk_mlp_test` at variant
// `variant` (0, or one of `V1`..`V6`), the lists `[workers, n, 32]` (`workers` 132, or 264 under
// `TWO_LANES`), with the precomputed rows the variant reads (`nrow [layers, 2816]` bf16 the normed
// rows; `qrow [layers, 2816]`, `qact [layers, 2112]` bf16 the quantized rows' e4m3 values;
// `qscale [layers, 2]` f32 their scales) and, when `trace` is not null, `[workers, n, 8]` u64 of
// `clock64` stamps. A null row the variant reads, or another variant, is refused, as are the
// shapes `scratchy_cuda_mk_mlp_test` refuses.
extern "C" int scratchy_cuda_mk_mlp_ablation(
    uint32_t variant, const void* instructions, uint32_t n, void* counters, uint32_t counter_count,
    void* h, const void* gain, const void* gate, const void* gate_scale, const void* up,
    const void* up_scale, const void* down, const void* down_scale, void* act, uint32_t layers,
    float eps, float gain_offset, const void* nrow, const void* qrow, const void* qact,
    const void* qscale, void* trace, void* stream /* cudaStream_t; null = default */) {
    const int v = int(variant);
    const bool known = v == 0 || v == V1 || v == V2 || v == V3 || v == V4 || v == V5 || v == V6;
    const bool rows = (!(v & NORM_ONCE) || (v & QUANT_ONCE) || nrow != nullptr) &&
                      (!(v & QUANT_ONCE) || (qrow != nullptr && qact != nullptr && qscale != nullptr));
    if (!known || !rows || n == 0 || n > uint32_t(INT_MAX) || layers == 0 ||
        layers >= uint32_t(INT_MAX) || counter_count == 0 || counter_count > uint32_t(INT_MAX) ||
        !(eps > 0.0f && eps <= 3.0e38f)) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    if (const int e = check_sms(); e != 0) {
        return e;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        const uint32_t workers = (v & TWO_LANES) ? uint32_t(2 * SMS) : uint32_t(SMS);
        const mlp_args a{instructions, counters, h,    gain, gate, gate_scale, up,  up_scale,
                         down,         down_scale, act, nrow, qrow, qact,       qscale, trace};
        const mlp_globals G = mlp_views(a, n, counter_count, layers, workers, eps, gain_offset);
        switch (v) {
            case V1:
                return launch_mlp<V1>(G, stream);
            case V2:
                return launch_mlp<V2>(G, stream);
            case V3:
                return launch_mlp<V3>(G, stream);
            case V4:
                return launch_mlp<V4>(G, stream);
            case V5:
                return launch_mlp<V5>(G, stream);
            case V6:
                return launch_mlp<V6>(G, stream);
            default:
                return launch_mlp<0>(G, stream);
        }
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}

// Test support (`tests/megakernel.rs`), M0-A's chain outside the megakernel: one plain launch of
// `mk_gate_up_plain_ker` for layer `layer` of `layers`, at 66 CTAs of 4 rows a warp or, with
// `full_wave`, 132 of 2: `act[layer] = gelu_tanh_mul(gate(q), up(q))`, `q` the quantized
// `nrow[layer]`. Refused (SCRATCHY_CUDA_ERR_SHAPE) when `layer` is not below `layers` or `layers`
// is zero or at INT_MAX. Asynchronous on `stream`.
extern "C" int scratchy_cuda_mk_gate_up_plain(uint32_t full_wave, const void* gate,
                                              const void* gate_scale, const void* up,
                                              const void* up_scale, const void* nrow, void* act,
                                              uint32_t layers, uint32_t layer,
                                              void* stream /* cudaStream_t; null = default */) {
    if (layers == 0 || layers >= uint32_t(INT_MAX) || layer >= layers) {
        return SCRATCHY_CUDA_ERR_SHAPE;
    }
    // ThunderKittens' host code can throw; an exception must never unwind into Rust.
    try {
        // Only the gate/up's views are read; the others view `act` (a view needs an address).
        mlp_args a{};
        a.instructions = act;
        a.counters = act;
        a.h = act;
        a.gain = act;
        a.gate = gate;
        a.gate_scale = gate_scale;
        a.up = up;
        a.up_scale = up_scale;
        a.down = gate;
        a.down_scale = gate_scale;
        a.act = act;
        a.nrow = nrow;
        mlp_globals G = mlp_views(a, 1, 1, layers, 1, 1.0f, 0.0f);
        G.n = int(layer);
        void* args[] = {&G};
        const dim3 block{uint32_t(PLAIN_THREADS)};
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (full_wave) {
            const dim3 grid{uint32_t(GU_BLOCKS_FULL)};
            return static_cast<int>(cudaLaunchKernel(mk_gate_up_plain_ker<GU_ROWS_FULL>, grid,
                                                     block, args, PLAIN_SMEM_BYTES, s));
        }
        const dim3 grid{uint32_t(GU_BLOCKS)};
        return static_cast<int>(cudaLaunchKernel(mk_gate_up_plain_ker<GU_ROWS>, grid, block, args,
                                                 PLAIN_SMEM_BYTES, s));
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
