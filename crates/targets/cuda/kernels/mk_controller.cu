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
//   - A CTA is 20 warps. Warps 0..15 are the consumers, as two LANES of 8: lane `L` (warps
//     `8L .. 8L + 7`) plays its own instruction list, one instruction at a time, each instruction
//     the work of one of today's 8-warp CTAs (M0's opcodes, Day 2). Warp 16 + L is lane L's
//     controller. Warps 18 and 19 are reserved (Day 2's weight prefetch); they only take part in
//     the start and the end.
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
// THE OPCODES (M0 Day 1: the interpreter's own tests; Day 2 adds the MLP block's two):
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

// Lane `LANE`'s consumers: instruction `i` from stage `i % 2`, dispatched by its opcode.
template <int LANE>
__device__ inline void consume(const mk_globals& g, instruction_sv (&ring)[STAGES],
                               semaphore (&arrived)[STAGES], semaphore (&finished)[STAGES],
                               semaphore& echoed) {
    const int lw = kittens::warpid() % LANE_WARPS;
    int echoes = 0;
    for (int i = 0; i < g.n; i++) {
        const int s = i % STAGES;
        wait(arrived[s], (i / STAGES) % 2);
        const instruction_sv& ins = ring[s];
        const int op = ins[0];
        if (lw == 0 && op != OP_NOOP) {
            if (op == OP_CHAIN) {
                op_chain(g, ins);
            } else if (op == OP_PAGE_ECHO) {
                op_page_echo<LANE>(g, ins, echoed, echoes);
            }
        }
        echoes += op == OP_PAGE_ECHO ? 1 : 0;
        // Every warp of the lane is done with the stage before the controller refills it.
        group<LANE_WARPS>::sync(LANE_BARRIER + LANE);
        if (warp::laneid() == 0) {
            arrive(finished[s]);
        }
    }
}

// Lane `LANE`'s controller: instruction `i` of its worker into stage `i % 2`, once the stage's
// last instruction is done.
template <int LANE>
__device__ inline void control(const mk_globals& g, instruction_sv (&ring)[STAGES],
                               semaphore (&arrived)[STAGES], semaphore (&finished)[STAGES]) {
    const int worker = LANES * int(blockIdx.x) + LANE;
    for (int i = 0; i < g.n; i++) {
        const int s = i % STAGES;
        if (i >= STAGES) {
            wait(finished[s], (i / STAGES - 1) % 2);
        }
        warp::tma::expect_bytes(arrived[s], uint32_t(sizeof(instruction_sv)));
        warp::tma::load_async(ring[s], g.instructions, {0, worker, i, 0}, arrived[s]);
    }
}

__global__ __launch_bounds__(NUM_THREADS, 1) void mk_ker(const __grid_constant__ mk_globals g) {
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
    if (w < LANE_WARPS) {
        consume<0>(g, ring0, arrived0, finished0, echoed0);
    } else if (w < CONSUMER_WARPS) {
        consume<1>(g, ring1, arrived1, finished1, echoed1);
    } else if (w == CONTROLLER_WARP) {
        control<0>(g, ring0, arrived0, finished0);
    } else if (w == CONTROLLER_WARP + 1) {
        control<1>(g, ring1, arrived1, finished1);
    }
    // No warp leaves while another may still arrive on a semaphore it waits on.
    __syncthreads();
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
    return static_cast<int>(
        cudaFuncSetAttribute(mk_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, MK_SMEM_BYTES));
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
    int device = 0, sms = 0;
    if (const cudaError_t e = cudaGetDevice(&device); e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (const cudaError_t e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        e != cudaSuccess) {
        return static_cast<int>(e);
    }
    if (sms != SMS) {
        return SCRATCHY_CUDA_ERR_SHAPE;
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
        return static_cast<int>(cudaLaunchCooperativeKernel(mk_ker, grid, block, args,
                                                            MK_SMEM_BYTES,
                                                            static_cast<cudaStream_t>(stream)));
    } catch (...) {
        return SCRATCHY_CUDA_ERR_HOST;
    }
}
