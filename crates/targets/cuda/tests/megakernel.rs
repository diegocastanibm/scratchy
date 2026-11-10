// SPDX-License-Identifier: Apache-2.0
//! The megakernel's interpreter (FP8 S8 M0, `kernels/mk_controller.cu`) on the H100: its
//! persistent loop, instruction ring, pages and cross-SM counters, played from hand-written
//! instruction lists (M0 builds them in the test; M1 has `#[forward]` emit them).
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips. Every test arms
//! the watchdog (`common::watchdog`), so a counter wait that never ends fails its test instead of
//! hanging the binary.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod fp8;
    pub mod numerics;
    pub mod watchdog;
}

use common::numerics::{bf16_inputs, mismatched_bits};
use scratchy_target_cuda::kernels::{
    KernelError, MK_INSTRUCTION_WIDTH, MK_MLP_DOWN_BLOCKS, MK_MLP_GATE_UP_BLOCKS, MK_MLP_HIDDEN,
    MK_MLP_INTER, MK_MLP_WORKERS, MK_OP_CHAIN, MK_OP_DOWN, MK_OP_GATE_UP, MK_OP_NOOP,
    MK_OP_PAGE_ECHO, MK_PAGE_FLOATS, MK_WORKERS, MkMlpWeights, gate_up_gemv_fp8, gemv_fp8_residual,
    mk_mlp_test, mk_test, rms_norm_bf16,
};
use scratchy_target_cuda::kernels::{
    MK_MLP_FULL_WAVE, MK_MLP_NO_WAIT, MK_MLP_NORM_ONCE, MK_MLP_QUANT_ONCE, MK_MLP_TWO_LANES,
    MK_TRACE_WIDTH, MkMlpRows, mk_gate_up_plain, mk_mlp_ablation,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::GatedOp;

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open().expect("an H100: these tests run only where one is (GKE)")
}

/// `[workers, n, MK_INSTRUCTION_WIDTH]` i32, every instruction `OP_NOOP`.
struct Lists {
    n: usize,
    words: Vec<i32>,
}

impl Lists {
    fn new(n: usize) -> Self {
        Self::of(MK_WORKERS, n)
    }

    fn of(workers: usize, n: usize) -> Self {
        Self {
            n,
            words: vec![MK_OP_NOOP; workers * n * MK_INSTRUCTION_WIDTH],
        }
    }

    /// Worker `w`'s instruction `i`: `op` and its argument words.
    fn set(&mut self, w: usize, i: usize, op: i32, args: &[i32]) {
        let at = (w * self.n + i) * MK_INSTRUCTION_WIDTH;
        self.words[at] = op;
        self.words[at + 1..at + 1 + args.len()].copy_from_slice(args);
    }

    fn bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }
}

fn i32s(bytes: &[u16]) -> Vec<i32> {
    (bytes.as_chunks::<2>().0.iter())
        .map(|p| i32::from(p[0]) | (i32::from(p[1]) << 16))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bits: &[u16]) -> Vec<f32> {
    (bits.as_chunks::<2>().0.iter())
        .map(|p| f32::from_bits(u32::from(p[0]) | (u32::from(p[1]) << 16)))
        .collect()
}

/// One play of `lists` over fresh zeroed counters and chain, and `x` (`rows` rows); returns the
/// chain and `out`.
fn play(
    device: &CudaDevice,
    lists: &Lists,
    (counters, chain): (usize, usize),
    (x, rows): (&[f32], usize),
) -> Result<(Vec<i32>, Vec<f32>), KernelError> {
    let up = |b: &[u8]| DeviceBuffer::upload_bytes(device, b).expect("upload");
    let ins = up(&lists.bytes());
    let cb = up(&vec![0u8; 4 * counters]);
    let chb = up(&vec![0u8; 4 * chain]);
    let xb = up(&f32_bytes(x));
    let ob = up(&f32_bytes(&vec![f32::NAN; rows * MK_PAGE_FLOATS]));
    // SAFETY: every buffer holds its count of elements; every index the lists name is inside its
    // buffer, and every count waited for is published (the tests build them so); null is the
    // default stream, which the downloads wait for.
    unsafe {
        mk_test(
            (ins.ptr().cast_const(), lists.n as u32),
            (cb.ptr(), counters as u32),
            (chb.ptr(), chain as u32),
            (xb.ptr().cast_const(), ob.ptr(), rows as u32),
            std::ptr::null_mut(),
        )
    }?;
    Ok((
        i32s(&chb.download().expect("chain")),
        f32s(&ob.download().expect("out")),
    ))
}

/// M0 Day 1: a chain over every worker of every SM, in an order that crosses SMs at every link
/// (worker `w`'s link is worker `perm(w)`'s successor, `perm` a stride of 67 mod 264), several
/// links per worker, each waiting on the one before it through a counter. `chain[j] = j + 1` for
/// every link proves the ring delivered each instruction, the dispatch ran it, every acquire saw
/// the release before it, and nothing ran twice; a missed publish hangs, and the watchdog fails
/// the test. Then each lane's pages: every worker echoes a row of its own through each of its 6
/// pages, and `out` is `x`, bit for bit. Then the refusals.
#[test]
fn the_interpreter_plays_a_cross_sm_chain_and_its_pages() {
    let device = h100();
    let mut pass = true;
    // The chain: link `j` (0..LINKS) runs on worker `order[j % W]`, as its instruction `j / W`.
    const PER: usize = 3;
    let links = MK_WORKERS * PER;
    let order: Vec<usize> = (0..MK_WORKERS).map(|w| (w * 67) % MK_WORKERS).collect();
    let mut seen = vec![false; MK_WORKERS];
    order.iter().for_each(|&w| seen[w] = true);
    assert!(
        seen.iter().all(|&s| s),
        "67 is coprime with 264: every worker once"
    );
    let mut lists = Lists::new(PER);
    for j in 0..links {
        let (w, i) = (order[j % MK_WORKERS], j / MK_WORKERS);
        // Wait for counter `j − 1` (link j − 1 published it), read chain[j − 1], write chain[j],
        // publish counter `j`. Link 0 waits for nothing.
        let (wait_idx, want, read) = match j {
            0 => (0, 0, 0),
            _ => (j as i32 - 1, 1, j as i32 - 1),
        };
        lists.set(
            w,
            i,
            MK_OP_CHAIN,
            &[wait_idx, want, read, j as i32, j as i32],
        );
    }
    let rows = 1;
    let x = vec![0.0f32; rows * MK_PAGE_FLOATS];
    match play(&device, &lists, (links, links), (&x, rows)) {
        Ok((chain, _)) => {
            let bad = (chain.iter().enumerate())
                .filter(|&(j, &v)| v != j as i32 + 1)
                .count();
            eprintln!(
                "M0 interpreter chain: {links} links over {MK_WORKERS} workers, bad_links={bad} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            pass &= bad == 0;
        }
        Err(e) => {
            eprintln!("M0 interpreter chain: launch failed: {e} FAIL");
            pass = false;
        }
    }
    // The pages: worker `w` echoes row `6w + p` through its page `p`, for p in 0..6, one
    // instruction each, the lane's two workers per SM running at once.
    const PAGES: usize = 6;
    let rows = MK_WORKERS * PAGES;
    let x: Vec<f32> = (0..rows * MK_PAGE_FLOATS)
        .map(|i| (i as f32) * 0.5 - 7.25)
        .collect();
    let mut lists = Lists::new(PAGES);
    for w in 0..MK_WORKERS {
        for p in 0..PAGES {
            let row = (w * PAGES + p) as i32;
            lists.set(w, p, MK_OP_PAGE_ECHO, &[row, p as i32, row]);
        }
    }
    match play(&device, &lists, (1, 1), (&x, rows)) {
        Ok((_, out)) => {
            let bad = (out.iter().zip(&x))
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "M0 interpreter pages: {rows} rows through {PAGES} pages per lane, \
                 mismatched_bits={bad} {}",
                if bad == 0 { "PASS" } else { "FAIL" }
            );
            pass &= bad == 0;
        }
        Err(e) => {
            eprintln!("M0 interpreter pages: launch failed: {e} FAIL");
            pass = false;
        }
    }
    // The refusals, before any pointer is touched.
    let null: *mut std::ffi::c_void = std::ptr::null_mut();
    for dims in [[0u32, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]] {
        // SAFETY: the launcher refuses these counts before it touches a pointer.
        let got = unsafe {
            mk_test(
                (null.cast_const(), dims[0]),
                (null, dims[1]),
                (null, dims[2]),
                (null.cast_const(), null, dims[3]),
                null,
            )
        };
        let ok = got == Err(KernelError::MkShape { dims });
        eprintln!(
            "M0 interpreter refuses {dims:?}: {got:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        pass &= ok;
    }
    assert!(pass, "the interpreter failed: see the lines above");
}

// ── M0 Day 2: Gemma's dense-MLP block, the go/no-go ────────────────

/// Gemma 4 26B-A4B's pre-feedforward norm: eps, and its `Scale` gain convention's offset.
const MLP_EPS: f32 = 1e-6;
const MLP_GAIN_OFFSET: f32 = 0.0;
/// Gemma's dense MLP activation, GeGLU (`GatedOp::Gelu`).
const MLP_ACT: GatedOp = GatedOp::Gelu;

/// `layers` of synthetic weights, on the device: random bf16 gains and inputs, random e4m3 bytes
/// (no NaN) and positive per-channel f32 scales, so each layer's output stays in range.
struct Mlp {
    layers: usize,
    h0: Vec<u16>,
    gain: DeviceBuffer,
    gate: (DeviceBuffer, DeviceBuffer),
    up: (DeviceBuffer, DeviceBuffer),
    down: (DeviceBuffer, DeviceBuffer),
}

/// Random e4m3 bytes, never a NaN (`0x7F`/`0xFF`): the sign and every exponent and mantissa.
fn e4m3_bytes(len: usize, seed: u64) -> Vec<u8> {
    (bf16_inputs(len, seed).into_iter().enumerate())
        .map(|(i, b)| {
            let v = (b as u8) ^ (i as u8).wrapping_mul(37);
            if v & 0x7F == 0x7F { v & 0xFE } else { v }
        })
        .collect()
}

/// Positive per-channel scales near `1 / (sqrt(k) · 16)`, so a row's products sum to O(1).
fn scales(len: usize, k: usize, seed: u64) -> Vec<u8> {
    let base = 1.0 / ((k as f32).sqrt() * 16.0);
    (bf16_inputs(len, seed).into_iter())
        .map(|b| base * (1.25 + 0.5 * half::bf16::from_bits(b).to_f32()))
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn mlp(device: &CudaDevice, layers: usize, seed: u64) -> Mlp {
    let (h, i) = (MK_MLP_HIDDEN, MK_MLP_INTER);
    let up = |b: &[u8]| DeviceBuffer::upload_bytes(device, b).expect("upload");
    let gain: Vec<u16> = (bf16_inputs(layers * h, seed ^ 1).into_iter())
        .map(|b| half::bf16::from_f32(1.0 + 0.25 * half::bf16::from_bits(b).to_f32()).to_bits())
        .collect();
    let gain_bytes: Vec<u8> = gain.iter().flat_map(|b| b.to_le_bytes()).collect();
    Mlp {
        layers,
        h0: bf16_inputs(h, seed ^ 2),
        gain: up(&gain_bytes),
        gate: (
            up(&e4m3_bytes(layers * i * h, seed ^ 3)),
            up(&scales(layers * i, h, seed ^ 4)),
        ),
        up: (
            up(&e4m3_bytes(layers * i * h, seed ^ 5)),
            up(&scales(layers * i, h, seed ^ 6)),
        ),
        down: (
            up(&e4m3_bytes(layers * h * i, seed ^ 7)),
            up(&scales(layers * h, i, seed ^ 8)),
        ),
    }
}

/// Byte offset `at` into `b`.
fn at(b: &DeviceBuffer, at: usize) -> *mut std::ffi::c_void {
    b.ptr().cast::<u8>().wrapping_add(at).cast()
}

/// Today's launch chain over every layer, on the default stream: per layer the one-row RMSNorm,
/// the fp8 gate/up GEMV + GeLU and the fp8 down GEMV with the residual epilogue, each at one split
/// (`gemv_splits` at these shapes), exactly as the decode tape launches them. `h` (`[layers + 1,
/// hidden]`) holds the input in row 0; returns every row.
fn chain(device: &CudaDevice, m: &Mlp, launch: &mut dyn FnMut(&mut dyn FnMut())) -> Vec<u16> {
    let (hd, i) = (MK_MLP_HIDDEN, MK_MLP_INTER);
    let mut h0 = m.h0.clone();
    h0.resize((m.layers + 1) * hd, 0);
    let h = DeviceBuffer::upload(device, &h0).expect("h");
    let n = DeviceBuffer::upload(device, &vec![0u16; hd]).expect("n");
    let act = DeviceBuffer::upload(device, &vec![0u16; i]).expect("act");
    let null = std::ptr::null_mut();
    launch(&mut || {
        for l in 0..m.layers {
            // SAFETY: every operand is its layer's slice of a buffer of its size; one split, so no
            // partials; null is the default stream.
            unsafe {
                rms_norm_bf16(
                    at(&h, 2 * l * hd).cast_const(),
                    at(&m.gain, 2 * l * hd).cast_const(),
                    n.ptr(),
                    (1, hd as u32),
                    MLP_EPS,
                    MLP_GAIN_OFFSET,
                    null,
                )
                .expect("norm");
                gate_up_gemv_fp8(
                    (
                        n.ptr().cast_const(),
                        (
                            at(&m.gate.0, l * i * hd).cast_const(),
                            at(&m.gate.1, 4 * l * i).cast_const(),
                        ),
                        (
                            at(&m.up.0, l * i * hd).cast_const(),
                            at(&m.up.1, 4 * l * i).cast_const(),
                        ),
                    ),
                    (act.ptr(), null),
                    (i as u32, hd as u32, 1),
                    MLP_ACT.code(),
                    null,
                )
                .expect("gate/up");
                gemv_fp8_residual(
                    (
                        act.ptr().cast_const(),
                        (
                            at(&m.down.0, l * hd * i).cast_const(),
                            at(&m.down.1, 4 * l * hd).cast_const(),
                        ),
                        at(&h, 2 * (l + 1) * hd),
                    ),
                    null,
                    (hd as u32, i as u32, 1),
                    at(&h, 2 * l * hd).cast_const(),
                    null,
                )
                .expect("down");
            }
        }
    });
    h.download().expect("h")
}

/// The MLP program's lists for `layers`: layer `L`'s 66 gate/up blocks, then its 176 down blocks,
/// dealt round robin over the 132 workers in that order, so each worker's list is in layer order.
/// A gate/up block of layer L > 0 waits for layer L − 1's 176 down blocks (counter `2L − 1`); a
/// down block of layer L for its 66 gate/up blocks (counter `2L`); each publishes its own.
fn mlp_lists(layers: usize) -> Lists {
    let mut per: Vec<Vec<[i32; 6]>> = vec![Vec::new(); MK_MLP_WORKERS];
    let mut next = 0usize;
    for l in 0..layers as i32 {
        for b in 0..MK_MLP_GATE_UP_BLOCKS as i32 {
            let (wait, want) = if l == 0 {
                (0, 0)
            } else {
                (2 * l - 1, MK_MLP_DOWN_BLOCKS as i32)
            };
            per[next % MK_MLP_WORKERS].push([MK_OP_GATE_UP, l, b, wait, want, 2 * l]);
            next += 1;
        }
        for b in 0..MK_MLP_DOWN_BLOCKS as i32 {
            per[next % MK_MLP_WORKERS].push([
                MK_OP_DOWN,
                l,
                b,
                2 * l,
                MK_MLP_GATE_UP_BLOCKS as i32,
                2 * l + 1,
            ]);
            next += 1;
        }
    }
    let n = per.iter().map(Vec::len).max().unwrap_or(1);
    let mut lists = Lists::of(MK_MLP_WORKERS, n);
    for (w, ins) in per.iter().enumerate() {
        for (i, a) in ins.iter().enumerate() {
            lists.set(w, i, a[0], &a[1..]);
        }
    }
    lists
}

/// The megakernel over every layer: returns `h`'s rows.
fn megakernel(device: &CudaDevice, m: &Mlp, launch: &mut dyn FnMut(&mut dyn FnMut())) -> Vec<u16> {
    let (hd, i) = (MK_MLP_HIDDEN, MK_MLP_INTER);
    let lists = mlp_lists(m.layers);
    let counters = 2 * m.layers;
    let mut h0 = m.h0.clone();
    h0.resize((m.layers + 1) * hd, 0);
    let h = DeviceBuffer::upload(device, &h0).expect("h");
    let act = DeviceBuffer::upload(device, &vec![0u16; m.layers * i]).expect("act");
    let ins = DeviceBuffer::upload_bytes(device, &lists.bytes()).expect("lists");
    let c = DeviceBuffer::upload_bytes(device, &vec![0u8; 4 * counters]).expect("counters");
    let w = MkMlpWeights {
        gain: m.gain.ptr().cast_const(),
        gate: (m.gate.0.ptr().cast_const(), m.gate.1.ptr().cast_const()),
        up: (m.up.0.ptr().cast_const(), m.up.1.ptr().cast_const()),
        down: (m.down.0.ptr().cast_const(), m.down.1.ptr().cast_const()),
    };
    launch(&mut || {
        // The counters start at zero for every play.
        // SAFETY: `c` holds `counters` u32; null is the default stream.
        unsafe { cuda_memset(c.ptr(), 4 * counters) };
        // SAFETY: every buffer holds what the lists index; every wait's count is published by the
        // lists (`mlp_lists`); null is the default stream.
        unsafe {
            mk_mlp_test(
                (ins.ptr().cast_const(), lists.n as u32),
                (c.ptr(), counters as u32),
                (h.ptr(), act.ptr()),
                w,
                (m.layers as u32, MLP_EPS, MLP_GAIN_OFFSET),
                std::ptr::null_mut(),
            )
        }
        .expect("megakernel");
    });
    h.download().expect("h")
}

unsafe extern "C" {
    fn cudaMemsetAsync(
        dst: *mut std::ffi::c_void,
        value: i32,
        bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
}

/// `cudaEvent_t`, declared here for the timing lines only (the runtime declares no event call).
type CudaEvent = *mut std::ffi::c_void;

/// Zero `bytes` bytes at `dst` on the default stream.
///
/// # Safety
///
/// `dst` a device pointer to at least `bytes` bytes.
unsafe fn cuda_memset(dst: *mut std::ffi::c_void, bytes: usize) {
    // SAFETY: the caller's contract.
    assert_eq!(
        unsafe { cudaMemsetAsync(dst, 0, bytes, std::ptr::null_mut()) },
        0,
        "memset"
    );
}

/// Run `f` once.
fn once(f: &mut dyn FnMut()) {
    f();
}

/// The bits gate (the user's decision 2026-11-10: the megakernel is bit-identical to today's
/// launch chain): Gemma's dense-MLP block on synthetic weights, 1 layer and 30 layers chained, every
/// output row of every layer `mismatched_bits=0` against the chain, at 3 seeds for 1 layer.
#[test]
fn the_megakernel_mlp_block_is_its_launch_chain_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (layers, seeds) in [(1usize, 3u64), (30, 1)] {
        for seed in 0..seeds {
            let m = mlp(&device, layers, 0x4D30 ^ seed << 8 ^ layers as u64);
            let want = chain(&device, &m, &mut once);
            let got = megakernel(&device, &m, &mut once);
            let hd = MK_MLP_HIDDEN;
            let bad = mismatched_bits(&got[hd..], &want[hd..]);
            let first = (1..=layers)
                .find(|&l| mismatched_bits(&got[l * hd..][..hd], &want[l * hd..][..hd]) != 0);
            let ok = bad == 0;
            eprintln!(
                "M0 mlp block layers={layers} seed={seed}: mismatched_bits={bad} first_bad_layer={first:?} {}",
                if ok { "PASS" } else { "FAIL" }
            );
            pass &= ok;
        }
    }
    assert!(
        pass,
        "the megakernel's MLP block differs from today's chain: see the lines above"
    );
}

/// The median µs of 20 plays of `f` after 3 warm-ups, each between two CUDA events on the default
/// stream (hot: the 30 layers' 0.53 GB of weights exceed L2, so each play streams them).
fn event_median(f: &mut dyn FnMut()) -> f64 {
    let (mut start, mut end): (CudaEvent, CudaEvent) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: plain runtime calls on the default stream with out-pointers to locals; each event is
    // destroyed below.
    unsafe {
        assert_eq!(cudaEventCreate(&mut start), 0, "cudaEventCreate");
        assert_eq!(cudaEventCreate(&mut end), 0, "cudaEventCreate");
    }
    let mut times = Vec::new();
    for i in 0..23 {
        let mut ms = 0.0f32;
        // SAFETY: as above.
        unsafe {
            assert_eq!(cudaEventRecord(start, std::ptr::null_mut()), 0, "record");
            f();
            assert_eq!(cudaEventRecord(end, std::ptr::null_mut()), 0, "record");
            assert_eq!(cudaEventSynchronize(end), 0, "synchronize");
            assert_eq!(cudaEventElapsedTime(&mut ms, start, end), 0, "elapsed");
        }
        if i >= 3 {
            times.push(f64::from(ms) * 1e3);
        }
    }
    // SAFETY: both events were created above and are no longer used.
    unsafe {
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Today's in-tape µs of the same launches, 30 layers, from the W4c Gemma trace at input 128
/// (/tmp/w4prof-an/an-gemma-128.md, decode token 64, `tok-node-128.sqlite`): per layer the
/// pre-feedforward `rms_norm_one_row_ker<0>` 8.46 + `gate_up_gemv_fp8_ker<false, gelu>` 24.09 +
/// `gemv_fp8_ker<false, 2>` 11.03 = 43.59 µs, × 30.
const TODAY_IN_TAPE_US: f64 = 1307.6;
/// The gate (the user's decision 2026-11-10): at least 25% faster.
const GATE_SPEEDUP: f64 = 0.25;

/// M0's go/no-go, recorded: the 30 layers in one persistent launch against today's chain of 90
/// launches, both timed here by CUDA events (the chain as stream launches, not a graph), and
/// against today's in-tape sum of the same kernels. `GATE M0 … PASS` when the megakernel is at
/// least 25% faster than the in-tape sum; else `FAIL`, and the megakernel is dropped. Recorded, not
/// asserted: the gate is read off the line.
#[test]
fn the_megakernel_mlp_time_is_recorded() {
    let device = h100();
    let m = mlp(&device, 30, 0x4D40);
    let mut chain_us = 0.0;
    chain(&device, &m, &mut |f| chain_us = event_median(f));
    let mut mk_us = 0.0;
    megakernel(&device, &m, &mut |f| mk_us = event_median(f));
    let speedup = 1.0 - mk_us / TODAY_IN_TAPE_US;
    eprintln!(
        "M0 mlp time 30 layers: megakernel {mk_us:.1} us; chain here {chain_us:.1} us (stream \
         launches); today's in-tape {TODAY_IN_TAPE_US:.1} us (recorded, not asserted)"
    );
    eprintln!(
        "GATE M0 megakernel {mk_us:.1} us vs in-tape {TODAY_IN_TAPE_US:.1} us: {:.1}% faster (want \
         >= {:.0}%) {}",
        100.0 * speedup,
        100.0 * GATE_SPEEDUP,
        if speedup >= GATE_SPEEDUP {
            "PASS"
        } else {
            "FAIL"
        }
    );
}

// ── M0-A: the ablation (timing only) ───────────────────────────────

/// The six ablation variants and M0's program (V0), as `(name, flags, bits kept)`: a variant keeps
/// the bits when every value it reads is the one M0's program computes in the same order.
const VARIANTS: [(&str, u32, bool); 7] = [
    ("V0", 0, true),
    ("V1_norm_once", MK_MLP_NORM_ONCE, true),
    ("V2_no_requant", MK_MLP_NORM_ONCE | MK_MLP_QUANT_ONCE, true),
    ("V3_no_barriers", MK_MLP_NO_WAIT, false),
    ("V4_full_wave", MK_MLP_FULL_WAVE, true),
    (
        "V5_two_lanes",
        MK_MLP_NORM_ONCE | MK_MLP_QUANT_ONCE | MK_MLP_TWO_LANES,
        true,
    ),
    (
        "V6_all",
        MK_MLP_NORM_ONCE | MK_MLP_QUANT_ONCE | MK_MLP_FULL_WAVE | MK_MLP_TWO_LANES,
        true,
    ),
];

/// M0-A's precomputed rows: per layer the normed row, the gate/up's and the down's quantized rows
/// (e4m3 values as bf16) and their scales, from today's own kernels over the chain's real inputs
/// (`h` and `act` per layer), so every variant reads what the program would compute; and the
/// chain's `h` and activations themselves (`NO_WAIT` starts with them in place).
struct Pre {
    h: Vec<u16>,
    acts: Vec<u16>,
    nrow: DeviceBuffer,
    qrow: DeviceBuffer,
    qact: DeviceBuffer,
    qscale: DeviceBuffer,
}

/// Today's chain over `m` again, keeping every layer's normed row and activation; then their
/// quantization on the host (`common::fp8::quantize_rows`, vLLM's formula, `quant_rows_fp8`'s bits
/// and so the GEMV prologue's, `the_prologue_quantizes_as_quant_rows_fp8`).
fn pre(device: &CudaDevice, m: &Mlp) -> Pre {
    use common::fp8::{e4m3_to_bf16_bits, quantize_rows};
    let (hd, i) = (MK_MLP_HIDDEN, MK_MLP_INTER);
    let h = chain(device, m, &mut once);
    let hb = DeviceBuffer::upload(device, &h).expect("h");
    let n = DeviceBuffer::upload(device, &vec![0u16; hd]).expect("n");
    let act = DeviceBuffer::upload(device, &vec![0u16; i]).expect("act");
    let null = std::ptr::null_mut();
    let (mut nrows, mut acts) = (Vec::new(), Vec::new());
    for l in 0..m.layers {
        // SAFETY: as `chain`'s.
        unsafe {
            rms_norm_bf16(
                at(&hb, 2 * l * hd).cast_const(),
                at(&m.gain, 2 * l * hd).cast_const(),
                n.ptr(),
                (1, hd as u32),
                MLP_EPS,
                MLP_GAIN_OFFSET,
                null,
            )
            .expect("norm");
            gate_up_gemv_fp8(
                (
                    n.ptr().cast_const(),
                    (
                        at(&m.gate.0, l * i * hd).cast_const(),
                        at(&m.gate.1, 4 * l * i).cast_const(),
                    ),
                    (
                        at(&m.up.0, l * i * hd).cast_const(),
                        at(&m.up.1, 4 * l * i).cast_const(),
                    ),
                ),
                (act.ptr(), null),
                (i as u32, hd as u32, 1),
                MLP_ACT.code(),
                null,
            )
            .expect("gate/up");
        }
        nrows.extend(n.download().expect("n"));
        acts.extend(act.download().expect("act"));
    }
    let (qn, sn) = quantize_rows(&nrows, m.layers, hd);
    let (qa, sa) = quantize_rows(&acts, m.layers, i);
    let scales: Vec<u8> = (sn.iter().zip(&sa))
        .flat_map(|(a, b)| [*a, *b])
        .flat_map(f32::to_le_bytes)
        .collect();
    let up = |v: &[u16]| DeviceBuffer::upload(device, v).expect("upload");
    Pre {
        h,
        acts: acts.clone(),
        nrow: up(&nrows),
        qrow: up(&e4m3_to_bf16_bits(&qn)),
        qact: up(&e4m3_to_bf16_bits(&qa)),
        qscale: DeviceBuffer::upload_bytes(device, &scales).expect("scales"),
    }
}

/// The variant's lists: `mlp_lists`, with no waits under `NO_WAIT`, at 132 gate/up blocks under
/// `FULL_WAVE`, and over 264 workers under `TWO_LANES`.
fn variant_lists(layers: usize, flags: u32) -> (Lists, usize) {
    let workers = if flags & MK_MLP_TWO_LANES != 0 {
        2 * MK_MLP_WORKERS
    } else {
        MK_MLP_WORKERS
    };
    let gu = if flags & MK_MLP_FULL_WAVE != 0 {
        2 * MK_MLP_GATE_UP_BLOCKS
    } else {
        MK_MLP_GATE_UP_BLOCKS
    };
    let waits = flags & MK_MLP_NO_WAIT == 0;
    let mut per: Vec<Vec<[i32; 6]>> = vec![Vec::new(); workers];
    let mut next = 0usize;
    for l in 0..layers as i32 {
        for b in 0..gu as i32 {
            let (wait, want) = match (waits, l) {
                (false, _) | (true, 0) => (0, 0),
                (true, _) => (2 * l - 1, MK_MLP_DOWN_BLOCKS as i32),
            };
            per[next % workers].push([MK_OP_GATE_UP, l, b, wait, want, 2 * l]);
            next += 1;
        }
        for b in 0..MK_MLP_DOWN_BLOCKS as i32 {
            let want = if waits { gu as i32 } else { 0 };
            per[next % workers].push([MK_OP_DOWN, l, b, 2 * l, want, 2 * l + 1]);
            next += 1;
        }
    }
    let n = per.iter().map(Vec::len).max().unwrap_or(1);
    let mut lists = Lists::of(workers, n);
    for (w, ins) in per.iter().enumerate() {
        for (i, a) in ins.iter().enumerate() {
            lists.set(w, i, a[0], &a[1..]);
        }
    }
    (lists, workers)
}

/// One variant over `m`: its `h` rows after one play, its event-timed median µs (untraced), and one
/// traced play's stamps (`[workers, n, MK_TRACE_WIDTH]` u64) with its lists.
struct Played {
    h: Vec<u16>,
    us: f64,
    trace: Vec<u64>,
    lists: Lists,
    workers: usize,
}

fn play_variant(device: &CudaDevice, m: &Mlp, p: &Pre, flags: u32) -> Played {
    let (hd, i) = (MK_MLP_HIDDEN, MK_MLP_INTER);
    let (lists, workers) = variant_lists(m.layers, flags);
    let counters = 2 * m.layers;
    // `NO_WAIT` reads a layer's input and activation with no wait for their producers: they are
    // the chain's, in place before the launch (so each play reads data, not a race).
    let no_wait = flags & MK_MLP_NO_WAIT != 0;
    let mut h0 = m.h0.clone();
    h0.resize((m.layers + 1) * hd, 0);
    let h0 = if no_wait { p.h.clone() } else { h0 };
    let a0 = if no_wait {
        p.acts.clone()
    } else {
        vec![0u16; m.layers * i]
    };
    let h = DeviceBuffer::upload(device, &h0).expect("h");
    let act = DeviceBuffer::upload(device, &a0).expect("act");
    let ins = DeviceBuffer::upload_bytes(device, &lists.bytes()).expect("lists");
    let c = DeviceBuffer::upload_bytes(device, &vec![0u8; 4 * counters]).expect("counters");
    let tr = DeviceBuffer::upload_bytes(device, &vec![0u8; 8 * workers * lists.n * MK_TRACE_WIDTH])
        .expect("trace");
    let w = MkMlpWeights {
        gain: m.gain.ptr().cast_const(),
        gate: (m.gate.0.ptr().cast_const(), m.gate.1.ptr().cast_const()),
        up: (m.up.0.ptr().cast_const(), m.up.1.ptr().cast_const()),
        down: (m.down.0.ptr().cast_const(), m.down.1.ptr().cast_const()),
    };
    let rows = MkMlpRows {
        nrow: p.nrow.ptr().cast_const(),
        qrow: p.qrow.ptr().cast_const(),
        qact: p.qact.ptr().cast_const(),
        qscale: p.qscale.ptr().cast_const(),
    };
    let launch = |trace: *mut std::ffi::c_void| {
        // SAFETY: `c` holds `counters` u32; null is the default stream.
        unsafe { cuda_memset(c.ptr(), 4 * counters) };
        // SAFETY: as `megakernel`'s; the rows and the trace hold what the variant reads and writes.
        unsafe {
            mk_mlp_ablation(
                (flags, trace),
                (ins.ptr().cast_const(), lists.n as u32),
                (c.ptr(), counters as u32),
                (h.ptr(), act.ptr()),
                (w, rows),
                (m.layers as u32, MLP_EPS, MLP_GAIN_OFFSET),
                std::ptr::null_mut(),
            )
        }
        .expect("ablation");
    };
    launch(std::ptr::null_mut());
    let hout = h.download().expect("h");
    let us = event_median(&mut || launch(std::ptr::null_mut()));
    launch(tr.ptr());
    let bytes = tr.download().expect("trace");
    let trace = (bytes.as_chunks::<4>().0.iter())
        .map(|q| {
            u64::from(q[0])
                | (u64::from(q[1]) << 16)
                | (u64::from(q[2]) << 32)
                | (u64::from(q[3]) << 48)
        })
        .collect();
    Played {
        h: hout,
        us,
        trace,
        lists,
        workers,
    }
}

/// The SM clock in kHz (`cudaDevAttrClockRate`), to turn `clock64` ticks into µs.
fn sm_clock_khz() -> f64 {
    unsafe extern "C" {
        fn cudaGetDevice(device: *mut i32) -> i32;
        fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    }
    const CUDA_DEV_ATTR_CLOCK_RATE: i32 = 13;
    let (mut dev, mut khz) = (0i32, 0i32);
    // SAFETY: plain runtime queries with out-pointers to locals.
    unsafe {
        assert_eq!(cudaGetDevice(&mut dev), 0, "cudaGetDevice");
        assert_eq!(
            cudaDeviceGetAttribute(&mut khz, CUDA_DEV_ATTR_CLOCK_RATE, dev),
            0,
            "clock rate"
        );
    }
    f64::from(khz)
}

/// The phases a stamp pair spans (`T_*` of the kernel), in the output's order.
const PHASES: [(&str, usize, usize); 6] = [
    ("prime", 0, 1),
    ("wait", 1, 2),
    ("norm", 2, 3),
    ("quant", 3, 4),
    ("dots", 4, 5),
    ("store_arrive", 5, 7),
];

/// The variant's line per opcode: per phase the mean and max µs over its instructions, and the
/// critical path: per layer the slowest instruction of each opcode, wait excluded, summed.
fn report(name: &str, pl: &Played, layers: usize, khz: f64) -> Vec<String> {
    let us = |ticks: u64| ticks as f64 / khz * 1e3;
    let mut out = Vec::new();
    let mut crit = vec![[0.0f64; 2]; layers];
    for (op, label) in [(MK_OP_GATE_UP, "gate_up"), (MK_OP_DOWN, "down")] {
        let mut sums = [0.0f64; PHASES.len()];
        let mut maxs = [0.0f64; PHASES.len()];
        let mut count = 0usize;
        for w in 0..pl.workers {
            for i in 0..pl.lists.n {
                let at = (w * pl.lists.n + i) * MK_INSTRUCTION_WIDTH;
                if pl.lists.words[at] != op {
                    continue;
                }
                let layer = pl.lists.words[at + 1] as usize;
                let t = &pl.trace[(w * pl.lists.n + i) * MK_TRACE_WIDTH..][..MK_TRACE_WIDTH];
                count += 1;
                let mut busy = 0.0;
                for (k, &(_, a, b)) in PHASES.iter().enumerate() {
                    let d = us(t[b].saturating_sub(t[a]));
                    sums[k] += d;
                    maxs[k] = maxs[k].max(d);
                    if k != 1 {
                        busy += d;
                    }
                }
                let slot = usize::from(op == MK_OP_DOWN);
                crit[layer][slot] = crit[layer][slot].max(busy);
            }
        }
        let phases: Vec<String> = (PHASES.iter().enumerate())
            .map(|(k, &(p, ..))| format!("{p} {:.2}/{:.2}", sums[k] / count.max(1) as f64, maxs[k]))
            .collect();
        out.push(format!(
            "M0A {name} {label} instructions {count} | {} (mean/max us)",
            phases.join(" | ")
        ));
    }
    let path: f64 = crit.iter().map(|c| c[0] + c[1]).sum();
    out.insert(
        0,
        format!(
            "M0A {name} total_us {:.1} | critical_path_us {path:.1} (per layer the slowest gate/up \
             and down, waits excluded) | per layer {:.2}",
            pl.us,
            pl.us / layers as f64
        ),
    );
    out
}

/// M0-A, recorded (timing only, the user's request 2026-11-10): M0's program (V0) and the six
/// variants over 30 layers, each its event-timed total and, from one traced play, the per-phase
/// µs per opcode and the critical path; whether each keeps the bits against today's chain; then
/// the matching numbers without the megakernel: today's chain as plain launches, and the gate/up
/// at 66 and at 132 plain CTAs. Recorded, not asserted.
#[test]
fn the_megakernel_ablation_is_recorded() {
    let device = h100();
    let layers = 30;
    let m = mlp(&device, layers, 0x4D41);
    let want = chain(&device, &m, &mut once);
    let p = pre(&device, &m);
    let khz = sm_clock_khz();
    eprintln!("M0A sm_clock_khz {khz}");
    let hd = MK_MLP_HIDDEN;
    for (name, flags, bits) in VARIANTS {
        let pl = play_variant(&device, &m, &p, flags);
        for line in report(name, &pl, layers, khz) {
            eprintln!("{line}");
        }
        let bad = mismatched_bits(&pl.h[hd..], &want[hd..]);
        eprintln!(
            "M0A {name} bits mismatched_bits={bad} (expected {})",
            if bits {
                "0: bits kept"
            } else {
                "any: not bit-identical by design"
            }
        );
    }
    // Without the megakernel.
    let mut plain = 0.0;
    chain(&device, &m, &mut |f| plain = event_median(f));
    let i = MK_MLP_INTER;
    let act = DeviceBuffer::upload(&device, &vec![0u16; layers * i]).expect("act");
    let gate_up = |full: bool| {
        event_median(&mut || {
            for l in 0..layers {
                // SAFETY: the layer's weights, normed row and activation are inside their buffers.
                unsafe {
                    mk_gate_up_plain(
                        full,
                        (
                            (m.gate.0.ptr().cast_const(), m.gate.1.ptr().cast_const()),
                            (m.up.0.ptr().cast_const(), m.up.1.ptr().cast_const()),
                        ),
                        (p.nrow.ptr().cast_const(), act.ptr()),
                        (layers as u32, l as u32),
                        std::ptr::null_mut(),
                    )
                }
                .expect("plain gate/up");
            }
        })
    };
    let (gu66, gu132) = (gate_up(false), gate_up(true));
    eprintln!(
        "M0A_CHAIN plain {plain:.1} fullwave {:.1} | gate_up alone x30: 66 CTAs {gu66:.1} 132 CTAs \
         {gu132:.1} (the full-wave chain = plain - 66 + 132)",
        plain - gu66 + gu132
    );
}
