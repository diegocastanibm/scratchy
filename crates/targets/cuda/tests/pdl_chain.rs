// SPDX-License-Identifier: Apache-2.0
//! FP8 S8 W5, step 1 (measure first), on the H100: what does PDL (programmatic dependent launch)
//! save per launch in a captured decode graph? Two of the tape's real one-row kernels at Gemma's
//! shapes, chained as its 30 dense layers: the one-row RMSNorm over 2,816 columns, then the fp8
//! gate/up GEMV (2,112 × 2,816, GeGLU, one weight per layer): 60 launches, captured into one graph
//! twice. One graph keeps capture's ordinary edges; in the other every kernel-to-kernel edge is
//! made programmatic (`cudaGraphDependencyTypeProgrammatic` from
//! `cudaGraphKernelNodePortProgrammatic`), so each kernel may start before the previous one has
//! finished and waits in `pdl::wait()` before its first read of it.
//!
//! Each graph is replayed 20 times after 3 warm-ups, timed by CUDA events, in two rounds
//! (A, B, A, B), as `PDL_CHAIN <off|on> median=… us`; then `PDL_SAVED` is
//! `(off − on) / 60` µs per launch. Both graphs' outputs must be the same bits. Recorded, not
//! asserted, but for the bits. Built only under `nvcc` (`required-features`).

pub mod common {
    pub mod watchdog;
}

use std::ffi::c_void;

use scratchy_target_cuda::kernels::{gate_up_gemv_fp8, rms_norm_bf16};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

type Handle = *mut c_void;

/// `cudaGraphEdgeData`: `from_port`, `to_port`, `type`, 5 zero bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct EdgeData {
    from_port: u8,
    to_port: u8,
    kind: u8,
    reserved: [u8; 5],
}

const STREAM_NON_BLOCKING: u32 = 1;
const CAPTURE_THREAD_LOCAL: i32 = 1;
const NODE_TYPE_KERNEL: i32 = 0;
const KERNEL_PORT_PROGRAMMATIC: u8 = 1;
const DEPENDENCY_PROGRAMMATIC: u8 = 1;

unsafe extern "C" {
    fn cudaStreamCreateWithFlags(stream: *mut Handle, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: Handle) -> i32;
    fn cudaStreamSynchronize(stream: Handle) -> i32;
    fn cudaStreamBeginCapture(stream: Handle, mode: i32) -> i32;
    fn cudaStreamEndCapture(stream: Handle, graph: *mut Handle) -> i32;
    fn cudaGraphGetEdges_v2(
        graph: Handle,
        from: *mut Handle,
        to: *mut Handle,
        data: *mut EdgeData,
        count: *mut usize,
    ) -> i32;
    fn cudaGraphNodeGetType(node: Handle, kind: *mut i32) -> i32;
    fn cudaGraphRemoveDependencies(
        graph: Handle,
        from: *const Handle,
        to: *const Handle,
        count: usize,
    ) -> i32;
    fn cudaGraphAddDependencies_v2(
        graph: Handle,
        from: *const Handle,
        to: *const Handle,
        data: *const EdgeData,
        count: usize,
    ) -> i32;
    fn cudaGraphInstantiateWithFlags(exec: *mut Handle, graph: Handle, flags: u64) -> i32;
    fn cudaGraphLaunch(exec: Handle, stream: Handle) -> i32;
    fn cudaGraphExecDestroy(exec: Handle) -> i32;
    fn cudaGraphDestroy(graph: Handle) -> i32;
    fn cudaEventCreate(event: *mut Handle) -> i32;
    fn cudaEventRecord(event: Handle, stream: Handle) -> i32;
    fn cudaEventSynchronize(event: Handle) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> i32;
    fn cudaEventDestroy(event: Handle) -> i32;
}

/// Gemma's dense layers, hidden size and MLP width.
const LAYERS: usize = 30;
const HIDDEN: u32 = 2816;
const MLP: u32 = 2112;
/// The GeGLU code (`step::GatedOp`).
const GELU: u32 = 1;

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn ok(code: i32, what: &str) {
    assert_eq!(code, 0, "{what}");
}

/// The chain's buffers: the input row and the norm's gain and output, shared by every layer (so
/// each norm overwrites what the previous gate/up read), and each layer's gate/up weights and
/// output.
struct Chain {
    x: DeviceBuffer,
    gain: DeviceBuffer,
    h: DeviceBuffer,
    weights: Vec<[DeviceBuffer; 4]>,
    outs: Vec<DeviceBuffer>,
}

impl Chain {
    fn new(device: &CudaDevice) -> Self {
        let (hu, mu) = (HIDDEN as usize, MLP as usize);
        let bf = |seed: u16, len: usize| {
            let v: Vec<u16> = (0..len)
                .map(|i| 0x3C00 | ((i as u16).wrapping_mul(seed) & 0x3FF))
                .collect();
            DeviceBuffer::upload(device, &v).expect("bf16")
        };
        let bytes = |len: usize, f: &dyn Fn(usize) -> u8| {
            let v: Vec<u8> = (0..len).map(f).collect();
            DeviceBuffer::upload_bytes(device, &v).expect("bytes")
        };
        let scales = || {
            let v: Vec<u8> = (0..mu)
                .flat_map(|_| (1.0f32 / 512.0).to_le_bytes())
                .collect();
            DeviceBuffer::upload_bytes(device, &v).expect("scales")
        };
        let weights = (0..LAYERS)
            .map(|l| {
                let w = |salt: usize| {
                    bytes(mu * hu, &|i| {
                        (((i * 7 + l * 13 + salt) % 0x7E) as u8) | ((i & 1) << 7) as u8
                    })
                };
                [w(0), scales(), w(5), scales()]
            })
            .collect();
        Self {
            x: bf(37, hu),
            gain: bf(11, hu),
            h: bf(1, hu),
            weights,
            outs: (0..LAYERS)
                .map(|_| DeviceBuffer::upload(device, &vec![0u16; mu]).expect("c"))
                .collect(),
        }
    }

    /// The 60 launches on `stream`, asynchronous.
    fn launch(&self, stream: Handle) {
        for (w, c) in self.weights.iter().zip(&self.outs) {
            // SAFETY: `x`, `gain` and `h` hold HIDDEN bf16, three allocations; `stream` is live.
            unsafe {
                rms_norm_bf16(
                    self.x.ptr().cast_const(),
                    self.gain.ptr().cast_const(),
                    self.h.ptr(),
                    (1, HIDDEN),
                    1e-6,
                    0.0,
                    stream,
                )
            }
            .expect("rms_norm_bf16 launch");
            // SAFETY: `h` holds HIDDEN bf16, each weight MLP·HIDDEN e4m3 bytes and its MLP f32
            // scales, `c` MLP bf16, all their own allocations; one split, no partials.
            unsafe {
                gate_up_gemv_fp8(
                    (
                        self.h.ptr().cast_const(),
                        (w[0].ptr().cast_const(), w[1].ptr().cast_const()),
                        (w[2].ptr().cast_const(), w[3].ptr().cast_const()),
                    ),
                    (c.ptr(), std::ptr::null_mut()),
                    (MLP, HIDDEN, 1),
                    GELU,
                    stream,
                )
            }
            .expect("gate_up_gemv_fp8 launch");
        }
    }

    fn outputs(&self) -> Vec<Vec<u16>> {
        self.outs.iter().map(|c| c.download().expect("c")).collect()
    }
}

/// Make every kernel-to-kernel edge of `graph` programmatic; returns how many it changed.
fn programmatic(graph: Handle) -> usize {
    let mut count = 0usize;
    // SAFETY: a count query with null arrays.
    ok(
        unsafe {
            cudaGraphGetEdges_v2(
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut count,
            )
        },
        "cudaGraphGetEdges_v2 count",
    );
    let (mut from, mut to) = (
        vec![std::ptr::null_mut(); count],
        vec![std::ptr::null_mut(); count],
    );
    let mut data = vec![EdgeData::default(); count];
    // SAFETY: three arrays of `count` entries.
    ok(
        unsafe {
            cudaGraphGetEdges_v2(
                graph,
                from.as_mut_ptr(),
                to.as_mut_ptr(),
                data.as_mut_ptr(),
                &mut count,
            )
        },
        "cudaGraphGetEdges_v2",
    );
    let kernel = |node: Handle| {
        let mut kind = -1;
        // SAFETY: `node` is one of the graph's nodes.
        ok(
            unsafe { cudaGraphNodeGetType(node, &mut kind) },
            "cudaGraphNodeGetType",
        );
        kind == NODE_TYPE_KERNEL
    };
    let edge = EdgeData {
        from_port: KERNEL_PORT_PROGRAMMATIC,
        to_port: 0,
        kind: DEPENDENCY_PROGRAMMATIC,
        reserved: [0; 5],
    };
    let mut changed = 0;
    for (&f, &t) in from.iter().zip(&to) {
        if kernel(f) && kernel(t) {
            // SAFETY: `f → t` is one of the graph's edges; it is replaced by the same edge, typed.
            unsafe {
                ok(
                    cudaGraphRemoveDependencies(graph, &f, &t, 1),
                    "cudaGraphRemoveDependencies",
                );
                ok(
                    cudaGraphAddDependencies_v2(graph, &f, &t, &edge, 1),
                    "cudaGraphAddDependencies_v2",
                );
            }
            changed += 1;
        }
    }
    changed
}

/// The chain captured on `stream` and instantiated, its edges made programmatic when `pdl`.
fn capture(chain: &Chain, stream: Handle, pdl: bool) -> (Handle, Handle, usize) {
    let mut graph = std::ptr::null_mut();
    // SAFETY: `stream` is a live non-default stream; nothing else is captured on this thread.
    unsafe {
        ok(
            cudaStreamBeginCapture(stream, CAPTURE_THREAD_LOCAL),
            "cudaStreamBeginCapture",
        );
        chain.launch(stream);
        ok(
            cudaStreamEndCapture(stream, &mut graph),
            "cudaStreamEndCapture",
        );
    }
    let changed = if pdl { programmatic(graph) } else { 0 };
    let mut exec = std::ptr::null_mut();
    // SAFETY: `graph` was just captured.
    ok(
        unsafe { cudaGraphInstantiateWithFlags(&mut exec, graph, 0) },
        "cudaGraphInstantiateWithFlags",
    );
    (graph, exec, changed)
}

/// `(median, min, max)` µs of 20 replays of `exec` on `stream` after 3 warm-ups.
fn replay_times(exec: Handle, stream: Handle) -> (f64, f64, f64) {
    let (mut start, mut end) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: plain runtime calls with out-pointers to locals; the events are destroyed below.
    unsafe {
        ok(cudaEventCreate(&mut start), "cudaEventCreate");
        ok(cudaEventCreate(&mut end), "cudaEventCreate");
    }
    let mut times = Vec::new();
    for i in 0..23 {
        let mut ms = 0.0f32;
        // SAFETY: as above; `exec` is an instantiated graph.
        unsafe {
            ok(cudaEventRecord(start, stream), "record");
            ok(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
            ok(cudaEventRecord(end, stream), "record");
            ok(cudaEventSynchronize(end), "synchronize");
            ok(cudaEventElapsedTime(&mut ms, start, end), "elapsed");
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
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// Step 1's microbenchmark: the 60-launch chain with and without programmatic edges.
#[test]
fn the_pdl_chain_saving_is_recorded() {
    let device = h100();
    let chain = Chain::new(&device);
    let mut stream = std::ptr::null_mut();
    // SAFETY: an out-pointer to a local; the stream is destroyed below.
    ok(
        unsafe { cudaStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING) },
        "cudaStreamCreateWithFlags",
    );
    let launches = 2 * LAYERS;
    let (g_off, e_off, _) = capture(&chain, stream, false);
    let (g_on, e_on, changed) = capture(&chain, stream, true);
    eprintln!("PDL_EDGES programmatic={changed} of {} launches", launches);
    let mut sums = [0.0f64; 2];
    let mut bits = Vec::new();
    for round in 1..=2 {
        for (i, (name, exec)) in [("off", e_off), ("on", e_on)].into_iter().enumerate() {
            let (med, lo, hi) = replay_times(exec, stream);
            sums[i] += med;
            eprintln!(
                "PDL_CHAIN {name} median={med:.1}us min={lo:.1}us max={hi:.1}us \
                 ({launches} launches, {:.2} us each, round {round})",
                med / launches as f64
            );
            // SAFETY: `stream` is live; the replays above ran on it.
            ok(
                unsafe { cudaStreamSynchronize(stream) },
                "cudaStreamSynchronize",
            );
            if round == 1 {
                bits.push(chain.outputs());
            }
        }
    }
    let saved = (sums[0] - sums[1]) / 2.0 / launches as f64;
    eprintln!(
        "PDL_SAVED {saved:.2} us per launch (off {:.1}us, on {:.1}us; rule: >= 0.5) {}",
        sums[0] / 2.0,
        sums[1] / 2.0,
        if saved >= 0.5 { "PASS" } else { "FAIL" }
    );
    let same = bits[0] == bits[1];
    eprintln!(
        "PDL_BITS the programmatic graph's outputs are the ordinary graph's: {}",
        if same { "PASS" } else { "FAIL" }
    );
    // SAFETY: each handle was created above and is no longer used.
    unsafe {
        cudaGraphExecDestroy(e_off);
        cudaGraphExecDestroy(e_on);
        cudaGraphDestroy(g_off);
        cudaGraphDestroy(g_on);
        cudaStreamDestroy(stream);
    }
    assert_eq!(
        changed,
        launches - 1,
        "every edge of the chain is kernel to kernel"
    );
    assert!(same, "PDL changed the chain's bits");
}
