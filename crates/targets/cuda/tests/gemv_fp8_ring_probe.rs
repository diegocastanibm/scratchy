// SPDX-License-Identifier: Apache-2.0
//! FP8 S8 W2c's probe (/tmp/d-w2c.md §3), on the H100: is the fp8 decode GEMV bound by the rate
//! of its TMA operations? `kernels/tma_ring_probe.cu` streams the 27B's gate+up (34,816 × 5,120)
//! and down (5,120 × 17,408) e4m3 weights through the GEMV's grid, warps and rings, with no dot
//! product, at five sizes of TMA operation: 256-byte vectors at 3 and at 4 rows a warp (today's),
//! 512-byte 2-row tiles, 4 KB 16-row tiles and 16 KB 16 × 1,024 tiles per 4 warps.
//!
//! Each variant is timed by CUDA events, cold (L2 cleared before each launch) and hot, in two
//! rounds, and printed as `PROBE <variant> <shape> cold|hot <TB/s>` of the weight bytes. Recorded,
//! not asserted, but for a check that every variant reads the weight and not the TMA's zero fill:
//! every byte is 1.0 (e4m3 0x38), and lane `l` of each row's warp folds byte `l` of each 256
//! columns, so the folds of a variant add up to exactly `n · (k / 256) · 32`.
//!
//! Built only under `nvcc` (`required-features`). No GPU fails rather than skips.

pub mod common {
    pub mod watchdog;
}

use std::ffi::c_void;

use scratchy_target_cuda::kernels::tma_ring_probe;
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};

/// `cudaEvent_t`, declared here for the timing only (the runtime declares no event call).
type CudaEvent = *mut c_void;

unsafe extern "C" {
    fn cudaEventCreate(event: *mut CudaEvent) -> i32;
    fn cudaEventRecord(event: CudaEvent, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: CudaEvent) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: CudaEvent, end: CudaEvent) -> i32;
    fn cudaEventDestroy(event: CudaEvent) -> i32;
    fn cudaMemsetAsync(dst: *mut c_void, value: i32, bytes: usize, stream: *mut c_void) -> i32;
}

/// The bytes cleared before a cold run: twice the H100's 50 MiB L2 (`tests/gemv_fp8.rs`'s).
const SCRUB_BYTES: usize = 128 << 20;

/// The variants, by the launcher's number.
const VARIANTS: &[(u32, &str)] = &[
    (0, "vec256_r3"),
    (1, "vec256"),
    (2, "pair512"),
    (3, "tile4k"),
    (4, "page16k"),
];

fn h100() -> CudaDevice {
    common::watchdog::arm();
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

/// `(median, min, max)` µs of 20 launches of `launch` after 3 warm-ups, each between two CUDA
/// events on the default stream; `scrub` (when given) is cleared before each
/// (`tests/gemv_fp8.rs`'s `event_times`).
fn event_times(scrub: Option<&DeviceBuffer>, mut launch: impl FnMut()) -> (f64, f64, f64) {
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
        // SAFETY: as above; `scrub` is a live allocation of `SCRUB_BYTES`.
        unsafe {
            if let Some(s) = scrub {
                assert_eq!(
                    cudaMemsetAsync(s.ptr(), i, SCRUB_BYTES, std::ptr::null_mut()),
                    0
                );
            }
            assert_eq!(cudaEventRecord(start, std::ptr::null_mut()), 0, "record");
            launch();
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
    (times[times.len() / 2], times[0], times[times.len() - 1])
}

/// The sum of the f32 folds in `out`, in f64.
fn fold_sum(out: &DeviceBuffer) -> f64 {
    let bytes: Vec<u8> = (out.download().expect("download").iter())
        .flat_map(|h| h.to_le_bytes())
        .collect();
    (bytes.as_chunks::<4>().0.iter())
        .map(|&c| f64::from(f32::from_le_bytes(c)))
        .sum()
}

/// The probe at the 27B's gate+up and down: every variant cold and hot, two rounds, one
/// `PROBE` line each; then each shape's variants must have folded the same bytes.
#[test]
fn the_tma_ring_probe_bandwidth_is_recorded() {
    let device = h100();
    let scrub = DeviceBuffer::upload_bytes(&device, &vec![0u8; SCRUB_BYTES]).expect("scrub");
    let mut same = true;
    for (n, k, shape) in [
        (34816u32, 5120u32, "gate+up_34816x5120"),
        (5120, 17408, "down_5120x17408"),
    ] {
        let (nu, ku) = (n as usize, k as usize);
        // e4m3 0x38 is 1.0.
        let b = DeviceBuffer::upload_bytes(&device, &vec![0x38u8; nu * ku]).expect("b");
        let want = (nu * (ku / 256) * 32) as f64;
        let mut sums = Vec::new();
        for round in 1..=2 {
            for &(v, name) in VARIANTS {
                let out = DeviceBuffer::upload_bytes(&device, &vec![0u8; (16 * nu + 256) * 4])
                    .expect("out");
                for (temp, sc) in [("cold", Some(&scrub)), ("hot", None)] {
                    let (med, lo, hi) = event_times(sc, || {
                        // SAFETY: `b` holds n·k bytes and `out` 16·n + 256 f32, two cudaMalloc'd
                        // allocations; null is the default stream, which the events wait on.
                        unsafe {
                            tma_ring_probe(
                                (b.ptr().cast_const(), out.ptr()),
                                (n, k),
                                v,
                                std::ptr::null_mut(),
                            )
                        }
                        .expect("tma_ring_probe launch");
                    });
                    eprintln!(
                        "PROBE {name} {shape} {temp} {:.2} (round {round}: median={med:.1}us \
                         min={lo:.1}us max={hi:.1}us)",
                        (nu * ku) as f64 / med / 1e6
                    );
                }
                sums.push((name, fold_sum(&out)));
            }
        }
        let ok = sums.iter().all(|&(_, s)| s == want);
        eprintln!(
            "PROBE_FOLDS {shape}: want={want} {sums:?} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        same &= ok;
    }
    assert!(
        same,
        "a ring probe variant did not read every byte of its weight"
    );
}
