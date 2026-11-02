// SPDX-License-Identifier: Apache-2.0
//! The MLX affine 4-bit kernels (#949) on the H100: the dequant-fused GEMM at 256 and 64 columns
//! (`kernels/affine_gemm_bf16.cu`), the GEMV (`kernels/affine_gemv_bf16.cu`) and the embed gather
//! (`kernels/affine_embed_bf16.cu`), against the host dequant
//! (`scratchy_quantizations::affine::affine_dequant_b4_to_dtype`, bf16 scales to bf16) and their
//! bf16 twins (`gemm_bf16_abt`, `gemm_col64_bf16_abt`, `gemv_bf16_abt`) run on that dequantized
//! weight, bit for bit.
//!
//! - (a) every dense quantized tensor of GLM-4.5-Air's checkpoint shards 1 and 12, dequantized
//!   through the GEMM itself (`A` the identity, so `C = dequant(W)ᵀ`), and every embedding row
//!   through the embed;
//! - (b) one tensor of each GLM shape at m ∈ {1, 128, 512, 4096}: against its twin bit for bit,
//!   and within the GEMM's f64 bound (`common::numerics`), with a repeat run;
//! - (c) synthetic adversarial weights: every code in every position of a word, scales at ±bf16
//!   min-normal, ±1, ±bf16 max/16, biases at ±0, ±subnormal-producing and ±large values;
//! - (d) the invariants: no write past the GEMV's `n` or the embed's `rows · hidden`, a +0 row for
//!   an id outside the vocabulary;
//! - (e) the time of each real shape against its twin, printed, not gated.
//!
//! `±0` pairs count as equal (a `−0` weight sums to `+0` through the identity) and are counted
//! separately. Every case prints its line before a test fails. Built only under `nvcc`; no GPU
//! fails rather than skips, and so does a missing checkpoint, with the command that fetches it.

// `pub`: each test binary uses a different part of `common`, and nothing in it is dead code.
pub mod common {
    pub mod numerics;
}

use std::path::Path;

use common::numerics::{bf16_in, bf16_inputs, mismatched_bits, reference, widen};
use half::bf16;
use scratchy_layers::weights::GpuWeights;
use scratchy_quantizations::affine::affine_dequant_b4_to_dtype;
use scratchy_target_cuda::kernels::{
    AffineRegions, affine_embed_bf16, affine_gemm_bf16_abt, affine_gemm_col64_bf16_abt,
    affine_gemv_bf16_abt, gemm_bf16_abt, gemm_col64_bf16_abt, gemv_bf16_abt,
};
use scratchy_target_cuda::runtime::{CudaDevice, DeviceBuffer};
use scratchy_target_cuda::step::affine_weight;
use scratchy_target_cuda::weights::{CudaWeight, HostStaging, ScaleLayout, group_major, stage};
use scratchy_tensors::DType;

/// `mlx-community/GLM-4.5-Air-4bit` at `60837794f3caafc4682dd1a9188a82c55a9100ef`, its two shards
/// downloaded as the GKE plan of #949 does.
const CHECKPOINT: &str = "/data/glm45-4bit";
const SHARDS: [&str; 2] = [
    "model-00001-of-00012.safetensors",
    "model-00012-of-00012.safetensors",
];
/// The one affine group CUDA takes (`op_abi::CUDA_AFFINE_FORMS`).
const GROUP: u32 = 64;
/// Every output element before a launch: a bf16 NaN, so an element the kernel never writes fails.
const SENTINEL: u16 = 0xFFFF;
/// Extra output elements past the written ones, which the kernel must leave untouched.
const TAIL: usize = 4096;
/// GLM-4.5-Air's dense quantized shapes `(n, k)`: q, k/v, o, layer 0's gate/up and down, the
/// shared expert's gate/up and down, the lm_head.
const GLM_SHAPES: [(u32, u32); 8] = [
    (12288, 4096),
    (1024, 4096),
    (4096, 12288),
    (10944, 4096),
    (4096, 10944),
    (1408, 4096),
    (4096, 1408),
    (151552, 4096),
];

fn h100() -> CudaDevice {
    CudaDevice::open()
        .unwrap_or_else(|e| panic!("the `nvcc` feature's tests need an sm_90 GPU (H100): {e}"))
}

fn verdict(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// One MLX affine weight as the checkpoint stores it: `n` rows of `k` 4-bit codes (`[n, k/2]`
/// bytes, low nibble first), and `[n, k/64]` bf16 scales and biases, row-major.
struct Affine {
    n: u32,
    k: u32,
    codes: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
}

impl Affine {
    /// The host dequant: the dense `[n, k]` bf16 the kernels must equal.
    fn dense(&self) -> Vec<u16> {
        let bytes = affine_dequant_b4_to_dtype(
            &self.codes,
            &self.scales,
            &self.biases,
            self.n as usize,
            self.k as usize,
            GROUP,
            4,
            DType::BF16,
            DType::BF16,
        )
        .expect("the host dequant");
        let (pairs, _) = bytes.as_chunks::<2>();
        pairs.iter().map(|&p| u16::from_le_bytes(p)).collect()
    }

    /// The device buffer `step::affine_weight` lays out: the codes, then the scales and biases in
    /// `layout`'s order, the loader's own transpose (`weights::group_major`).
    fn buffer(&self, layout: ScaleLayout) -> Vec<u8> {
        let l = affine_weight(self.n, self.k, GROUP).expect("an affine layout");
        let mut out = vec![0u8; l.bytes as usize];
        out[l.weight as usize..][..self.codes.len()].copy_from_slice(&self.codes);
        for (src, at) in [(&self.scales, l.scales), (&self.biases, l.biases)] {
            let dst = &mut out[at as usize..][..src.len()];
            match layout {
                ScaleLayout::RowMajor => dst.copy_from_slice(src),
                ScaleLayout::GroupMajor => group_major(src, dst, (1, self.n, self.k / GROUP)),
            }
        }
        out
    }
}

/// The three regions of the affine buffer `buf` of an `[n, k]` weight.
fn regions(buf: &DeviceBuffer, n: u32, k: u32) -> AffineRegions {
    let l = affine_weight(n, k, GROUP).expect("an affine layout");
    let at = |o: u64| buf.ptr().cast_const().wrapping_byte_add(o as usize);
    AffineRegions {
        codes: at(l.weight),
        scales: at(l.scales),
        biases: at(l.biases),
    }
}

/// Which kernel takes a shape, as `steps_from_tape::gemm_kernel` picks the affine step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Gemv,
    Tiled,
    Col64,
}

impl Kind {
    fn of(m: u32, n: u32) -> Self {
        match (m, n % 256) {
            (1, _) => Self::Gemv,
            (_, 0) => Self::Tiled,
            _ => Self::Col64,
        }
    }
}

/// `C = A · dequant(W)ᵀ` through the affine kernel of `kind`, waited for: `C`'s `m·n` elements
/// over a NaN-poisoned buffer, and whether its tail held.
fn run_affine(
    device: &CudaDevice,
    kind: Kind,
    a: &DeviceBuffer,
    w: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) -> (Vec<u16>, bool) {
    let len = m as usize * n as usize;
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("upload C");
    let r = regions(w, n, k);
    let null = std::ptr::null_mut();
    // SAFETY: `a` holds `m·k` bf16, `w` an `affine_weight(n, k, 64)` buffer (group-major), `c`
    // `m·n + TAIL` bf16, all cudaMalloc'd (256-byte aligned), `c` overlapping neither;
    // `CudaDevice::open` ran `prepare_all`.
    let launched = unsafe {
        match kind {
            Kind::Gemv => affine_gemv_bf16_abt(a.ptr(), r, c.ptr(), (n, k, GROUP), null),
            Kind::Tiled => affine_gemm_bf16_abt(a.ptr(), r, c.ptr(), (m, n, k, GROUP), null),
            Kind::Col64 => affine_gemm_col64_bf16_abt(a.ptr(), r, c.ptr(), (m, n, k, GROUP), null),
        }
    };
    launched.unwrap_or_else(|e| panic!("{kind:?} m={m} n={n} k={k}: {e}"));
    let out = c.download().expect("download C");
    let tail = out[len..].iter().all(|&x| x == SENTINEL);
    (out[..len].to_vec(), tail)
}

/// The bf16 twin of `kind` on the dense weight `b`, waited for: `C`'s `m·n` elements.
fn run_twin(
    device: &CudaDevice,
    kind: Kind,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) -> Vec<u16> {
    let len = m as usize * n as usize;
    let c = DeviceBuffer::upload(device, &vec![SENTINEL; len]).expect("upload C");
    let null = std::ptr::null_mut();
    // SAFETY: `a`, `b` and `c` hold `m·k`, `n·k` and `m·n` bf16, cudaMalloc'd, `c` overlapping
    // neither; `CudaDevice::open` ran `prepare_all`.
    let launched = unsafe {
        match kind {
            Kind::Gemv => gemv_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (n, k), null),
            Kind::Tiled => gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, null),
            Kind::Col64 => gemm_col64_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (m, n, k), null),
        }
    };
    launched.unwrap_or_else(|e| panic!("twin {kind:?} m={m} n={n} k={k}: {e}"));
    c.download().expect("download C")
}

/// `(mismatched_bits, signed_zero_pairs)` of `gpu` against `want`: a `±0` pair is equal and
/// counted in the second.
fn dequant_bits(gpu: impl Iterator<Item = u16>, want: impl Iterator<Item = u16>) -> (usize, usize) {
    let (mut bad, mut zeros) = (0, 0);
    for (g, w) in gpu.zip(want) {
        if g == w {
            continue;
        }
        if (g | w) & 0x7FFF == 0 {
            zeros += 1;
        } else {
            bad += 1;
        }
    }
    (bad, zeros)
}

/// Dequantize `w` through its GEMM (`A` the first `k` rows of the identity, zero rows to a
/// multiple of 128, so `C[j, i] = dequant(W)[i, j]`) and compare with the host's `dense`; print the
/// line; return whether it passed.
fn identity_check(
    device: &CudaDevice,
    what: &str,
    w: &DeviceBuffer,
    dense: &[u16],
    (n, k): (u32, u32),
) -> bool {
    let (nu, ku) = (n as usize, k as usize);
    let m = k.next_multiple_of(128);
    let mut eye = vec![0u16; m as usize * ku];
    for j in 0..ku {
        eye[j * ku + j] = bf16::ONE.to_bits();
    }
    let a = DeviceBuffer::upload(device, &eye).expect("upload A");
    let kind = Kind::of(m, n);
    let (c, tail) = run_affine(device, kind, &a, w, (m, n, k));
    // `C[j, i]` against `dense[i, j]`, every `j < k`; the padding rows past `k` multiply zeros.
    let gpu = (0..nu)
        .flat_map(|i| (0..ku).map(move |j| (i, j)))
        .map(|(i, j)| c[j * nu + i]);
    let (bad, zeros) = dequant_bits(gpu, dense.iter().copied());
    let pass = bad == 0 && tail;
    eprintln!(
        "dequant {what} via {kind:?} n={n} k={k}: mismatched_bits={bad} signed_zero_pairs={zeros} \
         tail_intact={tail} {}",
        verdict(pass)
    );
    pass
}

/// Gather every row of the row-major affine table `w` (`n` rows of `k`) through the embed and
/// compare with the host's `dense`; print the line; return whether it passed.
fn embed_check(
    device: &CudaDevice,
    what: &str,
    w: &DeviceBuffer,
    dense: &[u16],
    (n, k): (u32, u32),
) -> bool {
    let ids: Vec<u8> = (0..n).flat_map(u32::to_le_bytes).collect();
    let tokens = DeviceBuffer::upload_bytes(device, &ids).expect("upload ids");
    let len = n as usize * k as usize;
    let out = DeviceBuffer::upload(device, &vec![SENTINEL; len + TAIL]).expect("upload out");
    // SAFETY: `tokens` holds `n` u32 below `n`, `w` an `affine_weight(n, k, 64)` buffer
    // (row-major), `out` `n·k + TAIL` bf16, all cudaMalloc'd, `out` overlapping neither.
    unsafe {
        affine_embed_bf16(
            tokens.ptr(),
            regions(w, n, k),
            out.ptr(),
            (n, k, n, GROUP),
            std::ptr::null_mut(),
        )
    }
    .unwrap_or_else(|e| panic!("embed {what}: {e}"));
    let got = out.download().expect("download out");
    let tail = got[len..].iter().all(|&x| x == SENTINEL);
    let bad = mismatched_bits(&got[..len], dense);
    let pass = bad == 0 && tail;
    eprintln!(
        "dequant {what} via embed rows={n} hidden={k}: mismatched_bits={bad} tail_intact={tail} {}",
        verdict(pass)
    );
    pass
}

/// Rows of `C` the f64 bound checks: 16 spread over `m`, all of them below 16.
// shortcut: a sample, not every row (lm_head at m = 4096 is 2.5·10¹² f64 products on the host);
// widen to every row if a twin mismatch ever needs locating.
fn sampled_rows(m: usize) -> Vec<usize> {
    (0..m).step_by((m / 16).max(1)).collect()
}

/// The GEMM's f64 bound (`common::numerics::compare`'s: `|c − ref| ≤ 2⁻⁷·|ref| + 2⁻¹⁴·Σ|a·w|`) of
/// `gpu` (`[m, n]`) against `a · bᵀ` on [`sampled_rows`], `b` widened 4096 rows at a time so a
/// 151,552-row weight never sits in memory as f64 whole: `(out_of_bound, worst ratio)`.
fn bound(gpu: &[u16], a: &[u16], b: &[u16], (m, n, k): (usize, usize, usize)) -> (usize, f64) {
    const COLS: usize = 4096;
    let rows = sampled_rows(m);
    let a_rows: Vec<u16> = rows
        .iter()
        .flat_map(|&r| a[r * k..][..k].iter().copied())
        .collect();
    let aw = widen(&a_rows);
    let (mut bad, mut worst) = (0, 0.0f64);
    for c0 in (0..n).step_by(COLS) {
        let cols = COLS.min(n - c0);
        let refs = reference(&aw, &widen(&b[c0 * k..][..cols * k]), rows.len(), cols, k);
        for (i, &r) in rows.iter().enumerate() {
            for (j, &(want, s)) in refs[i * cols..][..cols].iter().enumerate() {
                let g = f64::from(bf16::from_bits(gpu[r * n + c0 + j]).to_f32());
                let ratio =
                    (g - want).abs() / (want.abs() / 128.0 + s / 16384.0).max(f64::MIN_POSITIVE);
                if !(g.is_finite() && ratio <= 1.0) {
                    bad += 1;
                }
                worst = worst.max(if g.is_finite() { ratio } else { f64::INFINITY });
            }
        }
    }
    (bad, worst)
}

/// The median host-timed microseconds of 5 launches after one warm-up, each waited for through a
/// one-element download (which waits for the default stream).
fn time_us(device: &CudaDevice, mut launch: impl FnMut()) -> f64 {
    let fence = DeviceBuffer::upload(device, &[0u16; 8]).expect("upload fence");
    let mut times: Vec<f64> = (0..6)
        .map(|_| {
            let start = std::time::Instant::now();
            launch();
            fence.download().expect("wait");
            start.elapsed().as_secs_f64() * 1e6
        })
        .skip(1)
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Launch the affine kernel of `kind` once, unwaited, into `c`.
fn launch_affine(
    kind: Kind,
    a: &DeviceBuffer,
    w: AffineRegions,
    c: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) {
    let null = std::ptr::null_mut();
    // SAFETY: as `run_affine`'s; `c` holds at least `m·n` bf16.
    let launched = unsafe {
        match kind {
            Kind::Gemv => affine_gemv_bf16_abt(a.ptr(), w, c.ptr(), (n, k, GROUP), null),
            Kind::Tiled => affine_gemm_bf16_abt(a.ptr(), w, c.ptr(), (m, n, k, GROUP), null),
            Kind::Col64 => affine_gemm_col64_bf16_abt(a.ptr(), w, c.ptr(), (m, n, k, GROUP), null),
        }
    };
    launched.expect("launch");
}

/// Launch the twin of `kind` once, unwaited, into `c`.
fn launch_twin(
    kind: Kind,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    (m, n, k): (u32, u32, u32),
) {
    let null = std::ptr::null_mut();
    // SAFETY: as `run_twin`'s.
    let launched = unsafe {
        match kind {
            Kind::Gemv => gemv_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (n, k), null),
            Kind::Tiled => gemm_bf16_abt(a.ptr(), b.ptr(), c.ptr(), m, n, k, null),
            Kind::Col64 => gemm_col64_bf16_abt(a.ptr(), b.ptr(), c.ptr(), (m, n, k), null),
        }
    };
    launched.expect("launch");
}

/// `w` at `m` rows of seeded activations against its twin on `dense`: bit for bit, a repeat run
/// bit for bit, the f64 bound and the tail; then the time of both, printed. Returns whether it
/// passed.
fn twin_check(
    device: &CudaDevice,
    what: &str,
    w: &DeviceBuffer,
    dense: &[u16],
    (m, n, k): (u32, u32, u32),
) -> bool {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let a_bits = bf16_inputs(
        mu * ku,
        0xAF1E_0000 ^ (u64::from(m) << 32) ^ u64::from(n ^ k),
    );
    let a = DeviceBuffer::upload(device, &a_bits).expect("upload A");
    let b = DeviceBuffer::upload(device, dense).expect("upload B");
    let kind = Kind::of(m, n);
    let (c, tail) = run_affine(device, kind, &a, w, (m, n, k));
    let (again, _) = run_affine(device, kind, &a, w, (m, n, k));
    let twin = run_twin(device, kind, &a, &b, (m, n, k));
    let bad = mismatched_bits(&c, &twin);
    let repeat = mismatched_bits(&c, &again);
    let (out, ratio) = bound(&c, &a_bits, dense, (mu, nu, ku));
    let pass = bad == 0 && repeat == 0 && out == 0 && tail;
    eprintln!(
        "twin {what} {kind:?} m={m} n={n} k={k}: mismatched_bits={bad} repeat_mismatched_bits={repeat} \
         ratio={ratio:.4} out_of_bound={out} sampled_rows={} tail_intact={tail}{} {}",
        sampled_rows(mu).len(),
        if ratio > 0.99 {
            " (ratio above 0.99)"
        } else {
            ""
        },
        verdict(pass)
    );
    let out_c = DeviceBuffer::upload(device, &vec![0u16; mu * nu]).expect("upload C");
    let r = regions(w, n, k);
    let us = time_us(device, || launch_affine(kind, &a, r, &out_c, (m, n, k)));
    let twin_us = time_us(device, || launch_twin(kind, &a, &b, &out_c, (m, n, k)));
    let (wb, tb) = (w.bytes() as f64, (nu * ku * 2) as f64);
    eprintln!(
        "time {what} {kind:?} m={m} n={n} k={k}: affine={us:.1}us ({:.0} GB/s of weight) \
         twin={twin_us:.1}us ({:.0} GB/s) (recorded, not asserted)",
        wb / us / 1e3,
        tb / twin_us / 1e3
    );
    pass
}

// ── Real weights: GLM-4.5-Air's shards 1 and 12 ────────────────────

/// One shard, opened twice: once for the reference's row-major tensors, once for the loader's
/// staging (each takes the tensors it reads).
fn open_shard(shard: &str) -> (GpuWeights<HostStaging>, GpuWeights<HostStaging>) {
    let path = Path::new(CHECKPOINT).join(shard);
    let open = || {
        GpuWeights::from_single_file(&path, HostStaging).unwrap_or_else(|e| {
            panic!(
                "{}: {e}. Fetch it first: hf download mlx-community/GLM-4.5-Air-4bit --revision \
                 60837794f3caafc4682dd1a9188a82c55a9100ef --local-dir {CHECKPOINT} --include \
                 config.json model.safetensors.index.json {}",
                path.display(),
                SHARDS.join(" ")
            )
        })
    };
    (open(), open())
}

/// Every affine tensor prefix of a shard but the routed experts' (#950's), sorted: those with a
/// `.scales` sibling.
fn affine_prefixes(w: &GpuWeights<HostStaging>) -> Vec<String> {
    let mut out: Vec<String> = (w.names())
        .filter_map(|n| n.strip_suffix(".scales"))
        .filter(|p| !p.contains("switch_mlp"))
        .map(str::to_string)
        .collect();
    out.sort();
    out
}

/// Tensor `p`'s affine weight from the reference handle, and its device bytes through the
/// loader (`weights::stage` of a `CudaWeight::Affine` in `layout`).
fn load(
    p: &str,
    (reference, staging): (&mut GpuWeights<HostStaging>, &mut GpuWeights<HostStaging>),
    layout: ScaleLayout,
) -> (Affine, Vec<u8>) {
    let name = |s: &str| -> &'static str { format!("{p}.{s}").leak() };
    let (shape, _) = reference.tensor_info(name("weight")).expect("the codes");
    let (n, k) = (shape[0] as u32, shape[1] as u32 * 8);
    let mut take = |s: &str| reference.take_to_cpu_bytes(name(s)).expect("a tensor").0;
    let w = Affine {
        n,
        k,
        codes: take("weight"),
        scales: take("scales"),
        biases: take("biases"),
    };
    let entry = CudaWeight::Affine {
        weight: name("weight"),
        scales: name("scales"),
        biases: name("biases"),
        n,
        k,
        group: GROUP,
        layout,
    };
    let bytes = stage(0, &entry, staging).unwrap_or_else(|e| panic!("{p}: {e}"));
    (w, bytes)
}

/// (a): every dense quantized tensor of the two shards, through the GEMM, and every embedding row,
/// through the embed, bit for bit with the host dequant; every GLM shape among them.
#[test]
fn every_real_dense_tensor_dequantizes_bit_for_bit() {
    let device = h100();
    let (mut pass, mut shapes) = (true, Vec::new());
    for shard in SHARDS {
        let (mut reference, mut staging) = open_shard(shard);
        for p in affine_prefixes(&reference) {
            let embed = p.ends_with("embed_tokens");
            let layout = if embed {
                ScaleLayout::RowMajor
            } else {
                ScaleLayout::GroupMajor
            };
            let (w, bytes) = load(&p, (&mut reference, &mut staging), layout);
            let buf = DeviceBuffer::upload_bytes(&device, &bytes).expect("upload W");
            let dense = w.dense();
            pass &= if embed {
                embed_check(&device, &p, &buf, &dense, (w.n, w.k))
            } else {
                shapes.push((w.n, w.k));
                identity_check(&device, &p, &buf, &dense, (w.n, w.k))
            };
        }
    }
    let missing: Vec<_> = GLM_SHAPES.iter().filter(|s| !shapes.contains(s)).collect();
    eprintln!("GLM shapes not found in shards 1 and 12: {missing:?}");
    assert!(
        missing.is_empty(),
        "a GLM shape is missing from the two shards"
    );
    assert!(
        pass,
        "a dequantized tensor differs from the host's: see the lines above"
    );
}

/// (b) and (e): the first tensor of each GLM shape at m ∈ {1, 128, 512, 4096}, against its twin
/// on the host-dequantized weight and the f64 bound, with a repeat run; the times printed.
#[test]
fn every_real_shape_equals_its_twin_bit_for_bit() {
    let device = h100();
    let (mut pass, mut done) = (true, Vec::new());
    for shard in SHARDS {
        let (mut reference, mut staging) = open_shard(shard);
        for p in affine_prefixes(&reference) {
            let (shape, _) = reference
                .tensor_info(&format!("{p}.weight"))
                .expect("the codes");
            let nk = (shape[0] as u32, shape[1] as u32 * 8);
            if p.ends_with("embed_tokens") || done.contains(&nk) || !GLM_SHAPES.contains(&nk) {
                continue;
            }
            done.push(nk);
            let (w, bytes) = load(&p, (&mut reference, &mut staging), ScaleLayout::GroupMajor);
            let buf = DeviceBuffer::upload_bytes(&device, &bytes).expect("upload W");
            let dense = w.dense();
            for m in [1, 128, 512, 4096] {
                pass &= twin_check(&device, &p, &buf, &dense, (m, w.n, w.k));
            }
        }
    }
    assert_eq!(
        done.len(),
        GLM_SHAPES.len(),
        "every GLM shape found: {done:?}"
    );
    assert!(
        pass,
        "an affine GEMM differs from its twin or the bound: see the lines above"
    );
}

// ── Synthetic weights ──────────────────────────────────────────────

/// bf16 scales: ±min-normal, ±1, ±max/16.
const SCALES: [u16; 6] = [0x0080, 0x8080, 0x3F80, 0xBF80, 0x7D7F, 0xFD7F];
/// bf16 biases: ±0, ±the smallest subnormal and ±min-normal (subnormal results next to a
/// min-normal scale), ±1e30 (large, and finite whatever the scale: `15 · max/16 + 1e30` is below
/// bf16's max).
const BIASES: [u16; 8] = [
    0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x7149, 0xF149,
];

/// An adversarial `[n, k]` weight: code `(j + r + j/8) % 16` at column `j` of row `r`, so every
/// code sits in every position of a word across rows; the scales and biases cycle through
/// [`SCALES`] and [`BIASES`] by row and group, so every pair meets.
fn adversarial(n: u32, k: u32) -> Affine {
    let (nu, ku, gu) = (n as usize, k as usize, (k / GROUP) as usize);
    let code = |r: usize, j: usize| ((j + r + j / 8) % 16) as u8;
    let codes = (0..nu)
        .flat_map(|r| (0..ku / 2).map(move |b| code(r, 2 * b) | code(r, 2 * b + 1) << 4))
        .collect();
    let pick = |table: &[u16], r: usize, g: usize, mul: usize| table[(r * mul + g) % table.len()];
    let bytes = |table: &[u16], mul: usize| -> Vec<u8> {
        (0..nu)
            .flat_map(|r| (0..gu).map(move |g| (r, g)))
            .flat_map(|(r, g)| pick(table, r, g, mul).to_le_bytes())
            .collect()
    };
    Affine {
        n,
        k,
        codes,
        scales: bytes(&SCALES, 1),
        biases: bytes(&BIASES, 3),
    }
}

/// A seeded `[n, k]` weight with small finite scales and biases, for the twin and invariant cases.
fn seeded(n: u32, k: u32, seed: u64) -> Affine {
    let (nu, ku) = (n as usize, k as usize);
    let half = |v: Vec<u16>| v.iter().flat_map(|b| b.to_le_bytes()).collect();
    let groups = nu * ku / GROUP as usize;
    Affine {
        n,
        k,
        codes: bf16_inputs(nu * ku / 2, seed)
            .iter()
            .map(|b| *b as u8)
            .collect(),
        scales: half(bf16_in(groups, seed ^ 1, (0.001, 0.02))),
        biases: half(bf16_in(groups, seed ^ 2, (-0.05, 0.05))),
    }
}

/// (c): the adversarial weights at (n 256, k 128), (n 64, k 64) and (n 1408, k 1408) dequantize bit
/// for bit with the host, through the GEMM and through the embed (whose dequant reaches the output
/// with no tensor core between).
#[test]
fn adversarial_weights_dequantize_bit_for_bit() {
    let device = h100();
    let mut pass = true;
    for (n, k) in [(256, 128), (64, 64), (1408, 1408)] {
        let w = adversarial(n, k);
        let dense = w.dense();
        let what = format!("adversarial n={n} k={k}");
        let gm = DeviceBuffer::upload_bytes(&device, &w.buffer(ScaleLayout::GroupMajor))
            .expect("upload");
        let rm =
            DeviceBuffer::upload_bytes(&device, &w.buffer(ScaleLayout::RowMajor)).expect("upload");
        pass &= identity_check(&device, &what, &gm, &dense, (n, k));
        pass &= embed_check(&device, &what, &rm, &dense, (n, k));
    }
    assert!(
        pass,
        "an adversarial weight differs from the host's: see the lines above"
    );
}

/// (d): the GEMV at an `n` off the 64-column tile writes its `n` columns, its twin's bits, and
/// nothing past them; the embed writes every row, a +0 row for an id at or above `vocab` (`vocab`
/// itself and `u32::MAX`), the host's row otherwise, and nothing past `rows · hidden`.
#[test]
fn nothing_past_the_output_is_written() {
    let device = h100();
    let (n, k) = (200, 128);
    let w = seeded(n, k, 0xD0D0);
    let dense = w.dense();
    let gm =
        DeviceBuffer::upload_bytes(&device, &w.buffer(ScaleLayout::GroupMajor)).expect("upload");
    let a = DeviceBuffer::upload(&device, &bf16_inputs(k as usize, 0xD1)).expect("upload A");
    let b = DeviceBuffer::upload(&device, &dense).expect("upload B");
    let (c, tail) = run_affine(&device, Kind::Gemv, &a, &gm, (1, n, k));
    let bad = mismatched_bits(&c, &run_twin(&device, Kind::Gemv, &a, &b, (1, n, k)));
    let gemv = bad == 0 && tail;
    eprintln!(
        "gemv n={n} k={k}: mismatched_bits={bad} tail_intact={tail} {}",
        verdict(gemv)
    );

    let rm = DeviceBuffer::upload_bytes(&device, &w.buffer(ScaleLayout::RowMajor)).expect("upload");
    let ids = [0u32, n - 1, n, u32::MAX, 7];
    let tokens = DeviceBuffer::upload_bytes(
        &device,
        &ids.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>(),
    )
    .expect("upload ids");
    let (rows, ku) = (ids.len(), k as usize);
    let out = DeviceBuffer::upload(&device, &vec![SENTINEL; rows * ku + TAIL]).expect("upload out");
    // SAFETY: `tokens` holds `rows` u32, `rm` an `affine_weight(n, k, 64)` buffer (row-major),
    // `out` `rows·k + TAIL` bf16, all cudaMalloc'd, `out` overlapping neither.
    unsafe {
        affine_embed_bf16(
            tokens.ptr(),
            regions(&rm, n, k),
            out.ptr(),
            (rows as u32, k, n, GROUP),
            std::ptr::null_mut(),
        )
    }
    .expect("embed");
    let got = out.download().expect("download out");
    let want: Vec<u16> = (ids.iter())
        .flat_map(|&id| match dense.get(id as usize * ku..) {
            Some(row) if id < n => row[..ku].to_vec(),
            _ => vec![0u16; ku],
        })
        .collect();
    let bad = mismatched_bits(&got[..rows * ku], &want);
    let tail = got[rows * ku..].iter().all(|&x| x == SENTINEL);
    let embed = bad == 0 && tail;
    eprintln!(
        "embed ids={ids:?}: mismatched_bits={bad} tail_intact={tail} {}",
        verdict(embed)
    );
    assert!(
        gemv && embed,
        "an affine kernel wrote past its output: see the lines above"
    );
}
