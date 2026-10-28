// SPDX-License-Identifier: Apache-2.0
//! The weight table's host half (`scratchy_target_cuda::weights`) against a tiny synthetic
//! checkpoint, with no toolchain.
//!
//! The checkpoint is written with `safetensors::serialize`, once in each name ordering a VL
//! checkpoint ships (`model.language_model.*`, the official one, and `language_model.model.*`, the
//! repacked one); the table always names the second, which the shared `GpuWeights` reader aliases.
//! Checks: every `LoadError` arm, found before anything is staged; a bf16 tensor staged byte for
//! byte; a GDN bundle staged region by region from bf16 and f32 pieces, bf16 widened to f32
//! exactly and every gap between regions `+0.0`; and the stub allocator refusing an upload.

use std::path::Path;

use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use scratchy_target_cuda::step::expert_bank;
use scratchy_target_cuda::step::{gdn_bundle, moe_bundle};
use scratchy_target_cuda::tape::{ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, TapeInput};
use scratchy_target_cuda::weights::{
    CudaWeight, HostStaging, LoadError, check, open_checkpoint, stage,
};
use scratchy_tensors::{DType, DeviceAllocator};

/// The GDN geometry of the synthetic layer: 8 conv channels of 4 taps, 2 value heads of 32.
const CONV_DIM: u32 = 8;
const CONV_KERNEL: u32 = 4;
const NV: u32 = 2;
const HEAD_V: u32 = 32;

/// The tensors under the decoder root, as (name after the root, dtype, shape, bytes).
fn tensors() -> Vec<(&'static str, Dtype, Vec<usize>, Vec<u8>)> {
    let bf16s = |n: usize, seed: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| bf16::from_f32(seed + i as f32 / 64.0).to_le_bytes())
            .collect()
    };
    let f32s = |n: usize, seed: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (seed - i as f32 / 8.0).to_le_bytes())
            .collect()
    };
    let (c, k, nv, hv) = (
        CONV_DIM as usize,
        CONV_KERNEL as usize,
        NV as usize,
        HEAD_V as usize,
    );
    vec![
        (
            "embed_tokens.weight",
            Dtype::BF16,
            vec![16, 8],
            bf16s(128, 0.5),
        ),
        ("norm.weight", Dtype::BF16, vec![8], bf16s(8, -1.0)),
        (
            "layers.0.mlp.up_proj.weight",
            Dtype::F32,
            vec![4, 8],
            f32s(32, 1.0),
        ),
        (
            "layers.0.linear_attn.conv1d.weight",
            Dtype::BF16,
            vec![c, 1, k],
            bf16s(c * k, -0.25),
        ),
        (
            "layers.0.linear_attn.A_log",
            Dtype::F32,
            vec![nv],
            f32s(nv, 0.75),
        ),
        (
            "layers.0.linear_attn.dt_bias",
            Dtype::BF16,
            vec![nv],
            bf16s(nv, 2.0),
        ),
        (
            "layers.0.linear_attn.norm.weight",
            Dtype::F32,
            vec![hv],
            f32s(hv, 3.0),
        ),
    ]
}

/// The synthetic checkpoint in `dir` as one `model.safetensors`, every tensor under `root`.
fn write_checkpoint(dir: &Path, root: &str) {
    let all = tensors();
    let views: Vec<(String, TensorView<'_>)> = (all.iter())
        .map(|(name, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes).expect("a tensor view");
            (format!("{root}.{name}"), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize the checkpoint");
    std::fs::write(dir.join("model.safetensors"), bytes).expect("write the checkpoint");
}

/// A fresh directory under the target dir for one test's checkpoint.
fn checkpoint_dir(name: &str, root: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weight_table_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    write_checkpoint(&dir, root);
    dir
}

/// The table names the repacked ordering; both checkpoints must load through the alias.
const ROOT: &str = "language_model.model";
const ORDERINGS: [&str; 2] = ["model.language_model", "language_model.model"];

const EMBED: CudaWeight = CudaWeight::Bf16 {
    tensor: "language_model.model.embed_tokens.weight",
    shape: &[16, 8],
};
const NORM: CudaWeight = CudaWeight::Bf16 {
    tensor: "language_model.model.norm.weight",
    shape: &[8],
};
const BUNDLE: CudaWeight = CudaWeight::GdnBundle {
    conv1d: "language_model.model.layers.0.linear_attn.conv1d.weight",
    a_log: "language_model.model.layers.0.linear_attn.A_log",
    dt_bias: "language_model.model.layers.0.linear_attn.dt_bias",
    norm: "language_model.model.layers.0.linear_attn.norm.weight",
    conv_dim: CONV_DIM,
    conv_kernel: CONV_KERNEL,
    nv: NV,
    head_v: HEAD_V,
};

/// A tape that declares `weight_bytes` and nothing else the table checks read.
fn tape(weight_bytes: &'static [u64]) -> CudaTape {
    let span = ArenaSpan {
        offset: ArenaOffset(0),
        bytes: 0,
    };
    CudaTape {
        bucket_m: 1,
        arena: ArenaLayout {
            colour_bytes: &[],
            colour_offsets: &[],
            total_bytes: 0,
        },
        input: TapeInput::Hidden(span),
        result: span,
        weight_bytes,
        steps: &[],
        rotary: None,
        kv: None,
        gdn: None,
        moe: None,
        rotary_local: None,
    }
}

fn bundle_bytes() -> u64 {
    gdn_bundle(CONV_DIM, CONV_KERNEL, NV, HEAD_V)
        .expect("the bundle fits")
        .bytes
}

/// The table and the tapes of a well-formed model: the embedding table at 0, a source with no
/// weight at 1 (`None`, 0 bytes), the norm at 2, the GDN bundle at 3.
fn good() -> (Vec<Option<CudaWeight>>, Vec<CudaTape>) {
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes()]));
    (
        vec![Some(EMBED), None, Some(NORM), Some(BUNDLE)],
        vec![tape(bytes), tape(bytes)],
    )
}

#[test]
fn a_well_formed_table_checks_in_both_orderings() {
    let (table, tapes) = good();
    for root in ORDERINGS {
        let w = open_checkpoint(&checkpoint_dir(&format!("good_{root}"), root)).expect("open");
        check(&table, &tapes, &w).unwrap_or_else(|e| panic!("{root}: {e}"));
    }
}

#[test]
fn a_bf16_tensor_is_staged_byte_for_byte() {
    for root in ORDERINGS {
        let mut w = open_checkpoint(&checkpoint_dir(&format!("bf16_{root}"), root)).expect("open");
        let staged = stage(0, &EMBED, &mut w).expect("stage the table");
        let want = &tensors()[0].3;
        assert_eq!(&staged, want, "{root}: the embedding table's bytes");
    }
}

/// The bundle's four regions at `gdn_bundle`'s offsets, each piece widened to f32 exactly (bf16
/// to f32 loses nothing, f32 is copied), the conv as `[conv_dim, K]` in on-disk order, and every
/// byte between regions `+0.0`.
#[test]
fn a_gdn_bundle_is_packed_exactly() {
    let layout = gdn_bundle(CONV_DIM, CONV_KERNEL, NV, HEAD_V).expect("the bundle fits");
    assert_eq!(
        (
            layout.conv1d,
            layout.a_log,
            layout.dt_bias,
            layout.norm,
            layout.bytes
        ),
        (0, 256, 512, 768, 1024),
        "the synthetic bundle's regions"
    );
    let all = tensors();
    let widen = |bytes: &[u8], dtype: Dtype| -> Vec<f32> {
        match dtype {
            Dtype::BF16 => (bytes.as_chunks::<2>().0.iter())
                .map(|&p| bf16::from_le_bytes(p).to_f32())
                .collect(),
            _ => (bytes.as_chunks::<4>().0.iter())
                .map(|&q| f32::from_le_bytes(q))
                .collect(),
        }
    };
    for root in ORDERINGS {
        let mut w =
            open_checkpoint(&checkpoint_dir(&format!("bundle_{root}"), root)).expect("open");
        let staged = stage(3, &BUNDLE, &mut w).expect("stage the bundle");
        assert_eq!(
            staged.len() as u64,
            layout.bytes,
            "{root}: the bundle's bytes"
        );
        let f: Vec<f32> = (staged.as_chunks::<4>().0.iter())
            .map(|&q| f32::from_le_bytes(q))
            .collect();
        let mut written = vec![false; f.len()];
        for (piece, at) in [
            (3, layout.conv1d),
            (4, layout.a_log),
            (5, layout.dt_bias),
            (6, layout.norm),
        ] {
            let (name, dtype, _, bytes) = &all[piece];
            let want = widen(bytes, *dtype);
            let at = at as usize / 4;
            assert_eq!(
                &f[at..at + want.len()],
                want.as_slice(),
                "{root}: {name} at {at}"
            );
            written[at..at + want.len()].fill(true);
        }
        for (i, v) in f.iter().enumerate().filter(|(i, _)| !written[*i]) {
            assert_eq!(v.to_bits(), 0, "{root}: gap element {i} is {v}, not +0.0");
        }
        assert!(
            written.iter().any(|w| !w),
            "the synthetic bundle has gaps to check"
        );
    }
}

/// The MoE geometry of the synthetic layer: 4 experts, hidden 64, expert width 32.
const MOE_E: u32 = 4;
const MOE_H: u32 = 64;
const MOE_I: u32 = 32;

/// The synthetic MoE layer's three bf16 tensors under `{root}.layers.0.mlp`, as (name after the
/// root, shape, bytes): the router, the fused gate and up, and the down.
fn moe_tensors() -> Vec<(&'static str, Vec<usize>, Vec<u8>)> {
    let (e, h, i) = (MOE_E as usize, MOE_H as usize, MOE_I as usize);
    let bf16s = |n: usize, seed: u16| -> Vec<u8> {
        // Finite bf16 bit patterns that change with the element and differ between the tensors,
        // so a misplaced or swapped region shows.
        (0..n)
            .flat_map(|k| (seed.wrapping_add(k as u16) & 0x3FFF | 0x0400).to_le_bytes())
            .collect()
    };
    vec![
        ("layers.0.mlp.gate.weight", vec![e, h], bf16s(e * h, 0x1000)),
        (
            "layers.0.mlp.experts.gate_up_proj",
            vec![e, 2 * i, h],
            bf16s(e * 2 * i * h, 0x2000),
        ),
        (
            "layers.0.mlp.experts.down_proj",
            vec![e, h, i],
            bf16s(e * h * i, 0x0300),
        ),
    ]
}

/// The MoE layer's checkpoint in a fresh `dir`, every tensor under `root`.
fn moe_checkpoint_dir(name: &str, root: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weight_table_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    let all = moe_tensors();
    let views: Vec<(String, TensorView<'_>)> = (all.iter())
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::BF16, shape.clone(), bytes).expect("a tensor view");
            (format!("{root}.{name}"), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize the checkpoint");
    std::fs::write(dir.join("model.safetensors"), bytes).expect("write the checkpoint");
    dir
}

const MOE_BUNDLE: CudaWeight = CudaWeight::MoeBundle {
    router: "language_model.model.layers.0.mlp.gate.weight",
    gate_up: "language_model.model.layers.0.mlp.experts.gate_up_proj",
    down: "language_model.model.layers.0.mlp.experts.down_proj",
    experts: MOE_E,
    hidden: MOE_H,
    inter: MOE_I,
};

/// The MoE bundle (E = 4, H = 64, I = 32) staged byte for byte as the router, the fused gate and
/// up, and the down back to back at `moe_bundle`'s offsets, in both orderings; and the router-only
/// read, the router tensor staged on its own, is exactly the bundle's first `E·H` bf16, the bytes
/// the router GEMM reads at the bundle's byte 0.
#[test]
fn a_moe_bundle_is_packed_exactly() {
    let layout = moe_bundle(MOE_E, MOE_H, MOE_I).expect("the bundle fits");
    let (e, h, i) = (u64::from(MOE_E), u64::from(MOE_H), u64::from(MOE_I));
    assert_eq!(
        (layout.router, layout.gate_up, layout.down, layout.bytes),
        (
            0,
            e * h * 2,
            e * h * 2 + e * 2 * i * h * 2,
            e * h * 2 * (1 + 3 * i)
        ),
        "the synthetic bundle's regions"
    );
    assert_eq!(MOE_BUNDLE.packed_bytes(), Some(layout.bytes));
    assert_eq!(
        MOE_BUNDLE.lead(),
        "language_model.model.layers.0.mlp.gate.weight"
    );
    let want: Vec<u8> = (moe_tensors().into_iter()).flat_map(|(.., b)| b).collect();
    assert_eq!(want.len() as u64, layout.bytes, "the concatenation's bytes");
    let router_only = CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.mlp.gate.weight",
        shape: &[MOE_E as u64, MOE_H as u64],
    };
    for root in ORDERINGS {
        let dir = moe_checkpoint_dir(&format!("moe_bundle_{root}"), root);
        let mut w = open_checkpoint(&dir).expect("open");
        let staged = stage(0, &MOE_BUNDLE, &mut w).expect("stage the bundle");
        assert_eq!(
            staged.len() as u64,
            layout.bytes,
            "{root}: the bundle's bytes"
        );
        for (part, at, end) in [
            ("router", layout.router, layout.gate_up),
            ("gate_up", layout.gate_up, layout.down),
            ("down", layout.down, layout.bytes),
        ] {
            let (at, end) = (at as usize, end as usize);
            assert!(
                staged[at..end] == want[at..end],
                "{root}: the {part} region's bytes"
            );
        }
        let mut w = open_checkpoint(&dir).expect("open");
        let router = stage(0, &router_only, &mut w).expect("stage the router alone");
        assert_eq!(
            router.as_slice(),
            &staged[..layout.gate_up as usize],
            "{root}: the router-only read is the bundle's first region"
        );
        let table = [Some(MOE_BUNDLE)];
        let bytes: &'static [u64] = Box::leak(Box::new([layout.bytes]));
        let w = open_checkpoint(&dir).expect("open");
        check(&table, &[tape(bytes)], &w).unwrap_or_else(|e| panic!("{root}: {e}"));
    }
}

/// `table` against the good tapes (or `tapes`), with the checkpoint at the repacked root.
fn refusal(name: &str, table: &[Option<CudaWeight>], tapes: &[CudaTape]) -> LoadError {
    let w = open_checkpoint(&checkpoint_dir(name, ROOT)).expect("open");
    let e = check(table, tapes, &w).expect_err("the table is refused");
    eprintln!("{name}: {e}");
    e
}

#[test]
fn a_missing_checkpoint_is_refused() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("weight_table_none");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the dir");
    let Err(e) = open_checkpoint(&dir) else {
        panic!("an empty dir opened as a checkpoint");
    };
    assert!(matches!(e, LoadError::NoCheckpoint { .. }), "{e}");
}

#[test]
fn a_table_of_another_length_is_refused() {
    let (table, tapes) = good();
    let e = refusal("length", &table[..3], &tapes);
    assert!(
        matches!(
            e,
            LoadError::TableLength {
                table: 3,
                tape: 4,
                bucket: 0
            }
        ),
        "{e}"
    );
}

#[test]
fn a_missing_tensor_is_refused_by_name_and_index() {
    let (mut table, tapes) = good();
    table[2] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.final_norm.weight",
        shape: &[8],
    });
    let e = refusal("missing", &table, &tapes);
    assert!(e.to_string().contains("final_norm.weight"), "{e}");
    assert!(matches!(e, LoadError::Missing { ix: 2, .. }), "{e}");
}

/// A dense tensor stored as f32 is refused, never converted.
#[test]
fn a_dense_tensor_of_another_dtype_is_refused() {
    let (mut table, _) = good();
    table[2] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.mlp.up_proj.weight",
        shape: &[4, 8],
    });
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 64, bundle_bytes()]));
    let e = refusal("dtype", &table, &[tape(bytes)]);
    assert!(
        matches!(
            e,
            LoadError::Dtype {
                ix: 2,
                got: DType::F32,
                ..
            }
        ),
        "{e}"
    );
}

#[test]
fn a_tensor_of_another_shape_is_refused() {
    let (mut table, tapes) = good();
    table[0] = Some(CudaWeight::Bf16 {
        tensor: "language_model.model.embed_tokens.weight",
        shape: &[8, 16],
    });
    let e = refusal("shape", &table, &tapes);
    assert!(e.to_string().contains("[16, 8]"), "{e}");
    let LoadError::Shape {
        ix: 0, got, want, ..
    } = e
    else {
        panic!("{e}");
    };
    assert_eq!((got, want), (vec![16, 8], vec![8, 16]));
    // A GDN piece of another shape: a bundle of 3 value heads reads `[3]` A_log.
    let (mut table, _) = good();
    table[3] = Some(bundle_of(3));
    let e = refusal("gdn_shape", &table, &tapes);
    assert!(
        matches!(e, LoadError::Shape { ix: 3, tensor, .. } if tensor.ends_with("A_log")),
        "{e}"
    );
}

/// [`BUNDLE`] with `nv` value heads.
fn bundle_of(nv: u32) -> CudaWeight {
    let CudaWeight::GdnBundle {
        conv1d,
        a_log,
        dt_bias,
        norm,
        conv_dim,
        conv_kernel,
        head_v,
        ..
    } = BUNDLE
    else {
        unreachable!("BUNDLE is a bundle")
    };
    CudaWeight::GdnBundle {
        conv1d,
        a_log,
        dt_bias,
        norm,
        conv_dim,
        conv_kernel,
        nv,
        head_v,
    }
}

/// A packed size other than the tapes declare, and a `None` index that declares bytes.
#[test]
fn bytes_other_than_declared_are_refused() {
    let (table, _) = good();
    let short: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes() - 256]));
    let e = refusal("bytes", &table, &[tape(short)]);
    assert!(
        matches!(e, LoadError::Bytes { ix: 3, tensor: Some(t), packed, declared } if t == table[3].expect("an entry").lead() && packed == bundle_bytes() && declared == bundle_bytes() - 256),
        "{e}"
    );
    assert!(
        e.to_string().contains(table[3].expect("an entry").lead()),
        "{e}"
    );
    let unread: &'static [u64] = Box::leak(Box::new([256, 2, 16, bundle_bytes()]));
    let e = refusal("unread", &table, &[tape(unread)]);
    assert!(
        matches!(
            e,
            LoadError::Bytes {
                ix: 1,
                tensor: None,
                packed: 0,
                declared: 2
            }
        ),
        "{e}"
    );
}

/// The packed size is the largest any tape declares: a bucket declaring less does not hide one
/// declaring more.
#[test]
fn the_largest_declared_size_is_the_one_checked() {
    let (table, _) = good();
    let a: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes()]));
    let b: &'static [u64] = Box::leak(Box::new([512, 0, 16, bundle_bytes()]));
    let e = refusal("largest", &table, &[tape(a), tape(b)]);
    assert!(
        matches!(
            e,
            LoadError::Bytes {
                ix: 0,
                tensor: Some(_),
                packed: 256,
                declared: 512
            }
        ),
        "{e}"
    );
}

/// The stub allocator makes no device memory: an upload through it is refused.
#[test]
fn host_staging_refuses_an_upload() {
    let bytes = [0u8; 16];
    let mut a = HostStaging;
    // SAFETY: `bytes` is 16 readable bytes for the call.
    let got = unsafe { a.alloc_and_copy_host(bytes.as_ptr(), bytes.len()) };
    let e = got.expect_err("HostStaging allocated device memory");
    assert!(e.to_string().contains("DeviceBuffer"), "{e}");
    // SAFETY: as above.
    let got = unsafe { a.alloc_and_copy_host_aligned(bytes.as_ptr(), bytes.len(), 2) };
    assert!(got.is_err(), "the aligned upload went through");
}

/// The tie, as the compiler emits it: the lm_head's index has no entry and declares 0 bytes, and
/// the one tensor it shares with the embedding table is staged once, at the table's index. The
/// reader hands a tensor out once, so a second entry naming it would be refused as missing.
#[test]
fn a_tied_head_stages_its_tensor_once() {
    let bytes: &'static [u64] = Box::leak(Box::new([256, 0, 16, bundle_bytes(), 0]));
    let table = [Some(EMBED), None, Some(NORM), Some(BUNDLE), None];
    let mut w = open_checkpoint(&checkpoint_dir("tied", ROOT)).expect("open");
    check(&table, &[tape(bytes)], &w).expect("the tied table checks");
    let staged: Vec<usize> = (table.iter().enumerate())
        .filter_map(|(ix, e)| e.as_ref().map(|e| (ix, e)))
        .map(|(ix, e)| stage(ix, e, &mut w).map(|b| b.len()))
        .collect::<Result<_, _>>()
        .expect("stage every entry");
    assert_eq!(staged, [256, 16, bundle_bytes() as usize]);
    let e = stage(4, &EMBED, &mut w).expect_err("the table's tensor was handed out twice");
    assert!(matches!(e, LoadError::Missing { ix: 4, .. }), "{e}");
}

// ── Gemma 4's router tensors and expert bank (#885) ────────────────

/// The synthetic Gemma layer's five HF-layout bf16 tensors under `{root}.layers.0`, as (name
/// after the root, shape, bytes): the router's projection, scale and per-expert scale, and the
/// experts' fused gate and up and their down.
fn gemma_tensors() -> Vec<(&'static str, Vec<usize>, Vec<u8>)> {
    let (e, h, i) = (MOE_E as usize, MOE_H as usize, MOE_I as usize);
    let bf16s = |n: usize, seed: u16| -> Vec<u8> {
        (0..n)
            .flat_map(|k| (seed.wrapping_add(k as u16) & 0x3FFF | 0x0400).to_le_bytes())
            .collect()
    };
    vec![
        (
            "layers.0.router.proj.weight",
            vec![e, h],
            bf16s(e * h, 0x1100),
        ),
        ("layers.0.router.scale", vec![h], bf16s(h, 0x1200)),
        (
            "layers.0.router.per_expert_scale",
            vec![e],
            bf16s(e, 0x1300),
        ),
        (
            "layers.0.experts.gate_up_proj",
            vec![e, 2 * i, h],
            bf16s(e * 2 * i * h, 0x2100),
        ),
        (
            "layers.0.experts.down_proj",
            vec![e, h, i],
            bf16s(e * h * i, 0x0310),
        ),
    ]
}

/// The Gemma layer's checkpoint in a fresh `dir`, every tensor under `root`.
fn gemma_checkpoint_dir(name: &str, root: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weight_table_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    let all = gemma_tensors();
    let views: Vec<(String, TensorView<'_>)> = (all.iter())
        .map(|(name, shape, bytes)| {
            let view = TensorView::new(Dtype::BF16, shape.clone(), bytes).expect("a tensor view");
            (format!("{root}.{name}"), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize the checkpoint");
    std::fs::write(dir.join("model.safetensors"), bytes).expect("write the checkpoint");
    dir
}

/// The Gemma layer's table: the three router tensors as bf16 entries, and the expert bank.
const GEMMA_TABLE: [CudaWeight; 4] = [
    CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.router.proj.weight",
        shape: &[MOE_E as u64, MOE_H as u64],
    },
    CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.router.scale",
        shape: &[MOE_H as u64],
    },
    CudaWeight::Bf16 {
        tensor: "language_model.model.layers.0.router.per_expert_scale",
        shape: &[MOE_E as u64],
    },
    CudaWeight::ExpertBank {
        gate_up: "language_model.model.layers.0.experts.gate_up_proj",
        down: "language_model.model.layers.0.experts.down_proj",
        experts: MOE_E,
        hidden: MOE_H,
        inter: MOE_I,
    },
];

/// The Gemma layer (E = 4, H = 64, I = 32) packs to its three bf16 router tensors and one expert
/// bank, byte for byte in both orderings: the bank is the fused gate and up then the down, back to
/// back at `expert_bank`'s offsets, the bundle's experts with no router.
#[test]
fn a_gemma_expert_bank_is_packed_exactly() {
    let bank = expert_bank(MOE_E, MOE_H, MOE_I).expect("the bank fits");
    let (e, h, i) = (u64::from(MOE_E), u64::from(MOE_H), u64::from(MOE_I));
    assert_eq!(
        (bank.router, bank.gate_up, bank.down, bank.bytes),
        (0, 0, e * 2 * i * h * 2, e * h * 2 * 3 * i),
        "the synthetic bank's regions"
    );
    let bundle = moe_bundle(MOE_E, MOE_H, MOE_I).expect("the bundle fits");
    assert_eq!(
        bank.bytes,
        bundle.bytes - bundle.gate_up,
        "the bundle without its router"
    );
    assert_eq!(GEMMA_TABLE[3].packed_bytes(), Some(bank.bytes));
    assert_eq!(
        GEMMA_TABLE[3].lead(),
        "language_model.model.layers.0.experts.gate_up_proj"
    );
    let all = gemma_tensors();
    let want_bank: Vec<u8> = (all[3..].iter()).flat_map(|(.., b)| b.clone()).collect();
    assert_eq!(want_bank.len() as u64, bank.bytes, "the bank's bytes");
    for root in ORDERINGS {
        let dir = gemma_checkpoint_dir(&format!("gemma_bank_{root}"), root);
        for (ix, entry) in GEMMA_TABLE.iter().enumerate() {
            let mut w = open_checkpoint(&dir).expect("open");
            let staged = stage(ix, entry, &mut w).unwrap_or_else(|e| panic!("{root}: {e}"));
            let want = match ix {
                3 => want_bank.clone(),
                _ => all[ix].2.clone(),
            };
            assert!(staged == want, "{root}: weight {ix}'s bytes");
            assert_eq!(
                Some(staged.len() as u64),
                entry.packed_bytes(),
                "{root}: {ix}"
            );
        }
        let table = GEMMA_TABLE.map(Some);
        let bytes: Vec<u64> = (GEMMA_TABLE.iter())
            .map(|w| w.packed_bytes().expect("fits"))
            .collect();
        let bytes: &'static [u64] = Box::leak(bytes.into_boxed_slice());
        let w = open_checkpoint(&dir).expect("open");
        check(&table, &[tape(bytes)], &w).unwrap_or_else(|e| panic!("{root}: {e}"));
    }
}

/// An expert bank whose fused tensor has another shape is refused naming it.
#[test]
fn a_gemma_expert_bank_of_another_shape_is_refused() {
    let dir = gemma_checkpoint_dir("gemma_bank_shape", ROOT);
    let wide = CudaWeight::ExpertBank {
        gate_up: "language_model.model.layers.0.experts.gate_up_proj",
        down: "language_model.model.layers.0.experts.down_proj",
        experts: MOE_E,
        hidden: MOE_H,
        inter: 2 * MOE_I,
    };
    let mut w = open_checkpoint(&dir).expect("open");
    let e = stage(0, &wide, &mut w).expect_err("a bank of another width");
    assert!(
        matches!(
            e,
            LoadError::Shape {
                ix: 0,
                tensor: "language_model.model.layers.0.experts.gate_up_proj",
                ..
            }
        ),
        "{e}"
    );
}

// ── MLX-affine weights (#948) ──────────────────────────────────────

use scratchy_target_cuda::step::{affine_weight, moe_bundle_affine};
use scratchy_target_cuda::weights::{AffineTensors, ScaleLayout};

/// The affine fixtures' geometry, GLM-shaped and small: 4 experts, hidden 128, expert width 64,
/// group 64. An affine `[n, k]` weight of `n` = 96 rows (no multiple of 256 bytes of scales: the
/// layout's padding is exercised).
const AFF_E: u32 = 4;
const AFF_H: u32 = 128;
const AFF_I: u32 = 64;
const AFF_G: u32 = 64;
const AFF_N: u32 = 96;

/// One affine projection's HF-layout MLX tensors, `[.., rows, k]`: the packed `[.., rows, k/8]`
/// u32 codes and the `[.., rows, k/group]` bf16 scales and biases, each value distinct so a
/// misplaced byte shows. `seed` separates projections.
fn mlx(lead: &[usize], k: usize, seed: u32) -> [(Dtype, Vec<usize>, Vec<u8>); 3] {
    let count: usize = lead.iter().product();
    let groups = k / AFF_G as usize;
    let shape = |last: usize| lead.iter().copied().chain([last]).collect::<Vec<_>>();
    let codes: Vec<u8> = (0..count * k / 8)
        .flat_map(|w| (seed.wrapping_mul(0x9E37_79B9) ^ w as u32).to_le_bytes())
        .collect();
    let bf16s = |base: u16| -> Vec<u8> {
        (0..count * groups)
            .flat_map(|i| (base.wrapping_add(i as u16) & 0x3FFF | 0x0400).to_le_bytes())
            .collect()
    };
    [
        (Dtype::U32, shape(k / 8), codes),
        (
            Dtype::BF16,
            shape(groups),
            bf16s(0x1000 + seed as u16 * 0x100),
        ),
        (
            Dtype::BF16,
            shape(groups),
            bf16s(0x2000 + seed as u16 * 0x100),
        ),
    ]
}

/// `[blocks, n, groups]` bf16 bytes as `[blocks, groups, n]`, written independently of the
/// loader's `group_major`: by output position, reading its source element.
fn transposed(src: &[u8], blocks: usize, n: usize, groups: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    for b in 0..blocks {
        for g in 0..groups {
            for r in 0..n {
                let at = ((b * n + r) * groups + g) * 2;
                out.extend_from_slice(&src[at..at + 2]);
            }
        }
    }
    out
}

/// A checkpoint of `tensors` (name after `root`, dtype, shape, bytes) in a fresh `dir`.
fn tensors_dir(
    name: &str,
    root: &str,
    tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)],
) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("weight_table_{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("make the checkpoint dir");
    let views: Vec<(String, TensorView<'_>)> = (tensors.iter())
        .map(|(n, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes).expect("a tensor view");
            (format!("{root}.{n}"), view)
        })
        .collect();
    let bytes = safetensors::serialize(views, None).expect("serialize the checkpoint");
    std::fs::write(dir.join("model.safetensors"), bytes).expect("write the checkpoint");
    dir
}

use std::path::PathBuf;

/// The affine `[n, k]` weight's three tensors under `{root}.layers.0.self_attn.q_proj`.
fn affine_tensors() -> Vec<(String, Dtype, Vec<usize>, Vec<u8>)> {
    let [w, s, b] = mlx(&[AFF_N as usize], AFF_H as usize, 7);
    let p = "layers.0.self_attn.q_proj";
    vec![
        (format!("{p}.weight"), w.0, w.1, w.2),
        (format!("{p}.scales"), s.0, s.1, s.2),
        (format!("{p}.biases"), b.0, b.1, b.2),
    ]
}

/// The affine entry over [`affine_tensors`] in `layout`.
fn affine_entry(layout: ScaleLayout) -> CudaWeight {
    CudaWeight::Affine {
        weight: "language_model.model.layers.0.self_attn.q_proj.weight",
        scales: "language_model.model.layers.0.self_attn.q_proj.scales",
        biases: "language_model.model.layers.0.self_attn.q_proj.biases",
        n: AFF_N,
        k: AFF_H,
        group: AFF_G,
        layout,
    }
}

/// An affine `[96, 128]` weight at group 64 staged byte for byte in both scale layouts, against an
/// independently written expected buffer: the packed codes as on disk at byte 0, the scales and
/// biases at `affine_weight`'s 256-aligned offsets, as on disk (row-major) or transposed to
/// `[k/64, n]` (group-major), every padding byte 0; in both name orderings, and `check` accepts it.
#[test]
fn an_affine_weight_is_packed_exactly_in_both_layouts() {
    let layout = affine_weight(AFF_N, AFF_H, AFF_G).expect("fits");
    let packed = u64::from(AFF_N) * u64::from(AFF_H) / 2;
    let per = u64::from(AFF_N) * u64::from(AFF_H / AFF_G) * 2;
    assert_eq!(
        (layout.weight, layout.scales, layout.biases, layout.bytes),
        (
            0,
            packed,
            packed + per.next_multiple_of(256),
            packed + per.next_multiple_of(256) + per
        ),
        "the weight's regions"
    );
    let t = affine_tensors();
    let (codes, scales, biases) = (&t[0].3, &t[1].3, &t[2].3);
    let groups = (AFF_H / AFF_G) as usize;
    for (scale_layout, order) in [
        (ScaleLayout::RowMajor, false),
        (ScaleLayout::GroupMajor, true),
    ] {
        let mut want = vec![0u8; layout.bytes as usize];
        want[..codes.len()].copy_from_slice(codes);
        for (src, at) in [(scales, layout.scales), (biases, layout.biases)] {
            let bytes = match order {
                true => transposed(src, 1, AFF_N as usize, groups),
                false => src.clone(),
            };
            want[at as usize..][..bytes.len()].copy_from_slice(&bytes);
        }
        let entry = affine_entry(scale_layout);
        assert_eq!(entry.packed_bytes(), Some(layout.bytes));
        for root in ORDERINGS {
            let dir = tensors_dir(&format!("affine_{scale_layout:?}_{root}"), root, &t);
            let mut w = open_checkpoint(&dir).expect("open");
            let staged = stage(0, &entry, &mut w).expect("stage the affine weight");
            assert!(
                staged == want,
                "{root} {scale_layout:?}: the buffer's bytes"
            );
            let bytes: &'static [u64] = Box::leak(Box::new([layout.bytes]));
            let w = open_checkpoint(&dir).expect("open");
            check(&[Some(entry)], &[tape(bytes)], &w).unwrap_or_else(|e| panic!("{root}: {e}"));
        }
    }
}

/// The affine MoE bundle's 11 tensors under `{root}.layers.1.mlp`: the dense router `[E, H]`, the
/// f32 correction bias `[E]`, and each `switch_mlp` projection's codes, scales and biases. `router`
/// `scales: true` adds a `gate.scales` beside the router, as a quantized-router checkpoint ships.
fn bundle_tensors(router_scales: bool) -> Vec<(String, Dtype, Vec<usize>, Vec<u8>)> {
    let (e, h, i) = (AFF_E as usize, AFF_H as usize, AFF_I as usize);
    let p = "layers.1.mlp";
    let mut t = vec![
        (
            format!("{p}.gate.weight"),
            Dtype::BF16,
            vec![e, h],
            (0..e * h)
                .flat_map(|k| ((0x3000u16 + k as u16) & 0x3FFF | 0x0400).to_le_bytes())
                .collect(),
        ),
        (
            format!("{p}.gate.e_score_correction_bias"),
            Dtype::F32,
            vec![e],
            (0..e)
                .flat_map(|k| (k as f32 * 0.25 - 1.0).to_le_bytes())
                .collect(),
        ),
    ];
    for (proj, rows, k, seed) in [
        ("gate_proj", i, h, 1),
        ("up_proj", i, h, 2),
        ("down_proj", h, i, 3),
    ] {
        let [w, s, b] = mlx(&[e, rows], k, seed);
        t.push((format!("{p}.switch_mlp.{proj}.weight"), w.0, w.1, w.2));
        t.push((format!("{p}.switch_mlp.{proj}.scales"), s.0, s.1, s.2));
        t.push((format!("{p}.switch_mlp.{proj}.biases"), b.0, b.1, b.2));
    }
    if router_scales {
        t.push((
            format!("{p}.gate.scales"),
            Dtype::BF16,
            vec![e, h / AFF_G as usize],
            vec![0; e * h / AFF_G as usize * 2],
        ));
    }
    t
}

/// The affine MoE bundle entry over [`bundle_tensors`].
fn bundle_entry() -> CudaWeight {
    let p = "language_model.model.layers.1.mlp";
    let proj = |name: &str| AffineTensors {
        weight: Box::leak(format!("{p}.switch_mlp.{name}.weight").into_boxed_str()),
        scales: Box::leak(format!("{p}.switch_mlp.{name}.scales").into_boxed_str()),
        biases: Box::leak(format!("{p}.switch_mlp.{name}.biases").into_boxed_str()),
    };
    CudaWeight::MoeBundleAffine {
        router: "language_model.model.layers.1.mlp.gate.weight",
        correction_bias: "language_model.model.layers.1.mlp.gate.e_score_correction_bias",
        gate: proj("gate_proj"),
        up: proj("up_proj"),
        down: proj("down_proj"),
        experts: AFF_E,
        hidden: AFF_H,
        inter: AFF_I,
        group: AFF_G,
    }
}

/// The affine MoE bundle (E 4, H 128, I 64, group 64) staged byte for byte against an independently
/// written expected layout, as `a_moe_bundle_is_packed_exactly`: the router at byte 0, the f32 bias
/// at its 256-aligned slab, then the gate, up and down slabs (separate, as the checkpoint ships
/// them), each its codes as on disk and its scales and biases group-major per expert (`[E, k/64,
/// n]`), every slab 256-aligned and every padding byte 0; in both orderings, and `check` accepts it.
#[test]
fn an_affine_moe_bundle_is_packed_exactly() {
    let layout = moe_bundle_affine(AFF_E, AFF_H, AFF_I, AFF_G).expect("fits");
    let t = bundle_tensors(false);
    let mut want = vec![0u8; layout.bytes as usize];
    let mut at =
        |offset: u64, bytes: &[u8]| want[offset as usize..][..bytes.len()].copy_from_slice(bytes);
    at(layout.router, &t[0].3);
    at(layout.correction_bias, &t[1].3);
    let (e, h, i) = (AFF_E as usize, AFF_H as usize, AFF_I as usize);
    for (k, (slab, n, kk)) in [(layout.gate, i, h), (layout.up, i, h), (layout.down, h, i)]
        .into_iter()
        .enumerate()
    {
        let groups = kk / AFF_G as usize;
        let base = 2 + 3 * k;
        assert!(slab.weight.is_multiple_of(256), "slab {k}'s codes");
        assert!(slab.scales.is_multiple_of(256), "slab {k}'s scales");
        assert!(slab.biases.is_multiple_of(256), "slab {k}'s biases");
        at(slab.weight, &t[base].3);
        at(slab.scales, &transposed(&t[base + 1].3, e, n, groups));
        at(slab.biases, &transposed(&t[base + 2].3, e, n, groups));
    }
    let entry = bundle_entry();
    assert_eq!(entry.packed_bytes(), Some(layout.bytes));
    assert_eq!(
        entry.lead(),
        "language_model.model.layers.1.mlp.gate.weight"
    );
    for root in ORDERINGS {
        let dir = tensors_dir(&format!("affine_bundle_{root}"), root, &t);
        let mut w = open_checkpoint(&dir).expect("open");
        let staged = stage(0, &entry, &mut w).expect("stage the bundle");
        assert_eq!(
            staged.len() as u64,
            layout.bytes,
            "{root}: the bundle's bytes"
        );
        assert!(staged == want, "{root}: the bundle's bytes, byte for byte");
        let bytes: &'static [u64] = Box::leak(Box::new([layout.bytes]));
        let w = open_checkpoint(&dir).expect("open");
        check(&[Some(entry)], &[tape(bytes)], &w).unwrap_or_else(|e| panic!("{root}: {e}"));
    }
}

/// `t` with tensor `name` (after the root) replaced by `f`'s, or removed when `f` gives `None`.
fn edited(
    t: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    name: &str,
    f: impl Fn(&(String, Dtype, Vec<usize>, Vec<u8>)) -> Option<(String, Dtype, Vec<usize>, Vec<u8>)>,
) -> Vec<(String, Dtype, Vec<usize>, Vec<u8>)> {
    t.into_iter()
        .filter_map(|x| if x.0 == name { f(&x) } else { Some(x) })
        .collect()
}

/// `entry`'s refusal against a checkpoint of `t`.
fn affine_refusal(
    name: &str,
    entry: CudaWeight,
    t: &[(String, Dtype, Vec<usize>, Vec<u8>)],
) -> LoadError {
    let w = open_checkpoint(&tensors_dir(name, ROOT, t)).expect("open");
    let bytes: &'static [u64] = Box::leak(Box::new([entry.packed_bytes().expect("fits")]));
    let e = check(&[Some(entry)], &[tape(bytes)], &w).expect_err("the entry is refused");
    eprintln!("{name}: {e}");
    e
}

/// A missing `.scales` is refused as `Missing`, naming it.
#[test]
fn an_affine_weight_with_no_scales_is_refused_by_name() {
    let t = edited(affine_tensors(), "layers.0.self_attn.q_proj.scales", |_| {
        None
    });
    let e = affine_refusal(
        "affine_no_scales",
        affine_entry(ScaleLayout::GroupMajor),
        &t,
    );
    assert!(
        matches!(e, LoadError::Missing { ix: 0, tensor } if tensor.ends_with("q_proj.scales")),
        "{e}"
    );
}

/// A dense bf16 `.weight` where the packed u32 codes are expected is refused as `Dtype`.
#[test]
fn a_dense_weight_where_packed_is_expected_is_refused() {
    let t = edited(affine_tensors(), "layers.0.self_attn.q_proj.weight", |x| {
        let (n, k) = (AFF_N as usize, AFF_H as usize);
        Some((
            x.0.clone(),
            Dtype::BF16,
            vec![n, k / 8],
            vec![0; n * k / 8 * 2],
        ))
    });
    let e = affine_refusal("affine_dense", affine_entry(ScaleLayout::GroupMajor), &t);
    assert!(
        matches!(
            e,
            LoadError::Dtype {
                ix: 0,
                got: DType::BF16,
                ..
            }
        ),
        "{e}"
    );
}

/// A 3-bit packed width (`k · 3/32` words where 4-bit needs `k/8`) is refused as `Shape`, as GLM's
/// 3-bit `[12288, 384]` against the 4-bit `[12288, 512]`.
#[test]
fn a_3_bit_packed_width_is_refused() {
    let t = edited(affine_tensors(), "layers.0.self_attn.q_proj.weight", |x| {
        let (n, words) = (AFF_N as usize, AFF_H as usize * 3 / 32);
        Some((
            x.0.clone(),
            Dtype::U32,
            vec![n, words],
            vec![0; n * words * 4],
        ))
    });
    let e = affine_refusal("affine_3bit", affine_entry(ScaleLayout::GroupMajor), &t);
    let LoadError::Shape {
        ix: 0, got, want, ..
    } = e
    else {
        panic!("{e}");
    };
    assert_eq!((got, want), (vec![96, 12], vec![96, 16]));
}

/// F16 scales are refused as `Dtype`: the kernels read bf16, and nothing is converted.
#[test]
fn f16_affine_scales_are_refused() {
    let t = edited(affine_tensors(), "layers.0.self_attn.q_proj.scales", |x| {
        Some((x.0.clone(), Dtype::F16, x.2.clone(), x.3.clone()))
    });
    let e = affine_refusal("affine_f16", affine_entry(ScaleLayout::GroupMajor), &t);
    assert!(
        matches!(
            e,
            LoadError::Dtype {
                ix: 0,
                got: DType::F16,
                ..
            }
        ),
        "{e}"
    );
}

/// A bundle whose checkpoint ships `gate.scales` beside the router, which the table reads dense, is
/// refused as `QuantizedRouter`, naming both.
#[test]
fn a_quantized_router_is_refused_by_name() {
    let e = affine_refusal("affine_qrouter", bundle_entry(), &bundle_tensors(true));
    assert!(
        matches!(
            &e,
            LoadError::QuantizedRouter {
                ix: 0,
                router: "language_model.model.layers.1.mlp.gate.weight",
                scales,
            } if scales == "language_model.model.layers.1.mlp.gate.scales"
        ),
        "{e}"
    );
    assert!(e.to_string().contains("quantized on disk"), "{e}");
}
