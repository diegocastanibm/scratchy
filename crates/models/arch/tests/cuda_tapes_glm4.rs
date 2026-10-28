// SPDX-License-Identifier: Apache-2.0
//! GLM-4.5-Air's emitted CUDA tapes at its MLX 4-bit g64 variant, checked with no toolchain
//! (AI-native-Systems-Research/scratchy#948, step 2 of the 4-bit path; the kernels are #949 and
//! #950).
//!
//! `#[forward]` hands `glm-4.5-air-mlx-affine-b4-g64-qembed` (arch `glm4-moe`) to CUDA's compiler
//! through the target-emitter seam, and its module holds what came back: `CUDA_TAPES`,
//! `CUDA_UNLOWERED` and `CUDA_WEIGHTS`. The checks, each a prediction of #948's "The GLM tapes":
//!
//! - every emitted tape is, field by field, what CUDA's compiler lowers the same canonicals to
//!   (`lower_model`), each validates, and the emitted `CUDA_WEIGHTS` is its weight table;
//! - the buckets are [1, 512, 1024, 2048, 4096], and [2, 4, 8, 64] are left out;
//! - per bucket, the steps of #948's table: every projection an affine step (the embed, 323 GEMVs
//!   at one row, 231 tiled and 92 64-column GEMMs above it), 135 affine expert steps, the dense
//!   router and GLM's routing ×45 each, and no dense GEMM, GEMV or embed at all;
//! - one KV class (8 kv heads of 128) over 46 layers, and one rotary table (64 lanes, θ 1e6);
//! - the weight table: 600 entries, 324 affine weights (the embedding table row-major, every
//!   projection group-major) and 45 affine MoE bundles, 60,139,821,568 device bytes with no
//!   padding: the checkpoint's 1,698 tensors, byte for byte up to the transposes;
//! - the model is in the CUDA registry once, under its stem.
//!
//! The arenas and MoE scratch are printed per bucket: the dequant is inside the kernels, so no
//! affine step adds scratch. The 3-bit base (`glm-4.5-air` with no preset) is refused at expansion
//! by name (`GemmWeightRule::AffineForm { bits: 3, group: 64 }`), so a CUDA build names this
//! variant's preset: `model/glm-4.5-air,quant/mlx-affine-b4-g64-qembed`.

#[path = "../../../targets/cuda/tests/common/chains.rs"]
pub mod chains;

use std::collections::BTreeMap;
use std::path::PathBuf;

use scratchy_models::glm4_moe::glm_4_5_air_mlx_affine_b4_g64_qembed::{
    CUDA_TAPES, CUDA_UNLOWERED, CUDA_WEIGHTS,
};
use scratchy_subtile::handoff::TargetModelInput;
use scratchy_target_cuda::step::{CudaStep, moe_bundle_affine};
use scratchy_target_cuda::tape::{ArenaSpan, CudaTape, KvClass};
use scratchy_target_cuda::weights::{CudaWeight, ScaleLayout};
use scratchy_target_cuda_compiler::emit::lower_model;

/// The model the seam hands over, the Cargo feature that selects its base, and its preset.
const STEM: &str = "glm-4.5-air-mlx-affine-b4-g64-qembed";
const FEATURE: &str = "CARGO_FEATURE_GLM_4.5_AIR";
const PRESET: &str = "mlx-affine-b4-g64-qembed";

/// Every lowered bucket, ascending: decode's one row, then the prefill buckets.
const TAPE_BUCKETS: [u32; 5] = [1, 512, 1024, 2048, 4096];
/// The buckets off the GEMM's 128-row tile, above one row.
const LEFT_OUT: [u32; 4] = [2, 4, 8, 64];
const VOCAB: u32 = 151_552;
const HIDDEN: u32 = 4096;
const LAYERS: usize = 46;
/// The one attention class: 8 kv heads of 128, no window.
const CLASS: KvClass = KvClass {
    kv_heads: 8,
    head_dim: 128,
    window: None,
};
/// The rotary table: 64 of the head's 128 lanes (`partial_rotary_factor` 0.5), θ 1e6.
const ROT_DIM: u32 = 64;
const THETA: f64 = 1e6;
/// The MoE block: 128 experts, top-8, an expert width of 1408, group 64.
const EXPERTS: u32 = 128;
const TOP_K: u32 = 8;
const INTER: u32 = 1408;
const GROUP: u32 = 64;

/// The steps every bucket shares, whatever its kernels (#938's table for the rest).
const COMMON_STEPS: [(&str, usize); 11] = [
    ("Add", 137),
    ("AffineEmbed", 1),
    ("BiasAdd", 138),
    ("ExpertCombine", 45),
    ("MoeRoute", 45),
    ("RmsNorm", 93),
    ("Rope", 46),
    ("RopeAppend", 46),
    ("RouterLogits", 45),
    ("SiluMul", 91),
    ("ExpertSteps", 135),
];
/// The decode tape's own steps: every projection an affine GEMV (q, k, v, o × 46; layer 0's gate,
/// up, down; the shared expert's gate, up, down × 45; the lm_head), its experts gathered.
const DECODE_STEPS: [(&str, usize); 3] = [
    ("AffineGemv", 323),
    ("AttnDecode", 46),
    ("ExpertGemvAffine", 135),
];
/// Every prefill tape's own steps: the affine GEMMs on the 256- and 64-column tiles, the experts
/// sorted, then the affine grouped GEMM.
const PREFILL_STEPS: [(&str, usize); 5] = [
    ("AffineGemm", 231),
    ("AffineGemmCol64", 92),
    ("AttnPrefill", 46),
    ("ExpertGemmAffine", 135),
    ("ExpertSort", 45),
];
const DECODE_STEP_COUNT: usize = 1191;
const PREFILL_STEP_COUNT: usize = 1236;

/// The weight table: 600 entries naming the checkpoint's 1,698 tensors, 60,139,821,568 device
/// bytes; 324 affine weights and 45 affine MoE bundles.
const WEIGHTS: usize = 600;
const AFFINE: usize = 324;
const BUNDLES: usize = 45;
const TENSORS: usize = 1698;
const WEIGHT_BYTES: u64 = 60_139_821_568;
/// One affine MoE bundle's bytes, exactly its checkpoint's 11 tensors.
const BUNDLE_BYTES: u64 = 1_246_757_376;

/// The model's canonicals, as `#[forward]` handed them to the seam: selected, as the build script
/// that expanded `CUDA_TAPES` did, by `CARGO_FEATURE_<STEM>` and this variant's preset in
/// `SCRATCHY_QUANTS`, set once here, before the pipeline reads them; no other test in this binary
/// reads the environment. The preset replaces the 3-bit base, so exactly one input arrives.
fn input() -> TargetModelInput {
    static INPUT: std::sync::OnceLock<TargetModelInput> = std::sync::OnceLock::new();
    INPUT.get_or_init(capture).clone()
}

fn capture() -> TargetModelInput {
    // SAFETY: set before the pipeline's own threads start, and no other thread of this binary
    // reads or writes the environment.
    unsafe {
        std::env::set_var(FEATURE, "1");
        std::env::set_var("SCRATCHY_QUANTS", PRESET);
    }
    let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dsl = std::fs::read_to_string(models.join("dsl/glm4-moe.py")).expect("read the DSL");
    let carrier = scratchy_forward_compiler_macro::parse_python_file(&dsl).expect("parse the DSL");
    let inputs = scratchy_forward_compiler_macro::target_model_inputs(
        carrier,
        &models.join("configs/glm4-moe"),
    )
    .expect("the forward pipeline");
    assert_eq!(inputs.len(), 1, "one model handed to the seam");
    let mut inputs: Vec<_> = (inputs.into_iter()).filter(|i| i.stem == STEM).collect();
    assert_eq!(inputs.len(), 1, "the one model is {STEM}");
    inputs.remove(0)
}

/// The lowered tape of bucket `m`, looked up by its `bucket_m`, never by position.
fn tape_of(m: u32) -> &'static CudaTape {
    (CUDA_TAPES.iter())
        .find(|t| t.bucket_m == m)
        .unwrap_or_else(|| panic!("no m={m} tape"))
}

/// One tape's steps, by kind; plus `ExpertSteps`, every expert projection whatever its kernel.
fn histogram(tape: &CudaTape) -> BTreeMap<&'static str, usize> {
    let mut h = BTreeMap::new();
    for step in tape.steps {
        let kind = match step {
            CudaStep::Embed(_) => "Embed",
            CudaStep::Gemm(_) => "Gemm",
            CudaStep::GemmNarrow(_) => "GemmNarrow",
            CudaStep::Gemv(_) => "Gemv",
            CudaStep::RmsNorm(_) => "RmsNorm",
            CudaStep::Add(_) => "Add",
            CudaStep::SiluMul(_) => "SiluMul",
            CudaStep::Rope(_) => "Rope",
            CudaStep::GateSplit(_) => "GateSplit",
            CudaStep::GateApply(_) => "GateApply",
            CudaStep::RopeAppend(_) => "RopeAppend",
            CudaStep::AttnPrefill(_) => "AttnPrefill",
            CudaStep::AttnDecode(_) => "AttnDecode",
            CudaStep::GatedDeltaNet(_) => "GatedDeltaNet",
            CudaStep::RouterLogits(_) => "RouterLogits",
            CudaStep::MoeRoute(_) => "MoeRoute",
            CudaStep::ExpertSort(_) => "ExpertSort",
            CudaStep::ExpertGemm(_) => "ExpertGemm",
            CudaStep::ExpertGemv(_) => "ExpertGemv",
            CudaStep::ExpertCombine(_) => "ExpertCombine",
            CudaStep::GateScale(_) => "GateScale",
            CudaStep::Scale(_) => "Scale",
            CudaStep::RmsNormUnit(_) => "RmsNormUnit",
            CudaStep::GeluMul(_) => "GeluMul",
            CudaStep::LayerScale(_) => "LayerScale",
            CudaStep::SoftCap(_) => "SoftCap",
            CudaStep::RouterNorm(_) => "RouterNorm",
            CudaStep::GemmCol64(_) => "GemmCol64",
            CudaStep::BiasAdd(_) => "BiasAdd",
            CudaStep::AffineGemm(_) => "AffineGemm",
            CudaStep::AffineGemmCol64(_) => "AffineGemmCol64",
            CudaStep::AffineGemv(_) => "AffineGemv",
            CudaStep::AffineEmbed(_) => "AffineEmbed",
            CudaStep::ExpertGemmAffine(_) => "ExpertGemmAffine",
            CudaStep::ExpertGemvAffine(_) => "ExpertGemvAffine",
        };
        *h.entry(kind).or_default() += 1;
        if matches!(
            step,
            CudaStep::ExpertGemm(_)
                | CudaStep::ExpertGemv(_)
                | CudaStep::ExpertGemmAffine(_)
                | CudaStep::ExpertGemvAffine(_)
        ) {
            *h.entry("ExpertSteps").or_default() += 1;
        }
    }
    h
}

const fn end(s: ArenaSpan) -> u64 {
    s.offset.0 + s.bytes
}

#[test]
fn every_emitted_tape_is_what_cuda_lowers_from_the_same_canonical() {
    let input = input();
    assert_eq!(input.arch, "glm4-moe", "the arch the seam names");
    let model = lower_model(&input).unwrap_or_else(|e| panic!("lower_model: {e}"));
    assert_eq!(model.tapes.len(), CUDA_TAPES.len(), "tapes");
    for (tape, lowered) in CUDA_TAPES.iter().zip(&model.tapes) {
        let name = format!("{STEM} m={}", tape.bucket_m);
        chains::assert_tape_is(&name, tape, lowered);
    }
    assert_eq!(CUDA_WEIGHTS, model.weights.as_slice(), "CUDA_WEIGHTS");
}

#[test]
fn the_tapes_are_the_decode_and_prefill_buckets_and_validate() {
    let ms: Vec<u32> = CUDA_TAPES.iter().map(|t| t.bucket_m).collect();
    assert_eq!(ms, TAPE_BUCKETS, "CUDA_TAPES' buckets");
    for tape in CUDA_TAPES {
        tape.validate()
            .unwrap_or_else(|e| panic!("m={}: {e}", tape.bucket_m));
    }
    let left: Vec<u32> = CUDA_UNLOWERED.iter().map(|&(m, _)| m).collect();
    assert_eq!(left, LEFT_OUT, "CUDA_UNLOWERED's buckets");
    for &(m, why) in CUDA_UNLOWERED {
        let want = format!("m = {m} must be a non-zero multiple of 128, the bf16 GEMM's tile");
        assert!(why.contains(&want), "m={m}: {why}");
    }
}

/// The step histogram of every bucket (#948's table), the step counts, and no dense projection.
#[test]
fn every_tape_has_the_predicted_steps() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let own: &[(&str, usize)] = if m == 1 {
            &DECODE_STEPS
        } else {
            &PREFILL_STEPS
        };
        let want: BTreeMap<&str, usize> = (COMMON_STEPS.iter().chain(own).copied()).collect();
        let got = histogram(tape);
        assert_eq!(got, want, "m={m}: step histogram");
        let count = if m == 1 {
            DECODE_STEP_COUNT
        } else {
            PREFILL_STEP_COUNT
        };
        assert_eq!(tape.steps.len(), count, "m={m}: steps");
        for dense in ["Gemm", "GemmCol64", "Gemv", "GemmNarrow", "Embed"] {
            assert!(!got.contains_key(dense), "m={m}: a dense {dense}");
        }
        // Every affine step is 4-bit at the checkpoint's group, and the head is the affine embed.
        let Some(CudaStep::AffineEmbed(e)) = tape.steps.first() else {
            panic!("m={m}: the first step is not the affine embed");
        };
        assert_eq!(
            (e.rows, e.hidden, e.vocab, e.group),
            (m, HIDDEN, VOCAB, GROUP)
        );
        for s in tape.steps {
            if let CudaStep::AffineGemm(g)
            | CudaStep::AffineGemmCol64(g)
            | CudaStep::AffineGemv(g) = *s
            {
                assert_eq!(g.group, GROUP, "m={m}: {s:?}");
            }
            if let CudaStep::ExpertGemmAffine(g) = *s {
                assert_eq!(g.group, GROUP, "m={m}: {s:?}");
            }
            if let CudaStep::ExpertGemvAffine(g) = *s {
                assert_eq!(g.group, GROUP, "m={m}: {s:?}");
            }
        }
    }
}

/// One attention class over every layer, and one rotary table.
#[test]
fn the_attention_class_and_rotary_table_are_the_models() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let kv = tape.kv.unwrap_or_else(|| panic!("m={m}: no KV cache"));
        assert_eq!(kv.classes, [CLASS], "m={m}: the KV class");
        assert_eq!(kv.layers.len(), LAYERS, "m={m}: KV layers");
        assert_eq!((kv.kv_heads, kv.head_dim), (8, 128), "m={m}: the pool");
        let rotary = tape
            .rotary
            .unwrap_or_else(|| panic!("m={m}: no rotary table"));
        assert_eq!(rotary.rot_dim, ROT_DIM, "m={m}: rot_dim");
        assert_eq!(rotary.theta(), THETA, "m={m}: theta");
        assert!(tape.rotary_local.is_none(), "m={m}: a second table");
    }
}

/// The arena and MoE scratch per bucket, printed beside the issue's step-1 memory table; each
/// MoE region 256-aligned and inside the arena. The affine steps add no scratch.
#[test]
fn the_moe_scratch_and_arena_are_inside_the_arena() {
    for tape in CUDA_TAPES {
        let m = tape.bucket_m;
        let moe = tape.moe.unwrap_or_else(|| panic!("m={m}: no MoE scratch"));
        assert_eq!(
            (moe.experts, moe.top_k, moe.hidden, moe.inter),
            (EXPERTS, TOP_K, HIDDEN, INTER),
            "m={m}: MoE geometry"
        );
        let span = moe.span();
        eprintln!(
            "{STEM} m={m}: arena {} bytes, MoE scratch {} bytes at {}, max tiles {}",
            tape.arena.total_bytes, span.bytes, span.offset.0, moe.max_tiles
        );
        assert!(
            end(span) <= tape.arena.total_bytes,
            "m={m}: the scratch's end"
        );
        for (region, s) in moe.regions() {
            assert!(
                s.offset.0.is_multiple_of(256),
                "m={m}: {region:?} at {}",
                s.offset.0
            );
        }
    }
}

/// The weight table: its counts and exact device bytes, every affine entry of the checkpoint's
/// group in its scale layout, and every bundle exactly its checkpoint's 11 tensors.
#[test]
fn the_weight_table_is_the_checkpoints_tensors() {
    let entries: Vec<(usize, CudaWeight)> = (CUDA_WEIGHTS.iter().enumerate())
        .filter_map(|(ix, w)| w.map(|w| (ix, w)))
        .collect();
    assert_eq!(entries.len(), WEIGHTS, "Some entries");
    let mut names = std::collections::BTreeSet::new();
    let (mut affine, mut bundles) = (0, 0);
    for (ix, w) in &entries {
        let tensors = match *w {
            CudaWeight::Bf16 { tensor, .. } => vec![tensor],
            CudaWeight::Affine {
                weight,
                scales,
                biases,
                n,
                k,
                group,
                layout,
            } => {
                affine += 1;
                assert_eq!(group, GROUP, "weight {ix}: group");
                // The embedding table's scales are row-major, every projection's group-major.
                let want = match weight {
                    "model.embed_tokens.weight" => {
                        assert_eq!((n, k), (VOCAB, HIDDEN), "the embedding table");
                        ScaleLayout::RowMajor
                    }
                    _ => ScaleLayout::GroupMajor,
                };
                assert_eq!(layout, want, "weight {ix} ({weight}): scale layout");
                vec![weight, scales, biases]
            }
            CudaWeight::MoeBundleAffine {
                router,
                correction_bias,
                gate,
                up,
                down,
                experts,
                hidden,
                inter,
                group,
            } => {
                bundles += 1;
                assert_eq!(
                    (experts, hidden, inter, group),
                    (EXPERTS, HIDDEN, INTER, GROUP),
                    "weight {ix}: bundle geometry"
                );
                assert_eq!(w.packed_bytes(), Some(BUNDLE_BYTES), "weight {ix}: bytes");
                let mut t = vec![router, correction_bias];
                for p in [gate, up, down] {
                    t.extend([p.weight, p.scales, p.biases]);
                }
                t
            }
            CudaWeight::GdnBundle { .. }
            | CudaWeight::MoeBundle { .. }
            | CudaWeight::ExpertBank { .. } => panic!("weight {ix}: GLM has no dense bundle"),
        };
        for t in tensors {
            assert!(names.insert(t), "weight {ix}: {t} named twice");
        }
        for tape in CUDA_TAPES {
            assert_eq!(
                Some(tape.weight_bytes[*ix]),
                w.packed_bytes(),
                "m={}: weight {ix}'s declared bytes",
                tape.bucket_m
            );
        }
    }
    assert_eq!(
        (affine, bundles),
        (AFFINE, BUNDLES),
        "affine weights, bundles"
    );
    assert_eq!(names.len(), TENSORS, "distinct tensors");
    let bytes: u64 = (entries.iter())
        .map(|(_, w)| w.packed_bytes().expect("fits"))
        .sum();
    assert_eq!(bytes, WEIGHT_BYTES, "the table's device bytes");
    let layout = moe_bundle_affine(EXPERTS, HIDDEN, INTER, GROUP).expect("the bundle fits");
    assert_eq!(layout.bytes, BUNDLE_BYTES, "moe_bundle_affine's bytes");
    // GLM's routing reads the correction bias at the bundle's own slab.
    for tape in CUDA_TAPES {
        for s in tape.steps {
            if let CudaStep::MoeRoute(r) = s
                && let scratchy_target_cuda::step::RouteProgram::GlmSigmoidBias {
                    bias_offset, ..
                } = r.program
            {
                assert_eq!(bias_offset, layout.correction_bias, "m={}", tape.bucket_m);
            }
        }
    }
}

/// The model is in the CUDA registry exactly once, under its stem, holding the emitted tapes and
/// weight table themselves.
#[test]
fn the_model_is_registered_under_its_stem() {
    let entries: Vec<_> = scratchy_target_cuda::registry::entries().collect();
    let stems: Vec<&str> = entries.iter().map(|e| e.stem).collect();
    assert_eq!(stems, [STEM], "the registry");
    assert!(
        std::ptr::eq(entries[0].tapes, CUDA_TAPES),
        "the entry's tapes"
    );
    assert!(
        std::ptr::eq(entries[0].weights, CUDA_WEIGHTS),
        "the entry's weight table"
    );
    // The bucket of one row is the decode tape, and the first prefill bucket's.
    assert_eq!(tape_of(1).bucket_m, 1);
    assert_eq!(tape_of(512).bucket_m, 512);
}
