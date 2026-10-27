// SPDX-License-Identifier: Apache-2.0
//! GLM-4.5-Air's routed MoE half at a small geometry (#938), as the shared bridge's
//! `RouterFlavor::GlmSigmoid` arm emits it (`to_wavefront.rs`, the `Moe` arm): the router's
//! projection at byte 0 of the dense shared-fused bundle, the sigmoid plus the f32 correction
//! bias (the bundle itself is the bias's operand, as the bridge binds it), the argsort and top-k of
//! the biased sigmoids, the UNBIASED sigmoids gathered and renormalised (`norm_topk_prob`), the
//! SwiGLU experts, and the combine. No routed scaling (GLM-4.5-Air's 1.0 emits none), and no
//! shared expert in the bundle: GLM's shared expert is the DSL's own ops.
//!
//! The order is HF's `Glm4MoeTopkRouter.forward` (transformers 4.54.0): `scores =
//! sigmoid(logits)`, `scores_for_choice = scores + e_score_correction_bias`, the top-k of those,
//! `topk_weights = scores.gather(topk_indices)`, divided by their sum.
//!
//! Geometry: hidden `H = 256`, 128 experts, top-8, expert width `I = 128` (a multiple of 64 and
//! not of 256, as GLM's 1408 is). Sources: 0 `x` (the embedded hidden), 1 the bundle (a `[1, 1]`
//! source; on the device the dense bundle with the bias appended, `step::moe_bundle_biased`).

use scratchy_subtile::handoff::{Expansion, ExpansionId, LoweredDecode};
use scratchy_subtile::lower::{ExpertWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    ExpertBundle, ExpertProj, GatedAct, NumExperts, RouterBundle, SharedExpertBound, SourceShape,
    SubOp, TopK,
};
use scratchy_target_cuda_compiler::canonical::{CudaModelFacts, SharedExpertOwner};

use super::chains::{GEMMA4_FACTS, with_hidden};

pub const H: u32 = 256;
pub const EXPERTS: u32 = 128;
pub const TOP_K: u32 = 8;
pub const I: u32 = 128;

/// GLM's facts for the block: its GEMMs also take the 64-column tile (`CUDA_GEMM_COL64_ARCHS`),
/// and its shared expert is the DSL's own ops.
pub const GLM_MOE_FACTS: CudaModelFacts = CudaModelFacts {
    shared_expert: SharedExpertOwner::Dsl,
    ..GEMMA4_FACTS
};

/// The block's ops at `m` rows, as the bridge emits them:
///
/// ```text
///  0 logits = RouterLogits(x, bundle)               [M, 128]
///  1 s      = RouteSigmoidBias(logits, bundle)      the sigmoid, plus the bias to choose
///  2 sorted = RouteArgsort(s)
///  3 idx    = RouteTopK(sorted)                     [M, 8]      1-3 fold into one MoeRoute
///  4 w      = RouteGatherScores(s, idx)
///  5 w      = RouteRenorm(w)                        [M, 8]      4-5 its epilogue
///  6 rows   = ExpertSort(x, idx)
///  7 gate   = ExpertMatmul(Gate, rows, rows, bundle)
///  8 up     = ExpertMatmul(Up, rows, rows, bundle)
///  9 act    = ExpertGatedAct(Silu, gate, up)
/// 10 down   = ExpertMatmul(Down, act, rows, bundle)
/// 11 back   = ExpertUnsort(down, rows)
/// 12 routed = ExpertCombine(back, w)                [M, 256]    the result
/// ```
pub fn glm_moe_input(m: u32) -> LoweringInput {
    let nz = |n| std::num::NonZeroU32::new(n).expect("non-zero");
    let (experts, k, bundle) = (
        NumExperts::new(nz(EXPERTS)),
        TopK::new(nz(TOP_K)),
        ExpertBundle::SharedFused,
    );
    let router = RouterBundle::SharedFused;
    let op = |op, inputs| OpDesc { op, m, inputs };
    let expert = |proj, n, inputs| {
        op(
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                weight: ExpertWeight::Dense,
                bundle,
            },
            inputs,
        )
    };
    use InputRef::{Ext, Op};
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: 1 },
        ],
        ops: vec![
            op(
                SubOp::RouterLogits { experts, router },
                vec![Ext(0), Ext(1)],
            ),
            op(SubOp::RouteSigmoidBias { router }, vec![Op(0), Ext(1)]),
            op(SubOp::RouteArgsort, vec![Op(1)]),
            op(SubOp::RouteTopK { k }, vec![Op(2)]),
            op(SubOp::RouteGatherScores, vec![Op(1), Op(3)]),
            op(SubOp::RouteRenorm, vec![Op(4)]),
            op(
                SubOp::ExpertSort { experts, k, bundle },
                vec![Ext(0), Op(3)],
            ),
            expert(ExpertProj::Gate, I, vec![Op(6), Op(6), Ext(1)]),
            expert(ExpertProj::Up, I, vec![Op(6), Op(6), Ext(1)]),
            op(
                SubOp::ExpertGatedAct {
                    act: GatedAct::Silu,
                },
                vec![Op(7), Op(8)],
            ),
            expert(ExpertProj::Down, H, vec![Op(9), Op(6), Ext(1)]),
            op(SubOp::ExpertUnsort, vec![Op(10), Op(6)]),
            op(
                SubOp::ExpertCombine {
                    hidden: H,
                    shared: SharedExpertBound(None),
                },
                vec![Op(11), Op(5)],
            ),
        ],
        result: 12,
    }
}

/// The block at `m` rows as the front end hands it over: source 0 the embedded hidden, every op
/// one MoE expansion.
pub fn glm_moe(m: u32) -> LoweredDecode {
    let mut l = with_hidden(glm_moe_input(m));
    for e in l.op_expansion.iter_mut() {
        *e = Some(Expansion {
            id: ExpansionId(0),
            guard: None,
        });
    }
    l
}

// ── GLM's attention prelude at head dim 128 ────────────────────────

use scratchy_subtile::handoff::SourceBinding;
use scratchy_subtile::lower::GemmWeight;
use scratchy_subtile::model_geometry::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};
use scratchy_subtile::subtile_ir::{AttnMask, EwKind, GainConvention, RopeFormTag};

/// GLM-4.5-Air's head dim, of which `ROT` lanes rotate (`partial_rotary_factor` 0.5), and a small
/// attention of its GQA group 12 (96 / 8): 12 query heads over 1 kv head.
pub const HD: u32 = 128;
pub const ROT: u32 = 64;
pub const Q_HEADS: u32 = 12;
pub const KV_HEADS: u32 = 1;
/// The hidden of the attention fixture, and its KV prefix capacity.
pub const AH: u32 = 256;
pub const PREFIX_CAPACITY: u32 = 8;

/// GLM's attention facts: one class of head dim 128, rotating 64 lanes half-rotated, θ 1e6.
pub const GLM_ATTN_FACTS: CudaModelFacts = CudaModelFacts {
    head_dim: HD,
    rot_dim: ROT,
    rope_theta: 1e6,
    ..GLM_MOE_FACTS
};

/// One GLM attention prelude at `m` rows, layer 0, with GLM's q/k/v biases:
///
/// ```text
///  0 n  = RmsNorm(x, g)                      [M, 256]
///  1 q  = n · Wq + bq   (2: BiasAdd)          [M, 12·128]
///  3 k  = n · Wk + bk   (4: BiasAdd)          [M, 128]
///  5 v  = n · Wv + bv   (6: BiasAdd)          [M, 128]
///  7 q' = RopeRotate(q, cos, sin)
///  8 k' = RopeAppend(k, cos, sin, v, pk, pv)
///  9 a  = AttnDecode(q', pk, pv, k', v)       [M, 12·128]   the result
/// ```
///
/// Sources: 0 `x` (the embedded hidden), 1 `g`, 2 `Wq`, 3 `bq`, 4 `Wk`, 5 `bk`, 6 `Wv`, 7 `bv`, 8
/// `cos`, 9 `sin`, 10 `pk`, 11 `pv` (layer 0's caches).
pub fn glm_attn_input(m: u32) -> LoweringInput {
    let (qw, kw) = (Q_HEADS * HD, KV_HEADS * HD);
    let geom = ModelAttnGeometry::mint(
        QueryHeads::new(Q_HEADS),
        KvHeads::new(KV_HEADS),
        HeadDim::new(HD),
    )
    .expect("a valid geometry");
    let op = |op, inputs| OpDesc { op, m, inputs };
    let gemm = |n, w| {
        op(
            SubOp::MatmulTile {
                n,
                weight: GemmWeight::Dense,
            },
            vec![InputRef::Op(0), InputRef::Ext(w)],
        )
    };
    let bias = |x, b| {
        op(
            SubOp::Elementwise(EwKind::BiasAdd),
            vec![InputRef::Op(x), InputRef::Ext(b)],
        )
    };
    use InputRef::{Ext, Op};
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: AH },
            SourceShape { rows: 1, cols: AH },
            SourceShape { rows: AH, cols: qw },
            SourceShape { rows: 1, cols: qw },
            SourceShape { rows: AH, cols: kw },
            SourceShape { rows: 1, cols: kw },
            SourceShape { rows: AH, cols: kw },
            SourceShape { rows: 1, cols: kw },
            SourceShape { rows: 1, cols: HD },
            SourceShape { rows: 1, cols: HD },
            SourceShape {
                rows: PREFIX_CAPACITY,
                cols: kw,
            },
            SourceShape {
                rows: PREFIX_CAPACITY,
                cols: kw,
            },
        ],
        ops: vec![
            op(
                SubOp::RmsNorm {
                    eps: 1e-5,
                    gain: GainConvention::Scale,
                },
                vec![Ext(0), Ext(1)],
            ),
            gemm(qw, 2),
            bias(1, 3),
            gemm(kw, 4),
            bias(3, 5),
            gemm(kw, 6),
            bias(5, 7),
            op(
                SubOp::rope_rotate(HeadDim::new(HD)),
                vec![Op(2), Ext(8), Ext(9)],
            ),
            op(
                SubOp::rope_append(HeadDim::new(HD), 0, AttnMask::Causal, RopeFormTag::NeoX),
                vec![Op(4), Ext(8), Ext(9), Op(6), Ext(10), Ext(11)],
            ),
            op(
                SubOp::attn_decode(
                    geom,
                    (HD as f32).powf(-0.5),
                    PREFIX_CAPACITY,
                    AttnMask::Causal,
                ),
                vec![Op(7), Ext(10), Ext(11), Op(8), Op(6)],
            ),
        ],
        result: 9,
    }
}

/// [`glm_attn_input`] as the front end hands it over: source 0 the embedded hidden, 8 and 9 the
/// global rotary table, 10 and 11 layer 0's prefix caches.
pub fn glm_attn(m: u32) -> LoweredDecode {
    let mut l = with_hidden(glm_attn_input(m));
    l.bindings[8] = SourceBinding::Cos { local: false };
    l.bindings[9] = SourceBinding::Sin { local: false };
    l.bindings[10] = SourceBinding::PrefixK { layer: 0 };
    l.bindings[11] = SourceBinding::PrefixV { layer: 0 };
    l
}
