// SPDX-License-Identifier: Apache-2.0
//! GLM-4.5-Air's two MLX variants claim only their own checkpoint (#947, criterion 2).
//!
//! A Metal `quant/mlx` build compiles both `glm_4_5_air` variants, the 3-bit base and the
//! `mlx-affine-b4-g64-qembed` one, and each must reject the other's checkpoint at load time. Their
//! configs agree on every other fingerprint axis (hidden size, layers, vocab, group size 64, no
//! `rope_scaling`, θ 1e6), so what keeps them apart is the per-role affine bit map each emits
//! (`__AFFINE_WIDTHS`), read here from the emission this very build produced (as
//! `gemma4_router_affine_fingerprint.rs` reads `gemma4_moe.rs`) and run, through the same
//! `affine_widths_agree_in` the emitted `fingerprint_matches` calls, against header-only
//! fixtures of both checkpoints: the shapes of the tensors the width gate reads, a few KB, no
//! weights.
//! - `mlx-community/GLM-4.5-Air-3bit` at `c4367db4696015335df032b8df2227814b277077`;
//! - `mlx-community/GLM-4.5-Air-4bit` at `60837794f3caafc4682dd1a9188a82c55a9100ef`.
//!
//! Both ship the router `mlp.gate.weight` dense (no `.scales`), which the 4-bit variant states
//! (`modules_to_not_convert: ["mlp.gate"]`): its bit map has no `mlp.gate` row, and the 3-bit one
//! keeps the row it has today, which the width gate passes over on a dense router.
#![cfg(all(feature = "glm-4.5-air", feature = "metal"))]

use std::collections::BTreeMap;

use scratchy_layers::weights::affine_widths_agree_in;

/// The `glm4_moe.rs` emission this build produced.
fn emission() -> String {
    std::fs::read_to_string(concat!(env!("OUT_DIR"), "/glm4_moe.rs"))
        .expect("scratchy-forwards.rs writes $OUT_DIR/glm4_moe.rs")
}

/// One variant module's body: from its `pub mod <name> {` to the next `pub mod`. `None` when this
/// build's quant scope left the variant out.
fn variant_module<'a>(src: &'a str, variant: &str) -> Option<&'a str> {
    let head = format!("pub mod {variant} {{");
    let start = src.find(&head)? + head.len();
    let body = &src[start..];
    Some(match body.find("pub mod ") {
        Some(next) => &body[..next],
        None => body,
    })
}

/// The `(pattern, bits, group)` rows of a variant's `__AFFINE_WIDTHS` const, parsed back out of
/// the token stream (`quote` spaces its tokens: `( "name" , 4 , 64 )`).
fn affine_rows(module: &str) -> Vec<(String, u32, u32)> {
    let start = module
        .find("__AFFINE_WIDTHS")
        .expect("the variant has an affine bit map");
    let body = &module[start..];
    let body = &body[body.find("= &").expect("the const's value")..];
    let body = &body[..body.find(';').expect("the const ends")];
    let squash: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    (squash.split("(\"").skip(1))
        .map(|row| {
            let (name, rest) = row.split_once("\",").expect("a quoted name");
            let mut nums = rest.trim_end_matches(['(', ')', ',', ']']).split(',');
            let mut num = || nums.next().and_then(|n| n.trim_matches(')').parse().ok());
            (
                name.to_string(),
                num().expect("bits"),
                num().expect("group"),
            )
        })
        .collect()
}

/// A checkpoint header, as the width gate reads it: tensor name to shape.
type Header = BTreeMap<String, Vec<usize>>;

/// One MLX checkpoint's header, every layer's affine triples and its dense router, at `bits` and
/// group 64: GLM-4.5-Air's geometry (hidden 4096, 46 layers, 96 + 8 + 8 heads of 128, layer 0's
/// dense MLP 10944 wide, 128 experts and a shared expert 1408 wide, vocab 151552).
fn glm_header(bits: usize) -> Header {
    const G: usize = 64;
    let mut h = Header::new();
    let mut affine = |name: &str, n: usize, k: usize| {
        h.insert(format!("{name}.weight"), vec![n, k * bits / 32]);
        h.insert(format!("{name}.scales"), vec![n, k / G]);
        h.insert(format!("{name}.biases"), vec![n, k / G]);
    };
    affine("model.embed_tokens", 151552, 4096);
    affine("lm_head", 151552, 4096);
    for l in 0..46 {
        let p = format!("model.layers.{l}");
        affine(&format!("{p}.self_attn.q_proj"), 12288, 4096);
        affine(&format!("{p}.self_attn.k_proj"), 1024, 4096);
        affine(&format!("{p}.self_attn.v_proj"), 1024, 4096);
        affine(&format!("{p}.self_attn.o_proj"), 4096, 12288);
        if l == 0 {
            affine(&format!("{p}.mlp.gate_proj"), 10944, 4096);
            affine(&format!("{p}.mlp.up_proj"), 10944, 4096);
            affine(&format!("{p}.mlp.down_proj"), 4096, 10944);
        } else {
            affine(&format!("{p}.mlp.shared_experts.gate_proj"), 1408, 4096);
            affine(&format!("{p}.mlp.shared_experts.up_proj"), 1408, 4096);
            affine(&format!("{p}.mlp.shared_experts.down_proj"), 4096, 1408);
        }
    }
    // The router is dense on both checkpoints: `[128, 4096]` BF16, no `.scales` sibling.
    for l in 1..46 {
        h.insert(format!("model.layers.{l}.mlp.gate.weight"), vec![128, 4096]);
    }
    h
}

/// Whether a variant's rows accept a checkpoint header, as `fingerprint_matches`'s width gate
/// decides it.
fn claims(rows: &[(String, u32, u32)], header: &Header) -> bool {
    let widths: Vec<(&str, u32, u32)> = rows.iter().map(|(n, b, g)| (n.as_str(), *b, *g)).collect();
    affine_widths_agree_in(
        header.keys().map(String::as_str),
        |name| header.get(name).map(Vec::as_slice),
        &widths,
    )
}

#[test]
fn each_glm_variant_claims_only_its_own_checkpoint() {
    let src = emission();
    let (b3_header, b4_header) = (glm_header(3), glm_header(4));
    let b3 = variant_module(&src, "glm_4_5_air_mlx_affine_b3_g64_qembed")
        .or_else(|| variant_module(&src, "glm_4_5_air"));
    let b4 = variant_module(&src, "glm_4_5_air_mlx_affine_b4_g64_qembed");
    assert!(
        b3.is_some() || b4.is_some(),
        "a glm-4.5-air build compiled neither GLM variant"
    );
    // With both presets in scope, both variants must be here, so the cross-checks below cannot
    // pass by checking one side only.
    let presets = scratchy_quantizations::enabled_presets();
    if ["mlx-affine-b3-g64-qembed", "mlx-affine-b4-g64-qembed"]
        .iter()
        .all(|p| presets.contains(p))
    {
        assert!(
            b3.is_some() && b4.is_some(),
            "both GLM presets are on but the emission lacks a variant: 3-bit {}, 4-bit {}",
            b3.is_some(),
            b4.is_some()
        );
    }
    if let Some(b4) = b4 {
        let rows = affine_rows(b4);
        assert!(
            !rows.iter().any(|(n, ..)| n.ends_with(".mlp.gate.weight")),
            "the 4-bit variant states its router dense, so its bit map has no mlp.gate row: \
             {rows:?}"
        );
        assert!(
            rows.iter().all(|&(_, b, g)| (b, g) == (4, 64)),
            "every affine row is 4-bit g64: {rows:?}"
        );
        assert!(
            claims(&rows, &b4_header),
            "the 4-bit variant rejects its own checkpoint"
        );
        assert!(
            !claims(&rows, &b3_header),
            "the 4-bit variant claims the 3-bit checkpoint"
        );
    }
    if let Some(b3) = b3 {
        let rows = affine_rows(b3);
        assert!(
            claims(&rows, &b3_header),
            "the 3-bit variant rejects its own checkpoint"
        );
        assert!(
            !claims(&rows, &b4_header),
            "the 3-bit variant claims the 4-bit checkpoint"
        );
    }
}
