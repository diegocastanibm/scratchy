// SPDX-License-Identifier: Apache-2.0
//! E2 of the Gemma 4 numerics review (#978): at one position of the accuracy run's fixed text,
//! where the CUDA prefill leaves exact arithmetic. A test-only harness: it changes nothing the
//! worker runs, and records rather than asserts.
//!
//! One case per (model, position `p`):
//! - the real checkpoint, loaded through the worker's own loader (`load_weights`, after the
//!   registry check `serving` makes);
//! - the prompt's first `p` ids (the accuracy run's own, `fixtures/e2_prompt_ids_*.json`),
//!   prefilled as the serving worker runs them: one fresh forward in the smallest bucket that
//!   holds them;
//! - after each layer, row `p − 1` of its residual, and for Gemma also the residual after its
//!   attention, and at each MoE layer the router's logits row and the routing's top-8.
//!
//! How a layer's rows are read: the prefill tape cut after the step that writes them, its result
//! span that step's output, played step by step (`forward_launched`). The cut tape is the same
//! steps with the same arguments up to there, so the rows are the ones the full forward computes.
//!
//! - Gemma 4: a layer ends at its `LayerScale` (the DSL's `scalar_weight_mul`). Its attention
//!   residual is the first `NormAdd` after the previous layer's end (`post_attention_layernorm`
//!   plus the residual), whose residual operand the harness checks is that layer's output.
//! - Qwen3.5: a layer's two residual adds are its two `AddRmsNorm`s' sums (each fused with the next
//!   norm): after the mixer, then the layer's output.
//!
//! The full tape is played too (its graph, as `serving` plays it), and its next-token logits row
//! recorded, so a case shows whether it reproduces the served token.
//!
//! Each case writes `e2-<case>.json` under `CARGO_TARGET_TMPDIR` and prints its path on an `E2_OUT`
//! line. `#[ignore]`d: a case needs an H100 and its checkpoint in the local Hub cache at the
//! pinned revision. Run one case per process:
//!
//! ```text
//! cargo test --release -p scratchy-target-cuda-worker -p scratchy-target-cuda-cli \
//!     --features nvcc,model/<stem>,quant/fp8-dynamic-per-channel --test e2_layers \
//!     -- --ignored --exact --nocapture <case>
//! ```

use std::path::PathBuf;

// Force the linker to keep the compiled models (and thus their `CudaModelEntry`
// `inventory::submit!`s): nothing here names the worker crate that anchors them, so without this
// the registry is empty.
extern crate scratchy_target_cuda_models as _;

use scratchy_target_cuda::registry;
use scratchy_target_cuda::runtime::{
    CudaDevice, CudaWorker, GdnPoolConfig, GdnSequence, Input, KvPoolConfig, KvSequence,
    load_weights,
};
use scratchy_target_cuda::step::CudaStep;
use scratchy_target_cuda::tape::{ArenaOffset, ArenaSpan, CudaTape};
use scratchy_target_cuda::weights::{check, open_checkpoint};
use serde_json::{Value, json};

/// One position of one model's fixed text.
struct Case {
    name: &'static str,
    repo: &'static str,
    /// The checkpoint's pinned revision (`backends/cuda/scripts/e2e.sh`).
    revision: &'static str,
    /// The prompt's ids, a fixture of this directory.
    ids: &'static str,
    /// The text position: the context is `ids[..pos]`, its next token row `pos − 1`.
    pos: usize,
    /// The model's decoder layers, which the harness checks it finds.
    layers: usize,
    /// vLLM's and the served scratchy token at `pos` (every id the token's text maps to).
    vllm: &'static [u32],
    scratchy: &'static [u32],
}

const GEMMA: (&str, &str, &str) = (
    "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
    "ed35d7abe5d940da41b4ff06eb482feb0be8cb44",
    include_str!("fixtures/e2_prompt_ids_gemma-4-26b-a4b-it.json"),
);
const QWEN_27B: (&str, &str, &str) = (
    "axteam/Qwen3.8-27B-FP8-dynamic",
    "6349d56f935f2ecc3e3627606b6a77f5b5a47a40",
    include_str!("fixtures/e2_prompt_ids_qwen3.8-27b.json"),
);

/// Row `p − 1` of every probe's span, as f32.
fn bf16_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect()
}

/// What one cut tape reads.
#[derive(Clone, Copy, Debug)]
enum Probe {
    /// `[rows, cols]` bf16 at `offset`: a residual.
    Residual {
        layer: usize,
        what: &'static str,
        offset: ArenaOffset,
        rows: u32,
        cols: u32,
    },
    /// Gemma's router at a MoE layer: its logits `[m, experts]` and the routing's top-k indices
    /// `[m, top_k]` u32.
    Router {
        layer: usize,
        logits: ArenaOffset,
        topk: ArenaOffset,
        experts: u32,
        top_k: u32,
    },
}

/// Every probe of `tape`, each with the index of the step after which it reads.
fn probes(tape: &CudaTape, layers: usize) -> Vec<(usize, Probe)> {
    let gemma = tape
        .steps
        .iter()
        .any(|s| matches!(s, CudaStep::LayerScale(_)));
    let mut out = Vec::new();
    if gemma {
        let mut layer = 0;
        // The previous layer's output, which the attention's `NormAdd` adds to; `None` before
        // layer 0, whose residual is the scaled embedding.
        let mut prev_out: Option<ArenaOffset> = None;
        let mut attn_seen = false;
        let mut logits: Option<(ArenaOffset, u32)> = None;
        for (i, step) in tape.steps.iter().enumerate() {
            match step {
                CudaStep::NormAdd(s) if !attn_seen => {
                    if let Some(prev) = prev_out {
                        assert_eq!(
                            s.residual, prev,
                            "layer {layer}: the first NormAdd adds to the previous layer's output"
                        );
                    }
                    attn_seen = true;
                    out.push((
                        i,
                        Probe::Residual {
                            layer,
                            what: "post_attn",
                            offset: s.norm.y,
                            rows: s.norm.rows,
                            cols: s.norm.cols,
                        },
                    ));
                }
                CudaStep::RouterLogits(s) => logits = Some((s.logits, s.experts)),
                CudaStep::MoeRoute(s) => {
                    let (l, experts) = logits.take().expect("a routing after its router logits");
                    assert_eq!(
                        l, s.logits,
                        "layer {layer}: the routing reads the router's logits"
                    );
                    out.push((
                        i,
                        Probe::Router {
                            layer,
                            logits: l,
                            topk: s.topk_idx,
                            experts,
                            top_k: s.top_k,
                        },
                    ));
                }
                CudaStep::LayerScale(s) => {
                    out.push((
                        i,
                        Probe::Residual {
                            layer,
                            what: "out",
                            offset: s.out,
                            rows: s.rows,
                            cols: s.cols,
                        },
                    ));
                    prev_out = Some(s.out);
                    attn_seen = false;
                    layer += 1;
                }
                _ => {}
            }
        }
        assert_eq!(layer, layers, "one LayerScale per Gemma layer");
    } else {
        let mut k = 0;
        for (i, step) in tape.steps.iter().enumerate() {
            if let CudaStep::AddRmsNorm(s) = step {
                out.push((
                    i,
                    Probe::Residual {
                        layer: k / 2,
                        what: if k % 2 == 0 { "post_mixer" } else { "out" },
                        offset: s.sum,
                        rows: s.norm.rows,
                        cols: s.norm.cols,
                    },
                ));
                k += 1;
            }
        }
        assert_eq!(k, 2 * layers, "two AddRmsNorms per Qwen3.5 layer");
    }
    out
}

/// The checkpoint's snapshot in the local Hub cache at `revision`.
fn snapshot(repo: &str, revision: &str) -> PathBuf {
    let root = hf_hub_downloader::cache::default_root();
    let dir = hf_hub_downloader::cache::repo_dir(&root, repo)
        .join("snapshots")
        .join(revision);
    assert!(
        dir.join("config.json").is_file(),
        "{repo}@{revision}: not in the local Hub cache at {} (`hf download {repo} --revision \
         {revision}`)",
        dir.display()
    );
    dir
}

fn run(case: &Case) {
    let ids: Vec<u32> = serde_json::from_str(case.ids).expect("the prompt ids");
    let n = case.pos;
    assert!(
        n >= 2 && n <= ids.len(),
        "position {n} of {} ids",
        ids.len()
    );
    let dir = snapshot(case.repo, case.revision);
    let device = CudaDevice::open().expect("a CUDA device");

    // The model, as `serving::load_model` finds it: the first registered table the checkpoint's
    // header passes.
    let header = open_checkpoint(&dir).expect("the checkpoint's header");
    let mut refusals = Vec::new();
    let entry = registry::entries()
        .find(|e| match check(e.weights, e.tapes, &header) {
            Ok(()) => true,
            Err(err) => {
                refusals.push(format!("{}: {err}", e.stem));
                false
            }
        })
        .unwrap_or_else(|| {
            panic!(
                "no CUDA model in this build matches {} (registered: {refusals:?})",
                dir.display()
            )
        });
    drop(header);
    let weights = load_weights(&device, &dir, entry.tapes, entry.weights).expect("the weights");

    // The bucket `serving::step` runs `n` tokens in: the smallest that holds them.
    let tape = (entry.tapes.iter())
        .filter(|t| t.bucket_m as usize >= n)
        .min_by_key(|t| t.bucket_m)
        .expect("a bucket holds the prompt");
    let m = tape.bucket_m;
    let probes = probes(tape, case.layers);
    let span = |offset: ArenaOffset, bytes: u64| ArenaSpan { offset, bytes };
    let cut = |last: usize, result: ArenaSpan| CudaTape {
        steps: &tape.steps[..=last],
        result,
        greedy: None,
        ..*tape
    };
    let mut tapes = vec![*tape];
    for &(last, probe) in &probes {
        let result = match probe {
            Probe::Residual {
                offset, rows, cols, ..
            } => span(offset, u64::from(rows) * u64::from(cols) * 2),
            Probe::Router {
                logits, experts, ..
            } => {
                let moe = tape.moe.expect("a MoE tape");
                assert_eq!(
                    moe.logits.offset, logits,
                    "the router's logits are the MoE region"
                );
                span(
                    logits,
                    u64::from(m) * u64::from(experts) * moe_logit_bytes(tape, experts),
                )
            }
        };
        tapes.push(cut(last, result));
    }
    let tapes: &'static [CudaTape] = Box::leak(tapes.into_boxed_slice());

    // One fresh sequence: blocks 1.. of a pool one block larger (block 0 the scheduler's null).
    let kv = tape.kv.expect("a KV tape");
    let need = n.div_ceil(kv.block_size as usize);
    let table: Vec<u32> = (1..=need as u32).collect();
    let kv_pool = Some(KvPoolConfig {
        num_blocks: need as u32 + 1,
    });
    let gdn_pool = tape.gdn.map(|_| GdnPoolConfig { num_slots: 1 });
    let mut worker =
        CudaWorker::new(&device, tapes, &weights, kv_pool, gdn_pool).expect("the worker");
    let mut padded = ids[..n].to_vec();
    padded.resize(m as usize, 0);
    let positions: Vec<u32> = match tape.rotary {
        Some(_) => (0..m).collect(),
        None => Vec::new(),
    };
    let seq = KvSequence {
        block_table: &table,
        prefix_len: 0,
        q_len: n as u32,
    };
    let gdn = tape.gdn.map(|_| GdnSequence {
        slot: 0,
        q_len: n as u32,
        fresh: true,
    });
    let row = n as u64 - 1;

    // The full forward, as `serving` plays it: the next-token logits row.
    let rows = u64::from(m);
    let mut logits = vec![0u16; (tape.result.bytes / rows) as usize / 2];
    worker
        .forward_row(
            0,
            Input::TokenIds(&padded),
            &positions,
            Some(&seq),
            gdn.as_ref(),
            (0, &mut logits),
        )
        .expect("the full forward");
    let logits = bf16_to_f32(&logits);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    let top: Vec<Value> = order[..8].iter().map(|&i| json!([i, logits[i]])).collect();
    let pick = |ids: &[u32]| {
        ids.iter()
            .map(|&i| logits[i as usize])
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let argmax = order[0] as u32;
    println!(
        "E2 {}: bucket {m}, argmax {argmax} (served {:?}: {}; vLLM {:?}), logit vLLM's {} \
         scratchy's {}",
        case.name,
        case.scratchy,
        if case.scratchy.contains(&argmax) {
            "reproduced"
        } else {
            "NOT reproduced"
        },
        case.vllm,
        pick(case.vllm),
        pick(case.scratchy),
    );

    // Every probe: its cut tape played step by step, then row `p − 1` of what it reads.
    let mut residuals = Vec::new();
    let mut routers = Vec::new();
    for (b, &(last, probe)) in probes.iter().enumerate() {
        let cut = &tapes[b + 1];
        let mut result = vec![0u16; cut.result.bytes as usize / 2];
        worker
            .forward_launched(
                b + 1,
                Input::TokenIds(&padded),
                &positions,
                Some(&seq),
                gdn.as_ref(),
                &mut result,
            )
            .unwrap_or_else(|e| panic!("probe {b} (after step {last}): {e}"));
        match probe {
            Probe::Residual {
                layer, what, cols, ..
            } => {
                let c = cols as usize;
                let r = &result[row as usize * c..(row as usize + 1) * c];
                residuals.push(json!({"layer": layer, "what": what, "row": bf16_to_f32(r)}));
            }
            Probe::Router {
                layer,
                topk,
                experts,
                top_k,
                ..
            } => {
                let e = experts as usize;
                let width = moe_logit_bytes(tape, experts);
                let row_logits: Vec<f32> = if width == 2 {
                    bf16_to_f32(&result[row as usize * e..(row as usize + 1) * e])
                } else {
                    (result[row as usize * e * 2..(row as usize + 1) * e * 2].chunks(2))
                        .map(|w| f32::from_bits(u32::from(w[0]) | (u32::from(w[1]) << 16)))
                        .collect()
                };
                let mut idx = vec![0u16; top_k as usize * 2];
                let at = ArenaOffset(topk.0 + row * u64::from(top_k) * 4);
                worker
                    .read_arena(span(at, u64::from(top_k) * 4), &mut idx)
                    .expect("the routing's top-k row");
                let topk_ids: Vec<u32> = (idx.chunks(2))
                    .map(|w| u32::from(w[0]) | (u32::from(w[1]) << 16))
                    .collect();
                routers.push(json!({"layer": layer, "logits": row_logits, "topk": topk_ids}));
            }
        }
    }
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("e2-{}.json", case.name));
    let doc = json!({
        "case": case.name, "repo": case.repo, "revision": case.revision, "stem": entry.stem,
        "pos": n, "bucket_m": m, "argmax": argmax, "top8": top,
        "vllm": case.vllm, "scratchy": case.scratchy,
        "residuals": residuals, "routers": routers,
    });
    std::fs::write(&path, serde_json::to_vec(&doc).expect("JSON")).expect("write the record");
    println!(
        "E2_OUT {} ({} residual rows, {} router rows)",
        path.display(),
        residuals.len(),
        routers.len()
    );
}

/// Bytes per router logit in `tape`'s MoE region: its span over `[bucket_m, experts]`.
fn moe_logit_bytes(tape: &CudaTape, experts: u32) -> u64 {
    let moe = tape.moe.expect("a MoE tape");
    let per = moe.logits.bytes / (u64::from(tape.bucket_m) * u64::from(experts));
    assert!(per == 2 || per == 4, "router logits of {per} bytes");
    per
}

macro_rules! cases {
    ($($name:ident: $model:ident, $pos:literal, $layers:literal, $vllm:expr, $scratchy:expr;)*) => {$(
        #[test]
        #[ignore = "needs an H100 and the checkpoint in the local Hub cache at its pinned revision"]
        fn $name() {
            let (repo, revision, ids) = $model;
            run(&Case {
                name: stringify!($name),
                repo,
                revision,
                ids,
                pos: $pos,
                layers: $layers,
                vllm: &$vllm,
                scratchy: &$scratchy,
            });
        }
    )*};
}

// The flips the exact reference gives to vLLM by ≥ 1 nat (E1), a control where both engines agree,
// and the 27B's band at a flip of its own (vLLM right there too) and at Gemma's 2846's length.
cases! {
    gemma_2846: GEMMA, 2846, 30, [684], [282, 236764];
    gemma_1026: GEMMA, 1026, 30, [768], [573];
    gemma_2223: GEMMA, 2223, 30, [4614], [4614];
    qwen27b_1623: QWEN_27B, 1623, 64, [11924], [1203];
    qwen27b_2638: QWEN_27B, 2638, 64, [279], [799];
}
