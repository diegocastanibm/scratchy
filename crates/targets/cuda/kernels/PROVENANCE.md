# `crates/targets/cuda/kernels/` — provenance of designs not our own

The kernels here are this repo's, written against the vendored ThunderKittens
(`thunderkittens/PROVENANCE.md`). This file records the outside designs they follow.

## The megakernel's interpreter (`mk_controller.cu`, FP8 S8 M0)

- **Modelled on:** HazyResearch's Megakernels, <https://github.com/HazyResearch/Megakernels>,
  commit `7309cec801537b61fea3b50d7dfe454a6cde578e` (2025-06-02), its `include/`
  (`megakernel.cuh`, `config.cuh`, `util.cuh`, `controller/*`) and
  `demos/low-latency-llama/*`.
- **License:** MIT, "Copyright (c) 2025 HazyResearch". MIT is compatible with this repo's
  Apache-2.0.
- **What we took is the design, not the code:**
  - one persistent CTA per SM playing a per-SM instruction list of 32-int instructions;
  - a 2-stage shared instruction ring a controller warp fills;
  - 13 pages of 16 KiB of dynamic shared memory;
  - cross-SM order through global counters, a release add after the stores and a sleeping
    acquire spin before the reads.
- **No line is copied.** Their `include/` reaches its pages and ring through `reinterpret_cast`
  views, `volatile` spins, `atomicAdd` and inline PTX (`%smid`, `trap`), which the TK check
  (`tests/kernels_use_thunderkittens.rs`) refuses. Ours types each page with a TK allocator, takes
  the SM index from `blockIdx.x` with 132 CTAs, and goes through the two approved TK-GAP helpers
  (`scratchy_ops.cuh`: `gmem_counter_arrive`, `gmem_counter_wait`) and one allowlisted fence.
