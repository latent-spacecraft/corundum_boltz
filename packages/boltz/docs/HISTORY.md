# Corundum — Field History

*A primary-source account of the first browser-native AlphaFold-class
structure predictor, recorded at the moment the work is being lifted out of
biocircus and forged into a standalone package (`@corundum/boltz`).*

Generated 2026-05-17, while the trail is still fresh.

---

## 1. Genesis

Corundum is the spin-out. The work itself was done inside **biocircus**,
the user-facing tent show of a thesis its proprietor calls **Rain
Computing** — the inversion of cloud computing, where compute returns to
the user's own machine instead of their data leaving it.

> *"Cloud Computing moves your data up. Rain Computing brings the compute
> back down."*

Geoff Taghon — long-time bioscientist, owner of the vision and the
slightly punk Victorian-field-guide brand voice — coined the phrase and
built biocircus.io as its flagship demo: a Haeckel-plate of bioinformatics
"specimens," each one an `<act>` running entirely in the browser via
**ONNX Runtime Web + WebGPU + Transformers.js**, with **no install, no
account, no network egress** beyond model fetches.

Structure prediction sits at the top of the biocircus tool priority list
(charter `FOUNDING.md`). The first specimen was the *Foveola plicata*
tent — a Mol\*-based viewer with a deliberately disabled "Predict
structure" button, waiting for an engine. Three engines were considered:

- **SimpleFold / SF-Turbo** — Geoff's own work, but it inherits Apple's
  research-only ML Research Model License (see memory
  `sf_turbo_shares_simplefold_weights.md`). Can't be hosted on a
  commercial CDN, can't be the default.
- **ESMFold** — workable, but Xenova's int8 ESM-2 exports on Hugging Face
  are silently broken (logits ~10⁹ rather than ~20; softmax collapses to
  one-hot — memory `xenova_quantized_esm2_broken.md`). A cautionary
  precedent that shaped every quant decision downstream.
- **Boltz** — MIT-licensed, AlphaFold3-class, weights public at
  `boltz-community/boltz-2` (~6.92 GB pickle, ~1.7B params).

Boltz won on three grounds: the license, the **install pain on
Mac/Windows** that made it the highest-leverage thing to deliver via the
browser (issues
[#275 — trifast Linux-only](https://github.com/jwohlwend/boltz/issues/275),
[#638 — torch ≥ 2.6 / no CUDA wheels](https://github.com/jwohlwend/boltz/issues/638),
[#506 — CUDA 12.9 / RTX 4080+](https://github.com/jwohlwend/boltz/issues/506)),
and the absence of any prior browser AlphaFold-class deployment.
Boltz-ONNX, full stop, did not exist on Hugging Face when the work began
— confirmed via HF Hub search and `model_explorer`.

Real Boltz on a 4090 runs ~0.5–2 min for ~300 residues. The bet: a modern
Apple Silicon GPU in a tab can do it in 1–5 min for the full pipeline at
reduced step counts. **Sit through the wait once. Predict any sequence
forever after** — weights live in OPFS, the page itself becomes the
laboratory.

---

## 2. Two-project architecture

Corundum inherits the two-project arrangement that made the work
tractable. Don't collapse it.

```
boltz-dev/  (Python — conversion + validation)
    └── produces: trunk.onnx, diffusion_step.onnx, confidence.onnx,
                  meta.json   →  pushed to HF v0.1 branch

biocircus/  (TypeScript — browser inference)   →  becomes Corundum
    └── consumes the HF manifest, runs ORT-Web sessions, owns all
        loops, decodes mmCIF, draws in Mol*
```

The boundary is **not arbitrary**. It is the boundary at which a Python
shop hands work to a JS shop: ONNX graph files plus a `meta.json` contract
plus a Python reference orchestrator the JS side mirrors. Each side can
be re-staffed independently. Each side has its own pitfalls. Each side
has its own license footprint to manage.

### Why three graphs, not one

The three-graph split (trunk · diffusion_step · confidence) is the single
most important design decision. The temptation to merge — "just bake the
loops into ONNX, ship one fat file" — is what one says before
discovering:

1. **Call frequencies differ by orders of magnitude.** Trunk runs
   `recycling_steps + 1` times (default 2). Diffusion step runs
   `sampling_steps` times (default 50). Confidence runs once. Different
   loops, different progress semantics, different cancel points.
2. **JS owns progress reporting.** Per-step UI updates require per-step
   control. A single fused graph is a black box from the moment
   `session.run()` returns its promise.
3. **Quantization tier tradeoffs differ per graph.** The trunk's 48
   Pairformer blocks behave differently under int8 than the
   diffusion step's atom transformer.

The graphs are pure forward passes. **All loops live in JS.** This
constraint is load-bearing — every sentinel in the `boltz-dev/CLAUDE.md`
is there to defend it.

---

## 3. The build journey, milestone by milestone

The `boltz-dev` tree's `phase_d*` directories are the climbing pitches.
Each is named for what it shipped, not what it tried.

| Phase | Deliverable | Notes |
|------:|------|------|
| `phase0_reference/` | Boltz CLI golden outputs for 1L2Y, 1CRN, 1UBQ, 1MBA, 1AKE | Every downstream RMSD argument is grounded here. |
| `phase1_trace/` | `anatomy.md` mapping every `if` in `Boltz1.forward()` | Identified the trunk/diffusion boundary tensors `(s, z)`. |
| `phase_d0_catalog.log` | Dynamic-axis catalog for all 78 feats tensors | Source of truth for which axes are `N`, `A`, `K`, or static. |
| `phase_d1_allatom/` | All-atom mmCIF writer + atom-name/element decoders | 326/326 atoms matched PyTorch on validation. |
| `phase_d2_trunk/` | `trunk_dyn.onnx` — fp32, dynamic `N`, 9 outputs | Pairformer × 48; widest tensor on the graph. |
| `phase_d3_diffusion/` | `diffusion_step_dyn.onnx` — dynamic `A`, `K=A/32` | Atom transformer + nested token transformer. |
| `phase_d4_confidence/` | `confidence_dyn.onnx` — four logit heads | pLDDT, PAE, PDE, resolved. |
| `phase_d5_orchestrate/` | `orchestrate_v01.py` — Python reference loop | The TS port is mirrored from this file. |
| `phase_d6_validation/` | 5 targets × 3 seeds RMSD matrix | ORT inside the PT inter-seed envelope on every target. |
| `phase_d7_quant/` | fp16 + int8 of all three graphs | `onnxconverter_common.float16` + `quantize_dynamic`. |

The TS side (now Corundum) absorbed the same shape:

- `src/acts/boltz/featurizer/index.ts` — 78-tensor featurizer. The port
  of Boltz's Python input pipeline, residue topology table-driven, with
  a `validate.ts` that diffs every tensor against a Python "golden" dump
  so drift cannot hide.
- `src/acts/boltz/math.ts` — Mulberry32 RNG + Box-Muller normal,
  **Karras sigma + gamma schedules**, **Haar-uniform rotations** via
  random unit quaternions, **3×3 SVD via single-side Jacobi**, **weighted
  Kabsch rigid alignment**. The whole numerical kit needed to drive the
  diffusion sampler. ~30 lines for Kabsch, ~15 for Haar — the math is
  small once you know which math it is.
- `src/acts/boltz/orchestrate.ts` — the recycling / sampling / scoring
  loop. Mirrors `orchestrate_v01.py` line-for-line where it can.
- `src/acts/boltz/mmcif.ts` — emits all-atom mmCIF with **pLDDT × 100 in
  the `B_iso_or_equiv` column**, so Mol\*'s default "By B-factor" palette
  renders the AlphaFold-style confidence heatmap with no further
  configuration.
- `src/acts/boltz/models.ts` — the `ModelManifest` for each
  `(graph × precision)` tier, pointing at the HF v0.1 branch.

The headline result is published at
`huggingface.co/latentspacecraft/boltz-2-onnx/tree/v0.1`. Three precisions:
fp32 (1.92 GB), **fp16 (1.01 GB, the desktop default)**, **int8 (518 MB,
the browser default)**. ORT on CPU at fp32 already runs 5–10× faster
than the Boltz CLI; projected WebGPU wall time on M3 Max / 4090 for
L<400 is 5–25 s.

---

## 4. Bug-whack chronicle

Recorded with intent — these are the patches a future maintainer needs to
recognise on sight when something regresses. The `P-n` entries are
classes 1–10 of pitfalls (export-time, documented in
`boltz-dev/EXPORT_PLAN.md`); the `F-n` entries are field hotfixes that
shipped to HF after the original v0.1 push.

### Export-time pitfalls (P-series)

- **P-1 — `to_keys` is a `functools.partial`, not a tensor.** Boltz's
  `AtomEncoder.forward` returns a closure carrying a tensor and two ints.
  Fix: discard `to_keys` from the trunk wrapper; reconstruct inside the
  diffusion graph via `get_indexing_matrix(K, W, H, device)`. (Why
  `to_keys` is not a diffusion-graph input.)

- **P-2 — `cyclic_pos_enc` triggers `GuardOnDataDependentSymNode`.** A
  `torch.any(feats["cyclic_period"] > 0)` lives inside an `if`; even when
  `cyclic_period` is all-zero the tensor reduction is opaque to dynamo.
  Fix: `model.rel_pos.cyclic_pos_enc = False` on the live module before
  export. (Cyclic peptides are out of v0.1 scope.)

- **P-3 — ~1e-3 drift on the trunk's `s`.** Dynamo's ONNX optimizer
  fuses linears in a different order than PyTorch; rounding accumulates
  along the longest chain (48 Pairformer blocks). Fix: none — within
  fp32 noise; fp16's tolerance window swallows it.

- **P-4 — `onnxconverter_common.float16` misses `Cast.to` attributes.**
  Initializers convert, dtype declarations convert, but `Cast` nodes
  keep their `to = FLOAT(1)` instead of `FLOAT16(10)`. Dynamo-exported
  graphs are Cast-heavy: **2756 on the trunk, 293 on the diffusion
  step.** Fix: post-process every Cast node — if `to == 1`, set
  `to = 10`. `_patch_node_attrs_to_fp16` in `scripts/quantize.py`.

- **P-5 — `ConstantOfShape` with no `value` attribute defaults to fp32.**
  ONNX spec: bare `ConstantOfShape` emits FLOAT-zero. The converter
  doesn't *add* a value attribute it didn't see. Fix: inject an explicit
  fp16-zero `value` on each bare node. Three on the trunk.

- **P-6 — `RandomUniformLike` from no-op dropout has no fp16 CPU
  kernel.** Boltz's `get_dropout_mask` computes `torch.rand(...) >= 0`
  in eval, dynamo bakes the rand in, ORT-CPU has no fp16 kernel for it.
  **284 such nodes on the trunk.** Fix: replace every `RandomUniformLike`
  / `RandomNormalLike` with `Identity`. The input is zeros;
  `Identity(zeros) >= 0` still produces the all-ones mask the original
  math wanted.

- **P-7 — stale `value_info` annotations.** The fp16 converter doesn't
  rewrite `graph.value_info`, so ORT trusts the old fp32 types and
  faults. Fix: `del fp16_model.graph.value_info[:]`. ORT re-infers
  cleanly. (P-8 is the same issue surfacing through `quantize_dynamic`.)

- **P-9 — `token_to_rep_atom` is Cβ, not Cα. The orchestration trap.**
  Boltz's feats dict carries **two** token→atom one-hot maps:
  `token_to_center_atom` (Cα) and `token_to_rep_atom` (Cβ for
  non-Gly, used internally as the distogram input). Using the wrong
  one for output extraction writes Cβ coordinates labeled "CA" — adjacent
  Cβ atoms sit ~5.3 Å apart, the chain renders as a tangled rope, the
  bug is invariant across fp32/fp16/int8 so quantization gets blamed
  first. Sanity check: consecutive Cα-Cα must be **3.78 ± 0.04 Å**.
  Fix:
  ```python
  ca = torch.einsum("bna,bad->bnd",
                    feats["token_to_center_atom"].float(),
                    atom_coords)
  ```
  This trap also tainted earlier validation — comparing ORT Cβ against
  PyTorch Cα inflated the RMSD by ~1 Å per pair, looked like a model
  problem when it was a pipeline problem.

- **P-10 — `_rename_dynamic_axes` IndexError.** Cosmetic;
  exporter bookkeeping pass references a pruned input. Saved `.onnx` is
  correct. Fix: monkey-patch the rename pass to swallow the IndexError
  (`_install_rename_workaround()`).

- **P-11 — Dynamo captures `D // L` as data-dependent.** Inside
  `DiffusionTransformer.forward`, `bias.view(B, N, M, L, D // L)`
  serialises a Reshape whose target pulls from an input-rank that
  doesn't exist at runtime once K ≥ ~65. ORT errors with `requested
  shape {K, 32, 128, 3, 0}` — the trailing `0` is the tell. D6
  validation only went to K=52 (1AKE, L=214); the bug only fires at
  larger L. Fix: replace `D // L` with `-1` via export-time monkey-patch
  in `_patch_diffusion_transformer_bias_split()`. (This is the bug the
  F4 hotfix re-uploaded for.)

- **P-12 — Zero-axis at trace time bakes as constant 0.** When a feats
  tensor has shape `[1, ..., 0]` at trace time *and* the axis isn't
  marked as a `Dim` symbol, torch.export bakes 0 as a literal. The
  graph's contract becomes "this axis must be exactly 0." Affects all
  21 schema-compatibility dummies (`chiral_*`, `stereo_*`, `rdkit_*`,
  `planar_*`, `contact_*`, `symmetric_*`, `connected_*`) used in
  multi-chain/ligand paths but always 0 in single-chain protein.
  Manifests at the WebGPU EP — ORT-Web's WebGPU Concat kernel fails to
  compile when any input has a size-0 axis and silently corrupts the
  output. Fix: declare a shared `Q = Dim("Q", min=0, max=256)` across
  all 21 dummies so the graph accepts size-1 padding. (This is what
  the F11 hotfix re-uploaded.)

- **P-13 — Wide Concat exceeds WebGPU's per-shader storage buffer
  limit.** `DiffusionConditioning.forward` assembles `token_trans_bias`
  with `torch.cat(per_layer_projections, dim=-1)` over **24 per-layer
  outputs** (token_transformer_depth=24). torch.export captures this as
  a single flat 24-input Concat node. WGSL caps `storage` buffers at
  **8 per shader stage** on desktops, **4 on phones**. ORT-Web fires
  `[Invalid ComputePipeline "Concat"] The number of storage buffers
  (25) ... exceeds the maximum per-stage limit (8).` and **silently
  emits zeros at run time** — sessions load, the trunk's pair-bias path
  zeroes out, predicted structures collapse, **on WebGPU only; WASM is
  unaffected.** Fix: export-time monkey-patch
  `_patch_wide_concat_in_diffusion_conditioning()` replaces the flat
  cat with a balanced tree of `max_width=4` cats. Pure topology change.
  Post-patch the trunk's widest Concat has 5 inputs; numerics drift
  1.5e-3 → 1.7e-3 on `s` (sub-noise). (F12 hotfix.)

### Field hotfixes (F-series, post-launch on HF)

The hotfixes are reconstructable from the
`boltz-dev/hf_upload_v01/push_v01_*.sh` commit messages — the cleanest
single source for the bug narrative because each script pushes only the
minimum graphs the fix touches:

- **F4 — diffusion-step bias split (`push_v01_diffusion_patch.sh`).**
  Re-export of `diffusion_step` (fp32 + fp16 + int8 only) to apply the
  P-11 `D // L → -1` fix. Trunk and confidence unchanged. Commit:
  *"v0.1 hotfix F4: re-export diffusion_step with bias-split D//L → -1
  (fixes K≥65 Reshape failure)."*

- **F11 — Q-axis on all three graphs (`push_v01_q_axis.sh`).** Full
  re-push of every graph at every precision with the new shared `Q` Dim
  symbol from P-12. Unblocks the WebGPU EP; TS featurizer now pads the
  21 dummies from size-0 to size-1 (see `featurizer/index.ts` §H, with
  the F11 comment in place). The model is invariant — 1UBQ pLDDT mean
  93.1 at size-0 = pLDDT mean 93.1 at size-1.

- **F12 — Trunk tree-cat (`push_v01_treecat.sh`).** Trunk only; the
  P-13 wide-Concat tree rewrite. After F11 unblocked size-0 Concats,
  F12 fell out as the *next* WebGPU compile failure. Once both ship,
  biocircus `models.ts` can flip `executionProviders: ['webgpu',
  'wasm']` without WebGPU silently collapsing to zeros.

  The F12 class of bug — **asymmetric-backend failure where WASM looks
  fine and WebGPU silently corrupts** — is the most insidious thing in
  this whole project. Numerical validation under WASM passed. Visual
  inspection under WebGPU produced tangled noise. There is no error
  message; the kernel just emits zeros. Future maintainers should hold
  the rule: **before declaring a graph WebGPU-ready, render the output
  in Mol\* under WebGPU and verify the Cα-Cα geometry.** Numerics-only
  validation is a trap.

(F-numbers between F4 and F11 belong to in-flight TS-side and
orchestration patches whose detail isn't preserved in-sandbox; the
push-script trio is the export-side hotfix record.)

---

## 5. Other lessons worth flagging

- **OPFS + COEP `credentialless` is the durable cache combo.** Origin
  Private File System holds the multi-gigabyte ONNX bundles across
  sessions; `Cross-Origin-Embedder-Policy: credentialless` keeps the
  page cross-origin-isolated (required for threaded WASM and shared
  memory) without blocking HF's CDN. Cross-origin isolation engagement
  *lags* navigation — after changing COOP/COEP headers, hard-reload
  before reading `globalThis.crossOriginIsolated`. The MCP browser
  extension was suspected of breaking isolation in dev; it does not
  once the page reloads cleanly (see memory
  `dev_cross_origin_isolation_gotcha.md`).

- **Xenova int8 ESM-2 is broken — verify any quantized HF export before
  trusting it.** Logits in the ±10⁹ range instead of ±20. The
  dequantization step on the final logits is missing in the int8
  pipeline. fp16 is the safe default. This precedent shaped Boltz-ONNX
  quantization: every tier was re-validated against PyTorch with
  per-target RMSD before publishing.

- **RDKit conformer non-determinism on `ref_pos`.** The featurizer's
  validate.ts treats `ref_pos` as a stochastic tensor — shape and dtype
  checked, values asserted finite and bounded but not byte-equal. This
  is correct behaviour, not a bug; RDKit's ETKDG initial conformer
  generation has internal RNG state. The model is invariant to small
  `ref_pos` perturbations because it's only used as a relative reference
  encoded into the atom encoder.

- **`sigma` is a `[B]` 1-D tensor, not a scalar.** The diffusion step
  graph errors if you pass a 0-D tensor. Wrap as `Float32Array([t_hat])`
  with shape `[1]`. `t_hat = sigma_tm * (1 + gamma)`.

- **All three dynamic axes (`N`, `A`, `K`) must agree per call.** `A`
  rounds up to a multiple of `W = 32`; `K = A / W`. The host orchestrator
  owns the padding invariant. The graph reads shape from the tensors;
  no length-specific session-creation argument.

- **Don't drift back to the original `BOLTZ_ONNX_SPEC.md`.** That was
  the v0 contract (concrete N=46, Cα-only, two graphs). v0.1 supersedes
  it. The authoritative contract is `meta.json` in the HF repo root.

---

## 6. Attributions

- **Boltz-2** — Wohlwend et al., 2024–2025, MIT-licensed.
  `boltz-community/boltz-2`. [doi:10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707).
  The model. Everything Corundum does is wrap and serve it.
- **lucidrains/alphafold3-pytorch** — the open AlphaFold3 lineage Boltz's
  diffusion sampler descends from. Karras schedule, Haar-rotation
  augmentation, weighted Kabsch alignment all flow from this line.
- **ONNX Runtime Web** — the engine. fp16 boundary tensors via
  `Tensor("float16", Uint16Array, shape)`, WebGPU + WASM backends, OPFS
  external-data loading. The substrate that makes Rain Computing
  possible.
- **Mol\*** — the viewer. Its default "By B-factor" coloring is what
  makes the pLDDT-in-B-factor trick render an AlphaFold-style confidence
  ribbon with zero downstream configuration.
- **Hugging Face** — `latentspacecraft/boltz-2-onnx` is where the
  artifacts live. Xet-stored sidecar `.onnx.data` files resolve
  automatically when their `.onnx` is fetched.
- **`onnxconverter_common`** and **`onnxruntime.quantization`** — the
  conversion plumbing, warts (P-4, P-5, P-7, P-8) and all.

---

## 7. Forward note for Corundum

What's in scope right now (v0.1):
- Single-sequence, single-chain protein.
- All-atom mmCIF + PDB output with pLDDT in B-factor.
- Three precisions; int8 default for browser, fp16 default for
  desktop.

What's deferred to v0.2 / v1:
- MSA-fed inference (new export with MSA path on, JS-side MSA fetch
  and featurizer).
- Multi-chain (`asym_id` varying) — code paths are exported but
  unvalidated; v0.2 will need per-category `Q` Dim splits.
- Affinity head (Boltz-2 ligand binding).
- pTM / ipTM scalars (recompute from `pae_logits` on the TS side; the
  port of `compute_ptms` from `boltz/model/layers/confidence_utils.py`
  is ~50 lines).

The "Predict structure" button in *Foveola plicata* lit up the moment
the orchestrator passed its first 1UBQ round-trip. Corundum's mandate
is to keep that lit and add specimens around it — without losing the
discipline that put it there in the first place.

— *The historian, 2026-05-17.*
