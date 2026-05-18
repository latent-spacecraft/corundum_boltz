# @corundum/boltz

**The first AlphaFold3-class protein structure predictor that runs entirely in a browser tab.**

No install. No compute account. No queue. No data egress. You give it a FASTA sequence; your own GPU folds the protein via WebGPU + ONNX Runtime Web, and a Mol\* viewer renders the result in seconds.

```
FASTA in  ─►  featurize (78 tensors)  ─►  trunk.onnx  ─►  diffusion_step.onnx × 50  ─►  confidence.onnx  ─►  mmCIF + pLDDT  ─►  Mol*
                                          (Pairformer)    (Karras schedule,            (per-residue
                                                           Kabsch reverse-diff)         B-factor)
```

## Why

[Boltz-2](https://github.com/jwohlwend/boltz) (Wohlwend et al., 2024, MIT-licensed) is an open AlphaFold3-class structure predictor. It's also notoriously painful to install — broken `trifast` kernels on Mac/Windows, CUDA wheel mismatches, MSA server dependencies. Three years after AlphaFold2, the average bench scientist still can't fold a sequence without a sysadmin's help.

Corundum's bet — call it **Rain Computing**, the inverse of cloud computing — is that the right place for this compute is the user's own machine, reached via WebGPU. Sit through one ~2 GB download; cache it in OPFS; predict any sequence forever after, on your laptop, with nothing leaving the device.

This package is the working proof.

## Quick start

```bash
# from the monorepo root
npm install
npm run dev
```

Open `https://localhost:5173/` (the dev server uses a self-signed cert via `@vitejs/plugin-basic-ssl` so WebGPU + `SharedArrayBuffer` are available on non-localhost too — the other devices on your LAN can hit `https://<your-ip>:5173/`).

In the app:

1. Click **Load engine** — fetches the three ONNX graphs (~2 GB at fp32) from HuggingFace and caches them to OPFS. First run only.
2. Paste a FASTA sequence (or click one of the example seeders: 1L2Y / 1CRN / 1UBQ).
3. Click **Predict structure** — runs trunk → diffusion (50 Karras steps) → confidence head → mmCIF, displays in Mol\*.

`Featurizer self-check` validates the in-browser featurizer against captured-from-Python goldens for 1L2Y and 1CRN (77/78 tensors byte-exact; the remaining 1 is `ref_pos`, which is RDKit-conformer-stochastic and the model is invariant to).

## Architecture

- **`src/engine/`** — Comlink-wrapped Web Worker hosting all ONNX sessions. OPFS-backed weight cache. Streaming `.onnx.data` sidecar loading for the large external-data graphs.
- **`src/acts/boltz/`** — the act itself.
  - `featurizer/` — full TypeScript port of Boltz-2's 78-tensor input pipeline, with the residue-topology JSON tables shipped under `featurizer/tables/`.
  - `orchestrate.ts` — recycling loop + Karras-schedule diffusion sampler + Kabsch reverse-diff alignment + confidence pass.
  - `math.ts` — Mulberry32 RNG, Box-Muller normals, Karras + gamma schedules, Haar-uniform random rotations, 3×3 SVD via Jacobi, weighted Kabsch.
  - `mmcif.ts` — minimal mmCIF writer with per-residue pLDDT in `B_iso_or_equiv`.
  - `models.ts` — HuggingFace manifest for the v0.1 ONNX bundle (trunk / diffusion_step / confidence at fp32 / fp16 / int8).
  - `MolViewer.tsx` — embedded Mol\* via headless `PluginContext`.
  - `featurizer/validate.ts` — golden-blob diff harness.
- **`src/debug/WebGpuDebug.tsx`** — collapsible bottom-of-page panel: adapter info, supported features, key limits (with the `maxStorageBuffersPerShaderStage` trip-wire flagged inline). See `docs/HISTORY.md` for why that one matters.

## Licensing

- **Code in this package**: MIT (see `LICENSE`).
- **Model weights**: MIT — Boltz-1 / Boltz-2 by Wohlwend et al. Served from [`latentspacecraft/boltz-2-onnx`](https://huggingface.co/latentspacecraft/boltz-2-onnx) on HuggingFace at the `v0.1` revision.
- **Mol\***: MIT.
- **Diffusion sampler lineage**: code patterns adapted from `lucidrains/alphafold3-pytorch` (MIT) and the upstream Boltz `AtomDiffusion.sample`.

## Provenance

See [`docs/HISTORY.md`](./docs/HISTORY.md) for the full build narrative — how this project came to exist, the two-repo Python-export / browser-inference arrangement, and the bug-whack chronicle (F4 reshape, F11 size-0 axes, F12 wide Concat, plus the P-numbered export issues from the boltz-dev side).
