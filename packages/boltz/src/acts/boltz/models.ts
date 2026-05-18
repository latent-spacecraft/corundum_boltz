/**
 * Boltz-2 model manifests.
 *
 * Three graphs × three precision tiers = nine manifests, assembled by a
 * builder from the v0.1 HF repo at `latentspacecraft/boltz-2-onnx@v0.1`.
 *
 * Sizes below are exact (HEAD-verified on 2026-05-15). Default exposed
 * tier is **int8** — 543 MB total — per the handoff's recommendation.
 *
 * Each graph requires both the `.onnx` and the `.onnx.data` sidecar to be
 * fetched and presented together to ORT-Web's session creator. The
 * fetcher and worker handle that via the manifest's externalData fields.
 *
 * Authoritative spec: `meta.json` in the HF repo. Treat this file as a
 * derived view of that.
 */
import type { ModelManifest } from '@/engine/models/registry'

export type BoltzPrecision = 'int8' | 'fp16' | 'fp32'
export type BoltzGraphId = 'trunk' | 'diffusion_step' | 'confidence'

const HF_BASE = 'https://huggingface.co/latentspacecraft/boltz-2-onnx/resolve/v0.1'

// Bytes per (graph, precision) — exact, HEAD-verified on 2026-05-15.
interface SizePair {
  onnx: number
  data: number
}
const SIZES: Record<BoltzPrecision, Record<BoltzGraphId, SizePair>> = {
  int8: {
    trunk:          { onnx:  26_381_999, data: 202_196_032 },
    diffusion_step: { onnx:   6_161_984, data: 280_034_304 },
    confidence:     { onnx:   3_157_473, data:  25_049_728 },
  },
  fp16: {
    trunk:          { onnx:  26_235_854, data: 403_535_232 },
    diffusion_step: { onnx:   5_391_246, data: 559_233_024 },
    confidence:     { onnx:   3_118_075, data:  49_984_000 },
  },
  fp32: {
    trunk:          { onnx:  26_700_266, data: 808_189_952 },
    diffusion_step: { onnx:   5_568_210, data: 1_118_502_912 },
    confidence:     { onnx:   3_171_768, data: 100_139_008 },
  },
}

/** v0.1 file-naming convention: `{graph}_{precisionSuffix}.onnx`. */
function fileStem(graph: BoltzGraphId, precision: BoltzPrecision): string {
  const suffix = precision === 'fp32' ? 'dyn' : precision
  return `${graph}_${suffix}`
}

const DISPLAY: Record<BoltzGraphId, string> = {
  trunk: 'Trunk · Pairformer + recycling',
  diffusion_step: 'Diffusion step · 50-step denoising loop',
  confidence: 'Confidence head · pLDDT / PAE / PDE',
}

const NOTES: Record<BoltzGraphId, string> = {
  trunk:
    'Runs once per recycling step. Outputs s, z + cached tensors fed to the diffusion graph.',
  diffusion_step:
    'Runs N times in the sampling loop (default 50). Single denoising step; the Karras schedule is JS-side.',
  confidence:
    'Runs once after sampling completes. Outputs pLDDT logits used for B-factor coloring in Mol*.',
}

/**
 * Build the ModelManifest for a single (graph, precision) pair.
 */
export function boltzManifest(
  graph: BoltzGraphId,
  precision: BoltzPrecision,
): ModelManifest {
  const stem = fileStem(graph, precision)
  const sizes = SIZES[precision][graph]
  const filename = `${stem}.onnx.data`
  return {
    id: `boltz2-${graph}-${precision}-v0.1`,
    displayName: `Boltz-2 · ${DISPLAY[graph]} · ${precision}`,
    provenance:
      'Boltz-2 (Wohlwend et al. 2024–2025, MIT) · ONNX export by latentspacecraft/boltz-dev, v0.1 branch.',
    url: `${HF_BASE}/${precision}/${stem}.onnx`,
    approxBytes: sizes.onnx,
    externalDataUrl: `${HF_BASE}/${precision}/${stem}.onnx.data`,
    externalDataApproxBytes: sizes.data,
    externalDataFilename: filename,
    // Per-precision EP routing. fp16 / fp32 go WebGPU-first with WASM
    // fallback; F11 padded the size-0 dummy axes and F12 re-exported the
    // wide Concat node as a tree of ≤8-input Concats so it fits
    // WebGPU's `maxStorageBuffersPerShaderStage` (= 8) budget.
    //
    // int8 is routed WASM-only on purpose. ORT-Web's WebGPU EP has
    // incomplete coverage for the quantized op set (`MatMulInteger`,
    // `DynamicQuantizeLinear`, et al.); the failure mode is silent
    // miscomputation rather than an error, which surfaces as exploded
    // atom coordinates after the diffusion sampler diverges. The CPU
    // (WASM+SIMD) kernels handle the quantized ops correctly and at
    // 2-4× fp32 throughput on modern AVX2/NEON, which is also the
    // path against which `boltz-dev/phase_d7_quant` validated the
    // exports. Mobile users get the 4× memory/download win without
    // the WebGPU silent-corruption risk.
    executionProviders: precision === 'int8' ? ['wasm'] : ['webgpu', 'wasm'],
    opset: 18,
    notes: NOTES[graph],
  }
}

/**
 * The trio of graphs at a given precision. Acts iterate this to load all
 * three in parallel.
 */
export function boltzBundle(precision: BoltzPrecision): {
  trunk: ModelManifest
  diffusion_step: ModelManifest
  confidence: ModelManifest
} {
  return {
    trunk: boltzManifest('trunk', precision),
    diffusion_step: boltzManifest('diffusion_step', precision),
    confidence: boltzManifest('confidence', precision),
  }
}

/** Combined size of all three graphs at a given precision. */
export function bundleApproxBytes(precision: BoltzPrecision): number {
  const s = SIZES[precision]
  return (
    s.trunk.onnx + s.trunk.data +
    s.diffusion_step.onnx + s.diffusion_step.data +
    s.confidence.onnx + s.confidence.data
  )
}

/** Pretty label for the precision picker. */
export const PRECISION_LABEL: Record<BoltzPrecision, string> = {
  int8: 'int8 · 543 MB · default',
  fp16: 'fp16 · 1.05 GB · near-PyTorch parity',
  fp32: 'fp32 · 2.06 GB · debug / reference',
}

export const PRECISIONS: readonly BoltzPrecision[] = ['int8', 'fp16', 'fp32']
export const DEFAULT_PRECISION: BoltzPrecision = 'int8'
