/**
 * Boltz-2 orchestration loop.
 *
 * Mirrors `boltz-dev/scripts/orchestrate_v01.py` for B=1, single-chain
 * protein only. The three ONNX graphs (trunk, diffusion_step, confidence)
 * are already loaded into the engine worker; this module wires the
 * recycling, sampling, and confidence calls together.
 *
 * Each `engine.run(handle.id, feeds)` call goes through Comlink to the
 * worker which owns the ORT sessions. We pass tensors by name; ORT does
 * name matching internally.
 */
import { engine } from '@/engine/client'
import type { SessionHandle, TensorPayload } from '@/engine/worker'
import type { FeatsBundle, FeatsTensor } from './featsLoader'
import {
  applyAffine,
  decodePlddt,
  gammaSchedule,
  karrasSchedule,
  makeRng,
  meanCenter,
  randomRotation,
  randomTranslation,
  weightedRigidAlign,
  type Rng,
} from './math'

/** Boltz-2 v0.1 diffusion constants — pulled from meta.json. */
const DIFFUSION = {
  sigma_min: 0.0001,
  sigma_max: 160.0,
  sigma_data: 16.0,
  rho: 7.0,
  gamma_0: 0.8,
  gamma_min: 1.0,
  noise_scale: 1.003,
  step_scale: 1.5,
  alignment_reverse_diff: true,
  token_s: 384,
  token_z: 128,
} as const

export interface PredictOptions {
  feats: FeatsBundle
  trunk: SessionHandle
  diffusion: SessionHandle
  confidence: SessionHandle
  recyclingSteps?: number
  samplingSteps?: number
  seed?: number
  onProgress?: (e: ProgressEvent) => void
  /**
   * Per-step denoised coordinates for live visualization. Called after each
   * sampling step with the model's projected final structure at that level.
   * Receives a defensive copy so the caller may retain it without worrying
   * about downstream mutation. Throttle in the consumer; this fires every step.
   */
  onStep?: (denoisedCoords: Float32Array, step: number, total: number) => void
}

export type ProgressEvent =
  | { phase: 'recycling'; step: number; total: number }
  | { phase: 'sampling'; step: number; total: number; sigma: number }
  | { phase: 'confidence' }
  | { phase: 'done' }

export interface PredictResult {
  atomCoords: Float32Array      // [A * 3]
  plddt: Float32Array           // [N] in [0, 100]
  elapsedMs: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Feeds construction

/** Convert a FeatsTensor to the worker's TensorPayload shape. */
function featToPayload(t: FeatsTensor): TensorPayload {
  // Cast the bool storage to the right ORT type so the runtime knows it's bool.
  if (t.dtype === 'bool') {
    return {
      data: t.data as Uint8Array,
      dims: t.shape,
      type: 'bool',
    }
  }
  if (t.dtype === 'float16') {
    return {
      data: t.data as Uint16Array,
      dims: t.shape,
      type: 'float16',
    }
  }
  if (t.dtype === 'int64') {
    return {
      data: t.data as BigInt64Array,
      dims: t.shape,
      type: 'int64',
    }
  }
  // All other dtypes (float32, int32, int16, int8, uint8) work via inferType.
  return {
    data: t.data as TensorPayload['data'],
    dims: t.shape,
  }
}

function feedsFromFeats(feats: FeatsBundle): Record<string, TensorPayload> {
  // The exported ONNX graphs prefix every dict-arg key with `feats_` (torch.export
  // names nested-pytree leaves from the outer parameter name). E.g. the
  // featurizer's `token_index` lands as ORT input `feats_token_index`.
  const out: Record<string, TensorPayload> = {}
  for (const [name, t] of Object.entries(feats.tensors)) {
    out[`feats_${name}`] = featToPayload(t)
  }
  return out
}

function f32Payload(data: Float32Array, dims: readonly number[]): TensorPayload {
  return { data, dims, type: 'float32' }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coercion helpers (output tensors come back as TensorPayloads; coerce to Float32Array)

function asFloat32(payload: TensorPayload | undefined): Float32Array {
  if (!payload) throw new Error('Expected tensor payload')
  if (payload.data instanceof Float32Array) {
    // Copy into an ArrayBuffer-backed Float32Array regardless of source
    // (Comlink may return SharedArrayBuffer-backed views).
    return new Float32Array(payload.data)
  }
  // float16 (Uint16Array) → fp32 via JS Float16 helper would be needed here.
  // For v0.1 we assume fp32 boundaries.
  throw new Error(
    `Expected float32 tensor, got ${payload.data.constructor.name}. ` +
      `If you loaded an fp16 graph, add a fp16→fp32 unpack step.`,
  )
}

function getOutput(
  outputs: Record<string, TensorPayload>,
  name: string,
): TensorPayload {
  const v = outputs[name]
  if (!v) {
    const keys = Object.keys(outputs).join(', ')
    throw new Error(`Output ${name} missing; got: ${keys}`)
  }
  return v
}

// ─────────────────────────────────────────────────────────────────────────────
// Sampling-loop coordinate updates

function elementwiseAdd(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]
  return out
}

function scaledSub(
  a: Float32Array,
  b: Float32Array,
  scale: number,
): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - b[i]) * scale
  return out
}

function axpy(
  a: Float32Array,
  b: Float32Array,
  scale: number,
): Float32Array {
  // out = a + scale * b
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = a[i] + scale * b[i]
  return out
}

function scaleAndNoise(rng: Rng, scale: number, A: number): Float32Array {
  const out = rng.normalArray(A * 3)
  for (let i = 0; i < out.length; i++) out[i] *= scale
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point

export async function predict(opts: PredictOptions): Promise<PredictResult> {
  const tStart = performance.now()
  const recyclingSteps = opts.recyclingSteps ?? 1
  const samplingSteps = opts.samplingSteps ?? 50
  const seed = opts.seed ?? 42

  const feats = opts.feats
  const N = feats.N
  const A = feats.A

  const baseFeeds = feedsFromFeats(feats)
  const rng = makeRng(seed)

  // ─── Recycling ─────────────────────────────────────────────────────────
  let sPrev = new Float32Array(N * DIFFUSION.token_s)
  let zPrev = new Float32Array(N * N * DIFFUSION.token_z)

  let sOut: Float32Array | null = null
  let zOut: Float32Array | null = null
  let qOut: Float32Array | null = null
  let cOut: Float32Array | null = null
  let aebOut: Float32Array | null = null
  let adbOut: Float32Array | null = null
  let ttbOut: Float32Array | null = null
  let sInputsOut: Float32Array | null = null

  // We need to know K for the bias shapes:
  const K = feats.K
  const W = 32
  const H = 128

  for (let r = 0; r <= recyclingSteps; r++) {
    opts.onProgress?.({ phase: 'recycling', step: r, total: recyclingSteps + 1 })
    const trunkFeeds: Record<string, TensorPayload> = {
      ...baseFeeds,
      s_prev: f32Payload(sPrev, [1, N, DIFFUSION.token_s]),
      z_prev: f32Payload(zPrev, [1, N, N, DIFFUSION.token_z]),
    }
    const trunkOut = (await engine.run(opts.trunk.id, trunkFeeds)) as Record<
      string,
      TensorPayload
    >
    sOut = asFloat32(getOutput(trunkOut, 's'))
    zOut = asFloat32(getOutput(trunkOut, 'z'))
    qOut = asFloat32(getOutput(trunkOut, 'q'))
    cOut = asFloat32(getOutput(trunkOut, 'c'))
    aebOut = asFloat32(getOutput(trunkOut, 'atom_enc_bias'))
    adbOut = asFloat32(getOutput(trunkOut, 'atom_dec_bias'))
    ttbOut = asFloat32(getOutput(trunkOut, 'token_trans_bias'))
    sInputsOut = asFloat32(getOutput(trunkOut, 's_inputs'))
    sPrev = new Float32Array(sOut)
    zPrev = new Float32Array(zOut)
  }
  if (!sOut || !zOut || !qOut || !cOut || !aebOut || !adbOut || !ttbOut || !sInputsOut) {
    throw new Error('Recycling produced no outputs')
  }

  // ─── Sampling ──────────────────────────────────────────────────────────
  const sigmas = karrasSchedule(
    samplingSteps,
    DIFFUSION.sigma_min,
    DIFFUSION.sigma_max,
    DIFFUSION.sigma_data,
    DIFFUSION.rho,
  )
  const gammas = gammaSchedule(sigmas, DIFFUSION.gamma_0, DIFFUSION.gamma_min)

  // Initial noisy coords ~ sigma_0 * N(0, I), shape [A * 3].
  let atomCoords = scaleAndNoise(rng, sigmas[0], A)
  let atomCoordsDenoised: Float32Array | null = null

  // atom_pad_mask used as both weights and mask for the Kabsch step.
  const atomPadMask = feats.tensors['atom_pad_mask'].data as Uint8Array
  const atomMaskF = new Float32Array(A)
  for (let a = 0; a < A; a++) atomMaskF[a] = atomPadMask[a]

  // Precompute the static portion of the diffusion feeds.
  const sPayload = f32Payload(sOut, [1, N, DIFFUSION.token_s])
  const sInputsPayload = f32Payload(sInputsOut, [1, N, DIFFUSION.token_s])
  const qPayload = f32Payload(qOut, [1, A, 128])
  const cPayload = f32Payload(cOut, [1, A, 128])
  const aebPayload = f32Payload(aebOut, [1, K, W, H, 12])
  const adbPayload = f32Payload(adbOut, [1, K, W, H, 12])
  const ttbPayload = f32Payload(ttbOut, [1, N, N, DIFFUSION.token_s])

  for (let step = 0; step < samplingSteps; step++) {
    const sigmaTm = sigmas[step]
    const sigmaT = sigmas[step + 1]
    const gamma = gammas[step + 1]

    const R = randomRotation(rng)
    const tr = randomTranslation(rng, 1.0)
    atomCoords = applyAffine(meanCenter(atomCoords), R, tr)
    if (atomCoordsDenoised) {
      atomCoordsDenoised = applyAffine(meanCenter(atomCoordsDenoised), R, tr)
    }

    const tHat = sigmaTm * (1 + gamma)
    const noiseVar = DIFFUSION.noise_scale * DIFFUSION.noise_scale * (tHat * tHat - sigmaTm * sigmaTm)
    const eps = scaleAndNoise(rng, Math.sqrt(Math.max(noiseVar, 0)), A)
    let atomCoordsNoisy = elementwiseAdd(atomCoords, eps)

    const diffFeeds: Record<string, TensorPayload> = {
      ...baseFeeds,
      // The diffusion graph names the trunk single-rep input `s_trunk`
      // (the trunk graph emits it as `s`, so we rebind here).
      s_trunk: sPayload,
      s_inputs: sInputsPayload,
      q: qPayload,
      c: cPayload,
      atom_enc_bias: aebPayload,
      atom_dec_bias: adbPayload,
      token_trans_bias: ttbPayload,
      x_noisy: f32Payload(atomCoordsNoisy, [1, A, 3]),
      sigma: f32Payload(new Float32Array([tHat]), [1]),
    }
    const diffOut = (await engine.run(opts.diffusion.id, diffFeeds)) as Record<
      string,
      TensorPayload
    >
    atomCoordsDenoised = asFloat32(getOutput(diffOut, 'x_denoised'))

    if (DIFFUSION.alignment_reverse_diff) {
      atomCoordsNoisy = weightedRigidAlign(
        atomCoordsNoisy,
        atomCoordsDenoised,
        atomMaskF,
      )
    }
    const denoisedOverSigma = scaledSub(atomCoordsNoisy, atomCoordsDenoised, 1 / tHat)
    atomCoords = axpy(
      atomCoordsNoisy,
      denoisedOverSigma,
      DIFFUSION.step_scale * (sigmaT - tHat),
    )

    opts.onProgress?.({
      phase: 'sampling',
      step: step + 1,
      total: samplingSteps,
      sigma: sigmaTm,
    })
    if (opts.onStep) {
      // Hand the consumer a defensive copy. atomCoordsDenoised is the
      // model's projected final at this denoising level — emits as a
      // monotonic condensation from noise to structure when streamed.
      opts.onStep(new Float32Array(atomCoordsDenoised), step + 1, samplingSteps)
    }
  }

  // ─── Confidence ────────────────────────────────────────────────────────
  opts.onProgress?.({ phase: 'confidence' })
  const confFeeds: Record<string, TensorPayload> = {
    ...baseFeeds,
    s_inputs: sInputsPayload,
    s: sPayload,
    z: f32Payload(zOut, [1, N, N, DIFFUSION.token_z]),
    x_pred: f32Payload(atomCoords, [1, A, 3]),
  }
  const confOut = (await engine.run(opts.confidence.id, confFeeds)) as Record<
    string,
    TensorPayload
  >
  const plddtLogits = asFloat32(getOutput(confOut, 'plddt_logits'))
  const plddt = decodePlddt(plddtLogits, N)

  opts.onProgress?.({ phase: 'done' })
  return {
    atomCoords,
    plddt,
    elapsedMs: performance.now() - tStart,
  }
}
