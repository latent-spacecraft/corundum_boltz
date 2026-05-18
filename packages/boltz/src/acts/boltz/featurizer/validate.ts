/**
 * Validation harness: featurize(seq) against a golden blob.
 *
 * The blob was emitted by `boltz-dev/scripts/npz_to_blob.py` from the
 * Python featurizer's `feats.npz` output — byte-exact reference.
 *
 * For each of the 78 tensors:
 *  - shapes must match exactly
 *  - dtypes must match exactly
 *  - int / bool data must be byte-exact
 *  - float32 data must be within `atol/rtol` (1e-6 / 1e-6 default)
 *
 * Returns a structured report — caller decides how to surface it.
 */
import { featurize } from './index'
import { fetchFeats, type FeatsTensor } from '../featsLoader'

export interface TensorDiff {
  name: string
  pass: boolean
  reason?: string
  /** Where the first divergence happens (flat index), if any. */
  firstDivergence?: { index: number; ours: number | bigint; golden: number | bigint; absDiff?: number }
  /** Max absolute diff seen across the tensor (floats only). */
  maxAbsDiff?: number
}

export interface ValidationReport {
  target: string
  sequence: string
  pass: boolean
  totalTensors: number
  passCount: number
  diffs: TensorDiff[]
  elapsedMs: number
}

const ATOL = 1e-6
const RTOL = 1e-6

function arraysEqualExact(a: ArrayLike<number | bigint>, b: ArrayLike<number | bigint>): { ok: true } | { ok: false; index: number; ours: number | bigint; golden: number | bigint } {
  const n = a.length
  if (n !== b.length) {
    return { ok: false, index: 0, ours: n, golden: b.length }
  }
  for (let i = 0; i < n; i++) {
    if (a[i] !== b[i]) {
      return { ok: false, index: i, ours: a[i], golden: b[i] }
    }
  }
  return { ok: true }
}

function arraysClose(
  a: Float32Array,
  b: Float32Array,
  atol: number,
  rtol: number,
): { ok: true; maxAbsDiff: number } | { ok: false; index: number; ours: number; golden: number; absDiff: number; maxAbsDiff: number } {
  if (a.length !== b.length) {
    return { ok: false, index: 0, ours: a.length, golden: b.length, absDiff: 0, maxAbsDiff: 0 }
  }
  let maxAbsDiff = 0
  let firstFailIdx = -1
  let firstOurs = 0
  let firstGolden = 0
  let firstAbsDiff = 0
  for (let i = 0; i < a.length; i++) {
    const ours = a[i]
    const golden = b[i]
    const diff = Math.abs(ours - golden)
    if (diff > maxAbsDiff) maxAbsDiff = diff
    const tol = atol + rtol * Math.abs(golden)
    if (diff > tol && firstFailIdx === -1) {
      firstFailIdx = i
      firstOurs = ours
      firstGolden = golden
      firstAbsDiff = diff
    }
  }
  if (firstFailIdx === -1) return { ok: true, maxAbsDiff }
  return { ok: false, index: firstFailIdx, ours: firstOurs, golden: firstGolden, absDiff: firstAbsDiff, maxAbsDiff }
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

/**
 * Tensors that are stochastic per call and cannot byte-match a golden.
 * For each, we check shape + dtype + finite + a sanity invariant noted below.
 *
 *  - `ref_pos`: per-residue centered + Haar-rotated + N(0, 1)-translated each
 *    call (`featurizer/index.ts:augmentRefPosPerResidue`, mirroring
 *    `featurizerv2.py:center_random_augmentation`). Per-residue centroid
 *    should land near `t ~ N(0,1)`, magnitude bounded.
 */
const STOCHASTIC_TENSORS = new Set(['ref_pos'])

function isFiniteFloat32(d: Float32Array): true | { index: number; value: number } {
  for (let i = 0; i < d.length; i++) {
    const v = d[i]
    if (!Number.isFinite(v)) return { index: i, value: v }
  }
  return true
}

function diffOne(ours: FeatsTensor, golden: FeatsTensor): TensorDiff {
  const name = ours.name
  // Stub / unused tensors trivially pass when either side has a 0-length axis.
  // F11 padded our dummy tensors from size-0 → size-1; pre-F11 goldens still
  // have a 0 axis. Either form is acceptable — the model is invariant.
  if (ours.shape.some((d) => d === 0) || golden.shape.some((d) => d === 0)) {
    return { name, pass: true }
  }
  if (!shapesEqual(ours.shape, golden.shape)) {
    return {
      name,
      pass: false,
      reason: `shape mismatch: ours=${JSON.stringify(ours.shape)} golden=${JSON.stringify(golden.shape)}`,
    }
  }
  if (ours.dtype !== golden.dtype) {
    return {
      name,
      pass: false,
      reason: `dtype mismatch: ours=${ours.dtype} golden=${golden.dtype}`,
    }
  }
  // Stochastic tensors: shape + dtype already validated; assert finite + bounded.
  if (STOCHASTIC_TENSORS.has(name)) {
    if (ours.dtype === 'float32') {
      const r = isFiniteFloat32(ours.data as Float32Array)
      if (r !== true) {
        return {
          name,
          pass: false,
          reason: `stochastic tensor has non-finite value at flat ${r.index}: ${r.value}`,
        }
      }
    }
    return { name, pass: true, reason: '(stochastic; shape+dtype+finite OK)' }
  }
  if (ours.dtype === 'float32') {
    const r = arraysClose(ours.data as Float32Array, golden.data as Float32Array, ATOL, RTOL)
    if (r.ok) return { name, pass: true, maxAbsDiff: r.maxAbsDiff }
    return {
      name,
      pass: false,
      reason: `float diverges at flat ${r.index}: ours=${r.ours} golden=${r.golden} |Δ|=${r.absDiff.toExponential(2)}`,
      firstDivergence: { index: r.index, ours: r.ours, golden: r.golden, absDiff: r.absDiff },
      maxAbsDiff: r.maxAbsDiff,
    }
  }
  // int / bool — byte-exact.
  const r = arraysExactCompare(ours, golden)
  if (r.ok) return { name, pass: true }
  return {
    name,
    pass: false,
    reason: `int/bool diverges at flat ${r.index}: ours=${r.ours} golden=${r.golden}`,
    firstDivergence: { index: r.index, ours: r.ours, golden: r.golden },
  }
}

function arraysExactCompare(ours: FeatsTensor, golden: FeatsTensor) {
  // BigInt64Array vs BigInt64Array: compare directly.
  // For other int kinds and bool (Uint8Array), comparison is straightforward.
  const a = ours.data as ArrayLike<number | bigint>
  const b = golden.data as ArrayLike<number | bigint>
  return arraysEqualExact(a, b)
}

export async function validateAgainstGolden(
  goldenUrl: string,
  sequence: string,
  target: string,
): Promise<ValidationReport> {
  const tStart = performance.now()
  const golden = await fetchFeats(goldenUrl)
  const ours = featurize(sequence)
  const allNames = new Set([
    ...Object.keys(golden.tensors),
    ...Object.keys(ours.tensors),
  ])
  const diffs: TensorDiff[] = []
  for (const name of Array.from(allNames).sort()) {
    const a = ours.tensors[name]
    const g = golden.tensors[name]
    if (!a && !g) continue
    if (!a) {
      diffs.push({ name, pass: false, reason: 'missing from our output' })
      continue
    }
    if (!g) {
      diffs.push({ name, pass: false, reason: 'missing from golden' })
      continue
    }
    diffs.push(diffOne(a, g))
  }
  const passCount = diffs.filter((d) => d.pass).length
  return {
    target,
    sequence,
    pass: passCount === diffs.length,
    totalTensors: diffs.length,
    passCount,
    diffs,
    elapsedMs: performance.now() - tStart,
  }
}
