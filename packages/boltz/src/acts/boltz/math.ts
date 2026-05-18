/**
 * Math kit for the Boltz orchestration loop.
 *
 * Pure, dependency-free helpers. All functions assume the single-batch
 * case (B = 1) that v0.1 supports; the dims are spelled out for clarity.
 *
 * Includes:
 *   - Seeded Mulberry32 RNG + Box-Muller standard normal sampling.
 *   - Karras sigma + gamma schedules.
 *   - Haar-uniform random rotations (via random quaternions).
 *   - 3×3 SVD via single-side Jacobi.
 *   - Weighted Kabsch rigid alignment.
 *   - softmax / argmax / sum-product helpers.
 *   - pLDDT decode (logits → [0, 100] per residue).
 *
 * Mirrors `boltz/model/modules/utils.py` and `boltz/model/loss/diffusionv2.py`
 * one-to-one for the parts the orchestrator uses.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Random number generation

export interface Rng {
  next: () => number
  normal: () => number
  normalArray: (n: number) => Float32Array
}

/** Mulberry32 — small, fast, deterministic. Sufficient for seeding inference noise. */
export function makeRng(seed: number): Rng {
  let s = seed >>> 0
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  // Box-Muller standard normal. Cached spare so we don't throw half the work away.
  let hasSpare = false
  let spare = 0
  const normal = () => {
    if (hasSpare) {
      hasSpare = false
      return spare
    }
    let u = 0
    let v = 0
    while (u === 0) u = next()
    while (v === 0) v = next()
    const mag = Math.sqrt(-2 * Math.log(u))
    const z0 = mag * Math.cos(2 * Math.PI * v)
    const z1 = mag * Math.sin(2 * Math.PI * v)
    spare = z1
    hasSpare = true
    return z0
  }
  const normalArray = (n: number) => {
    const out = new Float32Array(n)
    for (let i = 0; i < n; i++) out[i] = normal()
    return out
  }
  return { next, normal, normalArray }
}

// ─────────────────────────────────────────────────────────────────────────────
// Karras schedule

/**
 * Karras-style sigma schedule. Returns `numSteps + 1` values: the schedule
 * itself, then 0.0 appended as the final step. Mirrors
 * `AtomDiffusion.sample_schedule` in Boltz.
 */
export function karrasSchedule(
  numSteps: number,
  sigmaMin: number,
  sigmaMax: number,
  sigmaData: number,
  rho: number,
): Float32Array {
  const invRho = 1 / rho
  const out = new Float32Array(numSteps + 1)
  const a = Math.pow(sigmaMax, invRho)
  const b = Math.pow(sigmaMin, invRho)
  for (let i = 0; i < numSteps; i++) {
    const t = i / (numSteps - 1)
    out[i] = Math.pow(a + t * (b - a), rho) * sigmaData
  }
  out[numSteps] = 0
  return out
}

export function gammaSchedule(
  sigmas: Float32Array,
  gamma0: number,
  gammaMin: number,
): Float32Array {
  const out = new Float32Array(sigmas.length)
  for (let i = 0; i < sigmas.length; i++) {
    out[i] = sigmas[i] > gammaMin ? gamma0 : 0
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Haar-uniform rotation

/** Convert a real-first quaternion (w, x, y, z) to a 3×3 rotation matrix (row-major). */
function quatToMat3(w: number, x: number, y: number, z: number): Float32Array {
  const twoS = 2 / (w * w + x * x + y * y + z * z)
  const m = new Float32Array(9)
  m[0] = 1 - twoS * (y * y + z * z)
  m[1] = twoS * (x * y - z * w)
  m[2] = twoS * (x * z + y * w)
  m[3] = twoS * (x * y + z * w)
  m[4] = 1 - twoS * (x * x + z * z)
  m[5] = twoS * (y * z - x * w)
  m[6] = twoS * (x * z - y * w)
  m[7] = twoS * (y * z + x * w)
  m[8] = 1 - twoS * (x * x + y * y)
  return m
}

/** Sample a Haar-uniform random 3×3 rotation matrix. */
export function randomRotation(rng: Rng): Float32Array {
  let w = rng.normal()
  let x = rng.normal()
  let y = rng.normal()
  let z = rng.normal()
  const n = Math.sqrt(w * w + x * x + y * y + z * z) || 1
  w /= n; x /= n; y /= n; z /= n
  // Boltz convention: copysign(real, real) — keep real part non-negative.
  if (w < 0) { w = -w; x = -x; y = -y; z = -z }
  return quatToMat3(w, x, y, z)
}

/** Sample a translation `[3]` with each component ~ N(0, s_trans). */
export function randomTranslation(rng: Rng, sTrans = 1.0): Float32Array {
  const t = new Float32Array(3)
  t[0] = rng.normal() * sTrans
  t[1] = rng.normal() * sTrans
  t[2] = rng.normal() * sTrans
  return t
}

/** Apply rotation + translation to a flat `[A * 3]` coordinate array, in-place return. */
export function applyAffine(coords: Float32Array, R: Float32Array, tr: Float32Array): Float32Array {
  const A = coords.length / 3 | 0
  const out = new Float32Array(coords.length)
  // boltz applies as: out = X @ R + t (row-vec convention)
  for (let a = 0; a < A; a++) {
    const x = coords[a * 3]
    const y = coords[a * 3 + 1]
    const z = coords[a * 3 + 2]
    out[a * 3]     = x * R[0] + y * R[3] + z * R[6] + tr[0]
    out[a * 3 + 1] = x * R[1] + y * R[4] + z * R[7] + tr[1]
    out[a * 3 + 2] = x * R[2] + y * R[5] + z * R[8] + tr[2]
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate utilities

/** Subtract per-batch atom mean from coordinates. `coords` shape: `[A * 3]`. */
export function meanCenter(coords: Float32Array): Float32Array {
  const A = coords.length / 3 | 0
  let mx = 0, my = 0, mz = 0
  for (let a = 0; a < A; a++) {
    mx += coords[a * 3]
    my += coords[a * 3 + 1]
    mz += coords[a * 3 + 2]
  }
  mx /= A; my /= A; mz /= A
  const out = new Float32Array(coords.length)
  for (let a = 0; a < A; a++) {
    out[a * 3]     = coords[a * 3]     - mx
    out[a * 3 + 1] = coords[a * 3 + 1] - my
    out[a * 3 + 2] = coords[a * 3 + 2] - mz
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// 3×3 SVD via single-side Jacobi (sufficient for Kabsch)

/**
 * Symmetric 3×3 eigendecomposition via Jacobi rotations.
 * Input M is column-major 9-element array (we use row-major elsewhere but
 * for symmetric eigendecomp the layout doesn't matter).
 * Returns { eigvals: [3], eigvecs: [9] (columns are eigenvectors) }.
 */
function jacobi3(input: Float32Array): { eigvals: Float32Array; eigvecs: Float32Array } {
  // Make a working copy.
  const a = new Float32Array(input)
  // V starts as identity.
  const v = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
  const N_ITER = 30
  const EPS = 1e-10

  for (let iter = 0; iter < N_ITER; iter++) {
    // Find the largest off-diagonal magnitude.
    let p = 0, q = 1
    let maxOff = Math.abs(a[1])
    if (Math.abs(a[2]) > maxOff) { maxOff = Math.abs(a[2]); p = 0; q = 2 }
    if (Math.abs(a[5]) > maxOff) { maxOff = Math.abs(a[5]); p = 1; q = 2 }
    if (maxOff < EPS) break

    const app = a[p * 3 + p]
    const aqq = a[q * 3 + q]
    const apq = a[p * 3 + q]
    const theta = (aqq - app) / (2 * apq)
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
    const c = 1 / Math.sqrt(t * t + 1)
    const s = t * c

    // Apply rotation: A' = G^T A G.
    const newApp = app - t * apq
    const newAqq = aqq + t * apq
    a[p * 3 + p] = newApp
    a[q * 3 + q] = newAqq
    a[p * 3 + q] = 0
    a[q * 3 + p] = 0
    for (let r = 0; r < 3; r++) {
      if (r === p || r === q) continue
      const arp = a[r * 3 + p]
      const arq = a[r * 3 + q]
      a[r * 3 + p] = c * arp - s * arq
      a[p * 3 + r] = a[r * 3 + p]
      a[r * 3 + q] = s * arp + c * arq
      a[q * 3 + r] = a[r * 3 + q]
    }
    // Update V.
    for (let r = 0; r < 3; r++) {
      const vrp = v[r * 3 + p]
      const vrq = v[r * 3 + q]
      v[r * 3 + p] = c * vrp - s * vrq
      v[r * 3 + q] = s * vrp + c * vrq
    }
  }
  return {
    eigvals: new Float32Array([a[0], a[4], a[8]]),
    eigvecs: v,
  }
}

interface Svd3 { U: Float32Array; sigma: Float32Array; V: Float32Array }

/**
 * SVD of a 3×3 matrix `M = U Σ V^T`. Row-major 9-element matrices.
 * Returns U, sigma (descending), V.
 */
export function svd3(M: Float32Array): Svd3 {
  // M^T M is symmetric 3x3.
  const mtm = new Float32Array(9)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += M[k * 3 + i] * M[k * 3 + j]
      mtm[i * 3 + j] = s
    }
  }
  const { eigvals, eigvecs: V } = jacobi3(mtm)
  // Sort by eigenvalue descending.
  const order = [0, 1, 2].sort((a, b) => eigvals[b] - eigvals[a])
  const sigma = new Float32Array(3)
  const Vsorted = new Float32Array(9)
  for (let k = 0; k < 3; k++) {
    sigma[k] = Math.sqrt(Math.max(0, eigvals[order[k]]))
    for (let r = 0; r < 3; r++) Vsorted[r * 3 + k] = V[r * 3 + order[k]]
  }
  // U = M V Σ^{-1}.
  const U = new Float32Array(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      if (sigma[c] < 1e-12) {
        U[r * 3 + c] = c === r ? 1 : 0
        continue
      }
      let s = 0
      for (let k = 0; k < 3; k++) s += M[r * 3 + k] * Vsorted[k * 3 + c]
      U[r * 3 + c] = s / sigma[c]
    }
  }
  return { U, sigma, V: Vsorted }
}

/** Determinant of a 3×3 matrix (row-major). */
function det3(M: Float32Array): number {
  return (
    M[0] * (M[4] * M[8] - M[5] * M[7])
    - M[1] * (M[3] * M[8] - M[5] * M[6])
    + M[2] * (M[3] * M[7] - M[4] * M[6])
  )
}

/** A = B @ C for 3×3 row-major matrices. */
function matmul3(B: Float32Array, C: Float32Array): Float32Array {
  const out = new Float32Array(9)
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0
      for (let k = 0; k < 3; k++) s += B[i * 3 + k] * C[k * 3 + j]
      out[i * 3 + j] = s
    }
  }
  return out
}

/** Transpose a 3×3 row-major matrix. */
function transpose3(M: Float32Array): Float32Array {
  return new Float32Array([
    M[0], M[3], M[6],
    M[1], M[4], M[7],
    M[2], M[5], M[8],
  ])
}

// ─────────────────────────────────────────────────────────────────────────────
// Weighted Kabsch alignment

/**
 * Align `true_coords` to `pred_coords` by weighted rigid transform.
 * All coords are `[A * 3]`. Mask is `[A]` (0/1). Mirrors
 * `weighted_rigid_align` in boltz/model/loss/diffusionv2.py for B=1.
 *
 * Returns the rigidly-aligned true coords (i.e. true rotated into pred's
 * frame, then translated to pred's centroid).
 */
export function weightedRigidAlign(
  trueCoords: Float32Array,
  predCoords: Float32Array,
  mask: Float32Array,
): Float32Array {
  const A = mask.length
  // Compute weighted centroids. weights = mask (additional per-atom weights are 1).
  let wSum = 0
  let trueCx = 0, trueCy = 0, trueCz = 0
  let predCx = 0, predCy = 0, predCz = 0
  for (let a = 0; a < A; a++) {
    const w = mask[a]
    if (!w) continue
    wSum += w
    trueCx += w * trueCoords[a * 3]
    trueCy += w * trueCoords[a * 3 + 1]
    trueCz += w * trueCoords[a * 3 + 2]
    predCx += w * predCoords[a * 3]
    predCy += w * predCoords[a * 3 + 1]
    predCz += w * predCoords[a * 3 + 2]
  }
  if (wSum === 0) return new Float32Array(trueCoords)
  trueCx /= wSum; trueCy /= wSum; trueCz /= wSum
  predCx /= wSum; predCy /= wSum; predCz /= wSum

  // Center.
  const trueC = new Float32Array(A * 3)
  const predC = new Float32Array(A * 3)
  for (let a = 0; a < A; a++) {
    trueC[a * 3] = trueCoords[a * 3] - trueCx
    trueC[a * 3 + 1] = trueCoords[a * 3 + 1] - trueCy
    trueC[a * 3 + 2] = trueCoords[a * 3 + 2] - trueCz
    predC[a * 3] = predCoords[a * 3] - predCx
    predC[a * 3 + 1] = predCoords[a * 3 + 1] - predCy
    predC[a * 3 + 2] = predCoords[a * 3 + 2] - predCz
  }

  // Cov = sum_a w_a * pred_a (outer) true_a → 3×3.
  const cov = new Float32Array(9)
  for (let a = 0; a < A; a++) {
    const w = mask[a]
    if (!w) continue
    const px = predC[a * 3], py = predC[a * 3 + 1], pz = predC[a * 3 + 2]
    const tx = trueC[a * 3], ty = trueC[a * 3 + 1], tz = trueC[a * 3 + 2]
    cov[0] += w * px * tx
    cov[1] += w * px * ty
    cov[2] += w * px * tz
    cov[3] += w * py * tx
    cov[4] += w * py * ty
    cov[5] += w * py * tz
    cov[6] += w * pz * tx
    cov[7] += w * pz * ty
    cov[8] += w * pz * tz
  }

  // SVD(cov) = U Σ V^T.
  const { U, V } = svd3(cov)
  // R = U V^T, then enforce det == 1 by flipping the last column if det < 0.
  let R = matmul3(U, transpose3(V))
  if (det3(R) < 0) {
    // Flip the third column of U and recompute.
    const Uf = new Float32Array(U)
    Uf[2] = -Uf[2]; Uf[5] = -Uf[5]; Uf[8] = -Uf[8]
    R = matmul3(Uf, transpose3(V))
  }

  // aligned = true_centered @ R^T + pred_centroid
  const Rt = transpose3(R)
  const aligned = new Float32Array(A * 3)
  for (let a = 0; a < A; a++) {
    const x = trueC[a * 3], y = trueC[a * 3 + 1], z = trueC[a * 3 + 2]
    aligned[a * 3]     = x * Rt[0] + y * Rt[3] + z * Rt[6] + predCx
    aligned[a * 3 + 1] = x * Rt[1] + y * Rt[4] + z * Rt[7] + predCy
    aligned[a * 3 + 2] = x * Rt[2] + y * Rt[5] + z * Rt[8] + predCz
  }
  return aligned
}

// ─────────────────────────────────────────────────────────────────────────────
// Numeric helpers

/** Softmax along the last axis of a flat array; `lastDim` is the size of that axis. */
export function softmaxLast(data: Float32Array | ArrayLike<number>, lastDim: number): Float32Array {
  const len = data.length
  const out = new Float32Array(len)
  for (let off = 0; off < len; off += lastDim) {
    let max = -Infinity
    for (let i = 0; i < lastDim; i++) {
      const v = data[off + i]
      if (v > max) max = v
    }
    let sum = 0
    for (let i = 0; i < lastDim; i++) {
      const e = Math.exp(data[off + i] - max)
      out[off + i] = e
      sum += e
    }
    for (let i = 0; i < lastDim; i++) out[off + i] /= sum
  }
  return out
}

/** argmax along the last axis. Returns flat [outerCount]. */
export function argmaxLast(data: ArrayLike<number>, lastDim: number): Int32Array {
  const outerCount = (data.length / lastDim) | 0
  const out = new Int32Array(outerCount)
  for (let o = 0; o < outerCount; o++) {
    const base = o * lastDim
    let bestIdx = 0
    let bestVal = data[base]
    for (let i = 1; i < lastDim; i++) {
      const v = data[base + i]
      if (v > bestVal) { bestVal = v; bestIdx = i }
    }
    out[o] = bestIdx
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// pLDDT decode

/**
 * pLDDT decode. Input is `plddt_logits` of shape `[B=1, N, 50]` flat
 * (i.e. length N*50). Returns per-residue pLDDT × 100 in `[0, 100]`.
 */
export function decodePlddt(logits: Float32Array, N: number): Float32Array {
  const probs = softmaxLast(logits, 50)
  // Bin centers: (i + 0.5) / 50, scaled to [0, 100] = i + 0.5 × 2.
  const out = new Float32Array(N)
  for (let n = 0; n < N; n++) {
    const base = n * 50
    let s = 0
    for (let i = 0; i < 50; i++) s += probs[base + i] * (i + 0.5) * 2 // *2 because bin*0.02*100 = bin*2 with centre offset
    out[n] = s
  }
  return out
}
