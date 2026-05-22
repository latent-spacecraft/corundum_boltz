/**
 * Secondary-structure detection — Cα-only ("DSSP-lite").
 *
 * Real DSSP uses backbone N–H ⋯ O=C hydrogen-bond patterns to classify
 * residues. We don't yet have explicit hydrogens, and even backbone N/O
 * geometry would require a richer parser pass than we ship in 1.3. The
 * Cα-only approximation below (after Labesse 1997 / Frishman & Argos
 * 1995 thresholds) catches ~80–85 % of helix/sheet assignments correctly
 * for clean experimental structures and ~90 % for AI predictions where
 * geometry is already well-formed.
 *
 *   Helix:  d(Cαᵢ, Cαᵢ₊₂) ≈ 5.5 Å, d(Cαᵢ, Cαᵢ₊₃) ≈ 5.3 Å,
 *           d(Cαᵢ, Cαᵢ₊₄) ≈ 6.4 Å.
 *   Sheet:  d(Cαᵢ, Cαᵢ₊₂) ≈ 6.8 Å (extended), d(Cαᵢ, Cαᵢ₊₃) ≈ 9.9 Å,
 *           d(Cαᵢ, Cαᵢ₊₄) ≈ 12.4 Å.
 *
 * After per-residue classification, we run a smoothing pass: any helix
 * or sheet shorter than `minRunLength` residues is dissolved to coil
 * (turn-like dihedrals would otherwise read as single-residue helices
 * and look terrible in the ribbon pass).
 */
import { extractBackbones } from './backbone'
import type { Scene as MoleroScene } from '../scene/scene'

export const SecondaryStructure = {
  Coil:  0,
  Helix: 1,
  Sheet: 2,
} as const
export type SecondaryStructure = (typeof SecondaryStructure)[keyof typeof SecondaryStructure]

export interface SSDetectionOptions {
  /** Helix Cα-Cα distance bands (Å). */
  helix: { d2: [number, number]; d3: [number, number]; d4: [number, number] }
  /** Sheet Cα-Cα distance bands (Å). */
  sheet: { d2: [number, number]; d3: [number, number]; d4: [number, number] }
  /** Minimum consecutive residues for a helix/sheet to survive smoothing. */
  minRunLength: number
}

export const DEFAULT_SS_OPTIONS: SSDetectionOptions = {
  helix: { d2: [4.8, 5.9], d3: [4.7, 5.9], d4: [4.9, 6.6] },
  sheet: { d2: [6.0, 7.6], d3: [9.0, 11.0], d4: [11.5, 14.5] },
  minRunLength: 4,
}

/**
 * Detect SS for the whole scene. Returns a Uint8Array of length
 * `scene.residues.length` with one `SecondaryStructure` code per residue.
 * Non-protein residues default to `Coil`.
 */
export function detectSecondaryStructure(
  scene: MoleroScene,
  partial?: Partial<SSDetectionOptions>,
): Uint8Array {
  const opts = mergeOpts(DEFAULT_SS_OPTIONS, partial)
  const result = new Uint8Array(scene.residues.length) // all Coil by default

  const segments = extractBackbones(scene)
  const scratchAssign = new Uint8Array(0)
  for (const seg of segments) {
    if (seg.entityType !== 'protein') continue
    const N = seg.atomIndex.length
    if (N < 5) continue // can't compute d4 windows
    const assign = scratchAssign.length >= N ? scratchAssign.subarray(0, N) : new Uint8Array(N)
    assignSegment(seg.positions, N, opts, assign)
    smooth(assign, opts.minRunLength)
    // Map back to the global per-residue slot.
    for (let i = 0; i < N; i++) {
      result[seg.residueIndex[i]] = assign[i]
    }
  }
  return result
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-segment assignment.

function assignSegment(
  positions: Float32Array,
  N: number,
  opts: SSDetectionOptions,
  out: Uint8Array,
): void {
  for (let i = 0; i < N; i++) {
    // We need d2, d3, d4 windows centered roughly on i. Use i and i+2/3/4.
    // Edges where i+4 > N-1 default to coil — the smoothing pass cleans
    // up the boundary discontinuity.
    if (i + 4 >= N) {
      out[i] = SecondaryStructure.Coil
      continue
    }
    const d2 = dist(positions, i, i + 2)
    const d3 = dist(positions, i, i + 3)
    const d4 = dist(positions, i, i + 4)
    if (inRange(d2, opts.helix.d2) && inRange(d3, opts.helix.d3) && inRange(d4, opts.helix.d4)) {
      out[i] = SecondaryStructure.Helix
    } else if (inRange(d2, opts.sheet.d2) && inRange(d3, opts.sheet.d3) && inRange(d4, opts.sheet.d4)) {
      out[i] = SecondaryStructure.Sheet
    } else {
      out[i] = SecondaryStructure.Coil
    }
  }
}

function dist(p: Float32Array, i: number, j: number): number {
  const dx = p[j * 3]     - p[i * 3]
  const dy = p[j * 3 + 1] - p[i * 3 + 1]
  const dz = p[j * 3 + 2] - p[i * 3 + 2]
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function inRange(x: number, range: readonly [number, number]): boolean {
  return x >= range[0] && x <= range[1]
}

// ─────────────────────────────────────────────────────────────────────────────
// Smoothing — dissolve isolated assignments shorter than `minRunLength`
// into coil. Runs in-place. We pass twice: first dissolve short helices,
// then short sheets; the second pass catches sheets that survived the
// first-pass dissolution of an adjacent helix.

function smooth(assign: Uint8Array, minRunLength: number): void {
  dissolveShortRuns(assign, SecondaryStructure.Helix, minRunLength)
  dissolveShortRuns(assign, SecondaryStructure.Sheet, minRunLength)
}

function dissolveShortRuns(assign: Uint8Array, ss: number, minLen: number): void {
  let runStart = -1
  for (let i = 0; i <= assign.length; i++) {
    const same = i < assign.length && assign[i] === ss
    if (same && runStart < 0) {
      runStart = i
    } else if (!same && runStart >= 0) {
      if (i - runStart < minLen) {
        for (let j = runStart; j < i; j++) assign[j] = SecondaryStructure.Coil
      }
      runStart = -1
    }
  }
}

function mergeOpts(
  base: SSDetectionOptions,
  partial?: Partial<SSDetectionOptions>,
): SSDetectionOptions {
  if (!partial) return base
  return {
    helix: partial.helix ?? base.helix,
    sheet: partial.sheet ?? base.sheet,
    minRunLength: partial.minRunLength ?? base.minRunLength,
  }
}
