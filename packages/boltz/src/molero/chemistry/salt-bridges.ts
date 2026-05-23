/**
 * Salt-bridge detection — pairs of oppositely-charged atoms close in
 * space. The classic "ionic interaction" that holds protein structure
 * together: Arg/Lys/His side-chain nitrogens partnered with Asp/Glu
 * carboxylate oxygens.
 *
 * Detection is purely geometric, using the per-atom formal charge field
 * we already extracted in the valence model (slice 1.1):
 *   - Find all atoms with formalCharge > +chargeThreshold (cations)
 *   - For each cation, query the spatial grid for atoms with
 *     formalCharge < −chargeThreshold within maxDistance
 *   - Filter pairs in the same residue (intramolecular guanidinium /
 *     carboxylate are not salt bridges)
 *
 * Output is a flat array of `SaltBridge` records — the renderer wraps
 * each in a glowing additive cylinder.
 *
 * Slice 1.8 (H-bonds) will plug into this same spatial-grid pattern.
 */
import type { PropertyAttributes } from '../scene/scene'

export interface SaltBridge {
  /** Atom index of the cation (positive formal charge). */
  atomA: number
  /** Atom index of the anion (negative formal charge). */
  atomB: number
  /** Euclidean distance (Å) at detection time. */
  distance: number
  /** Signed formal charges of each partner (for emission weighting). */
  chargeA: number
  chargeB: number
}

export interface SaltBridgeOptions {
  /** Maximum heavy-atom distance to count as a salt bridge (Å). 4.5 is
   *  the standard biological cutoff; 5.0 is permissive (catches weaker
   *  ionic interactions). */
  maxDistance: number
  /** Minimum |charge| for an atom to count as charged. 0.25 matches the
   *  chargeBias used by AtomFlag.PositiveCharge / NegativeCharge. */
  chargeThreshold: number
}

export const DEFAULT_SALT_BRIDGE_OPTIONS: SaltBridgeOptions = {
  maxDistance: 4.5,
  chargeThreshold: 0.25,
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial grid for negative atoms only — we iterate positives and query
// negatives (smaller set typically, since most polar/charged sidechains
// are positive). Cell size = maxDistance ensures any cation's neighbors
// are in the 27 surrounding cells.

interface Grid {
  origin: [number, number, number]
  cellSize: number
  nx: number
  ny: number
  nz: number
  /** Cells store atom-index arrays of just the negative atoms. */
  cells: (Uint32Array | null)[]
}

function buildNegativeGrid(
  position: Float32Array,
  charges: Float32Array,
  count: number,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  cellSize: number,
  threshold: number,
): Grid {
  const ox = bbox.min[0] - cellSize
  const oy = bbox.min[1] - cellSize
  const oz = bbox.min[2] - cellSize
  const nx = Math.max(1, Math.ceil((bbox.max[0] - ox) / cellSize) + 1)
  const ny = Math.max(1, Math.ceil((bbox.max[1] - oy) / cellSize) + 1)
  const nz = Math.max(1, Math.ceil((bbox.max[2] - oz) / cellSize) + 1)

  // First pass — count negatives per cell.
  const counts = new Uint32Array(nx * ny * nz)
  const cellIdx = new Uint32Array(count)
  for (let i = 0; i < count; i++) {
    if (charges[i] >= -threshold) { cellIdx[i] = 0xffffffff; continue }
    const cx = Math.min(nx - 1, Math.max(0, ((position[i * 3]     - ox) / cellSize) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((position[i * 3 + 1] - oy) / cellSize) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((position[i * 3 + 2] - oz) / cellSize) | 0))
    const ci = cx + nx * (cy + ny * cz)
    cellIdx[i] = ci
    counts[ci]++
  }
  const cells: (Uint32Array | null)[] = new Array(nx * ny * nz).fill(null)
  const cursors = new Uint32Array(nx * ny * nz)
  for (let ci = 0; ci < counts.length; ci++) {
    if (counts[ci] > 0) cells[ci] = new Uint32Array(counts[ci])
  }
  for (let i = 0; i < count; i++) {
    const ci = cellIdx[i]
    if (ci === 0xffffffff) continue
    cells[ci]![cursors[ci]++] = i
  }
  return { origin: [ox, oy, oz], cellSize, nx, ny, nz, cells }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detection.

export function computeSaltBridges(
  attrs: PropertyAttributes,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  partial?: Partial<SaltBridgeOptions>,
): SaltBridge[] {
  const opts = { ...DEFAULT_SALT_BRIDGE_OPTIONS, ...partial }
  const A = attrs.count
  if (A === 0) return []

  const position = attrs.position
  const charges = attrs.formalCharge
  const residueIndex = attrs.residueIndex

  const grid = buildNegativeGrid(
    position, charges, A, bbox, opts.maxDistance, opts.chargeThreshold,
  )
  const { cells, nx, ny, nz, origin, cellSize } = grid
  const maxD2 = opts.maxDistance * opts.maxDistance

  const out: SaltBridge[] = []
  for (let i = 0; i < A; i++) {
    if (charges[i] <= opts.chargeThreshold) continue // not a cation
    const ax = position[i * 3]
    const ay = position[i * 3 + 1]
    const az = position[i * 3 + 2]
    const aRes = residueIndex[i]
    const cx = Math.min(nx - 1, Math.max(0, ((ax - origin[0]) / cellSize) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((ay - origin[1]) / cellSize) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((az - origin[2]) / cellSize) | 0))

    for (let dz = -1; dz <= 1; dz++) {
      const nz_ = cz + dz
      if (nz_ < 0 || nz_ >= nz) continue
      for (let dy = -1; dy <= 1; dy++) {
        const ny_ = cy + dy
        if (ny_ < 0 || ny_ >= ny) continue
        for (let dx = -1; dx <= 1; dx++) {
          const nx_ = cx + dx
          if (nx_ < 0 || nx_ >= nx) continue
          const ci = nx_ + nx * (ny_ + ny * nz_)
          const cell = cells[ci]
          if (!cell) continue
          for (let k = 0; k < cell.length; k++) {
            const j = cell[k]
            // Same-residue pair is not a salt bridge — Asp's two
            // carboxylate oxygens are equally negative but they don't
            // form a bridge with each other.
            if (residueIndex[j] === aRes) continue
            const ddx = position[j * 3]     - ax
            const ddy = position[j * 3 + 1] - ay
            const ddz = position[j * 3 + 2] - az
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz
            if (d2 > maxD2) continue
            out.push({
              atomA: i,
              atomB: j,
              distance: Math.sqrt(d2),
              chargeA: charges[i],
              chargeB: charges[j],
            })
          }
        }
      }
    }
  }
  return out
}
