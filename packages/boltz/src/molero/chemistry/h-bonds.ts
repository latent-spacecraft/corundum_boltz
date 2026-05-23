/**
 * Hydrogen-bond detection — donor / acceptor pair finder.
 *
 * Donors are atoms with the `HydrogenDonor` flag (N/O/S with at least
 * one bonded hydrogen — the totalH count from the valence model).
 * Acceptors carry `HydrogenAcceptor` (O always; N when it has no H and
 * isn't a positively-charged ammonium). For predicted structures we
 * usually don't have explicit hydrogens, so the test is purely
 * geometric on the heavy atoms.
 *
 * Geometry:
 *   - heavy-atom distance ≤ maxDistance (3.5 Å standard, 4.1 Å for S)
 *   - same-residue pairs skipped (intramolecular bonds don't count)
 *   - covalently-bonded pairs not filtered (we don't have the bond list
 *     here; in practice this only adds a handful of false positives for
 *     adjacent residues — Mol* does the same)
 *
 * Output: flat array of `HBond` records, fed to the golden-arc pass.
 *
 * For full Mol*-grade detection we'd also enforce donor / acceptor angle
 * geometry (donor-H-acceptor angle within ~30° of 180°, acceptor angle
 * within geometry-specific deviation) but those require explicit Hs and
 * the valence model's `idealGeometry` field. Slice 1.9+.
 */
import type { PropertyAttributes } from '../scene/scene'
import { AtomFlag } from '../scene/scene'

export interface HBond {
  /** Atom index of the donor (HydrogenDonor flag set). */
  atomA: number
  /** Atom index of the acceptor (HydrogenAcceptor flag set). */
  atomB: number
  /** Heavy-atom distance (Å). */
  distance: number
}

export interface HBondOptions {
  /** Heavy-atom maximum distance (Å). 3.5 catches most biological
   *  H-bonds; 4.0 is permissive (includes weak bonds and water
   *  bridges). */
  maxDistance: number
  /** Looser threshold for sulfur-containing pairs (Met SD, Cys SG). */
  sulfurMaxDistance: number
}

export const DEFAULT_HBOND_OPTIONS: HBondOptions = {
  maxDistance: 3.5,
  sulfurMaxDistance: 4.1,
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial grid — acceptors only (we iterate donors and query acceptors).

interface Grid {
  origin: [number, number, number]
  cellSize: number
  nx: number
  ny: number
  nz: number
  cells: (Uint32Array | null)[]
}

function buildAcceptorGrid(
  position: Float32Array,
  flags: Uint8Array,
  count: number,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  cellSize: number,
): Grid {
  const ox = bbox.min[0] - cellSize
  const oy = bbox.min[1] - cellSize
  const oz = bbox.min[2] - cellSize
  const nx = Math.max(1, Math.ceil((bbox.max[0] - ox) / cellSize) + 1)
  const ny = Math.max(1, Math.ceil((bbox.max[1] - oy) / cellSize) + 1)
  const nz = Math.max(1, Math.ceil((bbox.max[2] - oz) / cellSize) + 1)

  const counts = new Uint32Array(nx * ny * nz)
  const cellIdx = new Uint32Array(count)
  for (let i = 0; i < count; i++) {
    if ((flags[i] & AtomFlag.HydrogenAcceptor) === 0) {
      cellIdx[i] = 0xffffffff
      continue
    }
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

export function computeHBonds(
  attrs: PropertyAttributes,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  partial?: Partial<HBondOptions>,
): HBond[] {
  const opts = { ...DEFAULT_HBOND_OPTIONS, ...partial }
  const A = attrs.count
  if (A === 0) return []

  const position = attrs.position
  const flags = attrs.flags
  const atomicNumber = attrs.atomicNumber
  const residueIndex = attrs.residueIndex

  // Use the larger of the two thresholds for the cell size so any
  // sulfur-involving pair is in the 27-cell neighborhood.
  const cellSize = Math.max(opts.maxDistance, opts.sulfurMaxDistance)
  const grid = buildAcceptorGrid(position, flags, A, bbox, cellSize)
  const { cells, nx, ny, nz, origin } = grid

  const maxD2 = opts.maxDistance * opts.maxDistance
  const sulfurMaxD2 = opts.sulfurMaxDistance * opts.sulfurMaxDistance

  const out: HBond[] = []
  for (let i = 0; i < A; i++) {
    if ((flags[i] & AtomFlag.HydrogenDonor) === 0) continue
    const ax = position[i * 3]
    const ay = position[i * 3 + 1]
    const az = position[i * 3 + 2]
    const aRes = residueIndex[i]
    const aIsS = atomicNumber[i] === 16
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
            // Same-residue pair → skip (intramolecular doesn't count).
            // Also skip the same atom (would happen if an atom carries
            // both donor + acceptor flags, e.g. a hydroxyl).
            if (j === i || residueIndex[j] === aRes) continue
            const ddx = position[j * 3]     - ax
            const ddy = position[j * 3 + 1] - ay
            const ddz = position[j * 3 + 2] - az
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz
            // Per-pair threshold: sulfur on either side → looser.
            const cap = (aIsS || atomicNumber[j] === 16) ? sulfurMaxD2 : maxD2
            if (d2 > cap) continue
            // Minimum sanity gap — anything closer than 2.0 Å is a covalent
            // bond, not an H-bond.
            if (d2 < 4) continue
            out.push({
              atomA: i,
              atomB: j,
              distance: Math.sqrt(d2),
            })
          }
        }
      }
    }
  }
  return out
}
