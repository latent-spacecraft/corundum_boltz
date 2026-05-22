/**
 * Bond perception — Phase 1.2 (distance-based only).
 *
 * Spatial-grid neighbor search; for each pair within
 *   cov(A) + cov(B) + tol
 * we emit a covalent bond. H-H pairs skip. Metals coordinating to non-H
 * atoms get the Metallic flag instead of plain Covalent.
 *
 * Slice 1.3+ will layer in `_struct_conn` records (explicit cross-links
 * from mmCIF) and a CCD-template lookup ahead of distance fallback,
 * mirroring Mol*'s trust order. For Phase 1.2 distance alone is enough
 * for spheres + sticks of standard proteins / nucleic acids / typical
 * cofactors.
 *
 * Output: a flat `BondData` (struct-of-typed-arrays) for direct GPU
 * upload by the stick pass. No graph object — the renderer doesn't
 * need adjacency, just the edge list.
 */
import {
  COVALENT_RADII,
  isHydrogen,
  isMetal,
} from './elements'
import type { PropertyAttributes } from '../scene/scene'

// ─────────────────────────────────────────────────────────────────────────────
// Bond flags — bitfield. Most bonds carry exactly one of {Covalent,
// Metallic}; Aromatic / Disulfide layer on top.

export const BondFlag = {
  None:      0,
  Covalent:  1 << 0,
  Aromatic:  1 << 1,
  Metallic:  1 << 2, // coordination, not a real covalent bond
  Disulfide: 1 << 3,
} as const
export type BondFlag = (typeof BondFlag)[keyof typeof BondFlag]

export interface BondData {
  count: number
  /** Indices into PropertyAttributes arrays. */
  atomA: Uint32Array
  atomB: Uint32Array
  /** Bond order — 1/2/3 nominally; 0 = unknown / coordination. */
  order: Uint8Array
  flags: Uint8Array
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial grid — fixed cell size; one Int32Array per cell for indices.
// Uniform grid is the right shape here: bonds are short (<= ~3.5 Å) and
// proteins fit comfortably in memory.

const GRID_CELL = 3.5 // Å — > 2 × max covalent radius for biomolecules
const TOL = 0.4       // Å — distance tolerance on top of cov(A) + cov(B)
const METAL_TOL = 0.7 // Å — wider for coordination bonds

interface Grid {
  cellSize: number
  origin: [number, number, number]
  nx: number
  ny: number
  nz: number
  /** Flat array indexed by (x + nx * (y + ny * z)); each entry is a
   *  Uint32Array of atom indices in that cell. Null for empty cells. */
  cells: (Uint32Array | null)[]
}

function buildGrid(
  position: Float32Array,
  atomCount: number,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  cellSize: number,
): Grid {
  // Pad bbox by one cell so atoms at the boundary still fall inside.
  const ox = bbox.min[0] - cellSize
  const oy = bbox.min[1] - cellSize
  const oz = bbox.min[2] - cellSize
  const nx = Math.max(1, Math.ceil((bbox.max[0] - ox) / cellSize) + 1)
  const ny = Math.max(1, Math.ceil((bbox.max[1] - oy) / cellSize) + 1)
  const nz = Math.max(1, Math.ceil((bbox.max[2] - oz) / cellSize) + 1)

  // First pass — count per cell.
  const counts = new Uint32Array(nx * ny * nz)
  const cellIdx = new Uint32Array(atomCount)
  for (let i = 0; i < atomCount; i++) {
    const cx = Math.min(nx - 1, Math.max(0, ((position[i * 3]     - ox) / cellSize) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((position[i * 3 + 1] - oy) / cellSize) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((position[i * 3 + 2] - oz) / cellSize) | 0))
    const ci = cx + nx * (cy + ny * cz)
    cellIdx[i] = ci
    counts[ci]++
  }

  // Allocate per-cell typed arrays.
  const cells: (Uint32Array | null)[] = new Array(nx * ny * nz).fill(null)
  const cursors = new Uint32Array(nx * ny * nz)
  for (let ci = 0; ci < counts.length; ci++) {
    if (counts[ci] > 0) cells[ci] = new Uint32Array(counts[ci])
  }
  // Second pass — fill.
  for (let i = 0; i < atomCount; i++) {
    const ci = cellIdx[i]
    cells[ci]![cursors[ci]++] = i
  }
  return { cellSize, origin: [ox, oy, oz], nx, ny, nz, cells }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bond perception.

export function computeBonds(
  attrs: PropertyAttributes,
  bbox: { min: [number, number, number]; max: [number, number, number] },
): BondData {
  const A = attrs.count
  if (A === 0) {
    return {
      count: 0,
      atomA: new Uint32Array(0),
      atomB: new Uint32Array(0),
      order: new Uint8Array(0),
      flags: new Uint8Array(0),
    }
  }

  const position = attrs.position
  const atomicNumber = attrs.atomicNumber
  const grid = buildGrid(position, A, bbox, GRID_CELL)
  const { cells, nx, ny, nz, origin, cellSize } = grid

  // Edge buffers — dynamically grown.
  const aBuf: number[] = []
  const bBuf: number[] = []
  const oBuf: number[] = []
  const fBuf: number[] = []

  // Iterate atoms; for each, query neighbors in own + 26 surrounding cells.
  // We dedupe by requiring B > A (i.e., emit each pair once).
  for (let i = 0; i < A; i++) {
    const aZ = atomicNumber[i]
    const aIsH = isHydrogen(aZ)
    const aIsMetal = isMetal(aZ)
    const aCov = COVALENT_RADII[aZ] ?? COVALENT_RADII[0]
    const ax = position[i * 3]
    const ay = position[i * 3 + 1]
    const az = position[i * 3 + 2]
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
            if (j <= i) continue // dedupe: only emit (i, j) with j > i
            const bZ = atomicNumber[j]
            const bIsH = isHydrogen(bZ)
            // Skip H-H pairs (Mol*'s convention — avoids molecular-H noise).
            if (aIsH && bIsH) continue
            const bIsMetal = isMetal(bZ)
            const bCov = COVALENT_RADII[bZ] ?? COVALENT_RADII[0]

            const dxv = position[j * 3]     - ax
            const dyv = position[j * 3 + 1] - ay
            const dzv = position[j * 3 + 2] - az
            const d2 = dxv * dxv + dyv * dyv + dzv * dzv

            // Metal-coordination uses a wider threshold; pure covalent
            // uses cov(a)+cov(b)+TOL.
            const isMetallic = (aIsMetal || bIsMetal) && !(aIsH || bIsH)
            const threshold = isMetallic
              ? aCov + bCov + METAL_TOL
              : aCov + bCov + TOL
            // Minimum bond distance guard — skip overlapping atom errors
            // (alt-locs, duplicate records) that would emit a zero-length
            // cylinder later.
            if (d2 < 0.16 /* 0.4Å */) continue
            if (d2 > threshold * threshold) continue

            aBuf.push(i)
            bBuf.push(j)
            oBuf.push(1) // order — slice 1.2 doesn't infer double/aromatic
            fBuf.push(isMetallic ? BondFlag.Metallic : BondFlag.Covalent)
          }
        }
      }
    }
  }

  return {
    count: aBuf.length,
    atomA: Uint32Array.from(aBuf),
    atomB: Uint32Array.from(bBuf),
    order: Uint8Array.from(oBuf),
    flags: Uint8Array.from(fBuf),
  }
}
