/**
 * SASA — Shrake-Rupley solvent-accessible surface area, per atom.
 *
 * For each atom A:
 *   1. Place N test points uniformly on a sphere of radius vdW(A) + R_solvent
 *      centered on A. Fibonacci-sphere distribution gives even coverage
 *      without trig calls per point.
 *   2. For each test point, check whether any neighbor atom B occludes it
 *      (distance from the test point to B's center < vdW(B) + R_solvent).
 *   3. SASA(A) = (unoccluded fraction) × 4π·(vdW(A) + R_solvent)².
 *
 * Spatial grid (same shape as bond perception) restricts neighbor checks
 * to atoms within a cell radius derived from max vdW; for biomolecules a
 * 6 Å cell covers all relevant occluders.
 *
 * Output units: Å². For reference, a fully buried side-chain Cβ has
 * SASA ≈ 0; a fully exposed methyl C has SASA ≈ 60–80 Å²; a fully
 * exposed Arg NH ≈ 30 Å².
 */
import { VDW_RADII, isHydrogen } from './elements'
import type { PropertyAttributes } from '../scene/scene'

export interface SASAOptions {
  /** Solvent probe radius (Å). 1.4 = water (default). */
  solventRadius: number
  /** Test points per atom. 64 gives <5% error vs analytical; 256 is gold. */
  pointCount: number
  /** Ignore hydrogen atoms entirely (skip both as the queried atom and
   *  as occluders). Common simplification since explicit H is often
   *  absent in predicted structures. */
  ignoreHydrogens: boolean
}

export const DEFAULT_SASA_OPTIONS: SASAOptions = {
  solventRadius: 1.4,
  pointCount: 64,
  ignoreHydrogens: true,
}

const FIBONACCI_GOLDEN = Math.PI * (3 - Math.sqrt(5))

/**
 * Generate `n` unit-sphere points via the Fibonacci sphere construction.
 * Returns a Float32Array of length `3n` packed [x0,y0,z0, x1,y1,z1, ...].
 */
export function fibonacciSphere(n: number): Float32Array {
  const out = new Float32Array(n * 3)
  for (let i = 0; i < n; i++) {
    // y goes from 1 (north pole) to -1 (south pole).
    const y = 1 - (2 * i + 1) / n
    const r = Math.sqrt(1 - y * y)
    const theta = FIBONACCI_GOLDEN * i
    out[i * 3]     = Math.cos(theta) * r
    out[i * 3 + 1] = y
    out[i * 3 + 2] = Math.sin(theta) * r
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial grid (mirrors bonds.ts but using max vdW + 2·R_solvent for cell size).

const MAX_SOLVENT_INCLUSIVE = 4.5 // ≈ max(vdW) + R_solvent for biomolecules

interface Grid {
  origin: [number, number, number]
  cellSize: number
  nx: number
  ny: number
  nz: number
  cells: (Uint32Array | null)[]
}

function buildGrid(
  position: Float32Array,
  atomCount: number,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  cellSize: number,
  skipMask?: Uint8Array,
): Grid {
  const ox = bbox.min[0] - cellSize
  const oy = bbox.min[1] - cellSize
  const oz = bbox.min[2] - cellSize
  const nx = Math.max(1, Math.ceil((bbox.max[0] - ox) / cellSize) + 1)
  const ny = Math.max(1, Math.ceil((bbox.max[1] - oy) / cellSize) + 1)
  const nz = Math.max(1, Math.ceil((bbox.max[2] - oz) / cellSize) + 1)
  const counts = new Uint32Array(nx * ny * nz)
  const cellIdx = new Uint32Array(atomCount)
  for (let i = 0; i < atomCount; i++) {
    if (skipMask && skipMask[i]) { cellIdx[i] = 0xffffffff; continue }
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
  for (let i = 0; i < atomCount; i++) {
    const ci = cellIdx[i]
    if (ci === 0xffffffff) continue
    cells[ci]![cursors[ci]++] = i
  }
  return { origin: [ox, oy, oz], cellSize, nx, ny, nz, cells }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-atom SASA.

export function computeSASA(
  attrs: PropertyAttributes,
  bbox: { min: [number, number, number]; max: [number, number, number] },
  partial?: Partial<SASAOptions>,
): Float32Array {
  const opts = { ...DEFAULT_SASA_OPTIONS, ...partial }
  const A = attrs.count
  const out = new Float32Array(A)
  if (A === 0) return out

  const skipMask = opts.ignoreHydrogens
    ? Uint8Array.from(attrs.atomicNumber, (z) => (isHydrogen(z) ? 1 : 0))
    : undefined
  const grid = buildGrid(attrs.position, A, bbox, MAX_SOLVENT_INCLUSIVE, skipMask)
  const { cells, nx, ny, nz, origin, cellSize } = grid

  const sphere = fibonacciSphere(opts.pointCount)
  const N = opts.pointCount
  const probe = opts.solventRadius
  const position = attrs.position
  const atomicNumber = attrs.atomicNumber

  for (let i = 0; i < A; i++) {
    if (skipMask && skipMask[i]) continue
    const riVdW = VDW_RADII[atomicNumber[i]] ?? VDW_RADII[0]
    const ri = riVdW + probe // probe-extended radius
    const ax = position[i * 3]
    const ay = position[i * 3 + 1]
    const az = position[i * 3 + 2]
    const cx = Math.min(nx - 1, Math.max(0, ((ax - origin[0]) / cellSize) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((ay - origin[1]) / cellSize) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((az - origin[2]) / cellSize) | 0))

    // Gather candidate neighbors in a one-cell radius around (cx, cy, cz).
    // Allocate a typed scratch list once per atom; it'll never exceed ~50
    // for typical protein density.
    const neighbors: number[] = []
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
            if (j === i) continue
            neighbors.push(j)
          }
        }
      }
    }

    // Stamp each test point onto the probe sphere; test occlusion.
    let exposed = 0
    for (let p = 0; p < N; p++) {
      const px = ax + ri * sphere[p * 3]
      const py = ay + ri * sphere[p * 3 + 1]
      const pz = az + ri * sphere[p * 3 + 2]
      let occluded = false
      for (let k = 0; k < neighbors.length; k++) {
        const j = neighbors[k]
        const rj = (VDW_RADII[atomicNumber[j]] ?? VDW_RADII[0]) + probe
        const dxv = position[j * 3]     - px
        const dyv = position[j * 3 + 1] - py
        const dzv = position[j * 3 + 2] - pz
        if (dxv * dxv + dyv * dyv + dzv * dzv < rj * rj) {
          occluded = true
          break
        }
      }
      if (!occluded) exposed++
    }
    // 4π r² × exposed fraction.
    out[i] = 4 * Math.PI * ri * ri * (exposed / N)
  }
  return out
}
