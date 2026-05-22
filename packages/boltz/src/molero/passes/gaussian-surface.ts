/**
 * Molecular surface — Gaussian density field + marching cubes.
 *
 * Each atom contributes a 3D Gaussian peak to a scalar field:
 *     density(x) = Σ over atoms of exp( − |x − atomᵢ|² / (2 σᵢ²) )
 * with σᵢ scaled to (vdWᵢ + probe) so the iso=0.5 surface lands at
 * approximately the SAS distance for an isolated atom but blends
 * smoothly across overlapping atoms (no Connolly cusps, no Lego
 * voxel-aligned bumps).
 *
 * This is the smoothing PyMOL applies under the hood for its molecular-
 * surface render. The hard-edged Solvent-Accessible Surface (the literal
 * union of probe-extended spheres) is more anatomically correct but
 * visually unappealing — the per-atom spheres show as cubic bumps once
 * marched. The Gaussian approximation is what every molecular viewer
 * actually uses for "show smooth surface". For mathematically correct
 * Connolly SES, slice 1.6 lands EDTSurf or a probe-shrink pass.
 *
 * Atom-first rasterization: for each atom, walk the voxels in a ~3σ
 * bbox and accumulate the Gaussian. ~1-2 M ops for a 1 k-atom protein.
 *
 * Output snapshot bakes `mc.matrixWorld` into each vertex so the result
 * is in world coords (mc.position + mc.scale handled at extraction).
 * Per-vertex `aGlass` vec4 is baked from the nearest atom — the glass
 * pass routes it into PBR channels.
 */
import {
  BufferGeometry,
  Float32BufferAttribute,
  MeshBasicMaterial,
} from 'three/webgpu'
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js'
import { computeSASA } from '../chemistry/sasa'
import { computeHydrophobicity, hydrophobicityNorm } from '../chemistry/hydrophobicity'
import { VDW_RADII, isHydrogen } from '../chemistry/elements'
import { AtomFlag, type Scene as MoleroScene } from '../scene/scene'

export interface GaussianSurfaceOptions {
  /** Marching-cubes grid resolution per axis. 80 = ~0.5 Å voxels for a
   *  typical 40 Å protein. Smoothness comes from the Gaussian kernel,
   *  not the grid — bumping this is rarely necessary. */
  resolution: number
  /** Solvent probe radius (Å). 1.4 = water. */
  probeRadius: number
  /**
   * Gaussian σ multiplier. Per-atom σ = (vdWᵢ + probe) × this. The
   * theoretical "isolated atom surface at vdW + probe" value is ≈ 0.85
   * (for isolation 0.5); bumping higher (1.0–1.2) over-smooths and
   * dissolves per-atom bumps into a single continuous envelope at the
   * cost of slightly fattening the surface around isolated atoms.
   */
  sigmaFactor: number
  /** Marching-cubes iso threshold. 0.5 ≈ isolated-atom surface; lower
   *  values widen the envelope (catches more inter-atom blending). */
  isolation: number
  /** Bbox padding beyond the atom centers (Å). Must clear the Gaussian
   *  tail (~3σ) so the surface doesn't clip at the box wall. */
  padding: number
  /** Skip hydrogen atoms entirely — hydrogens are often missing in
   *  predicted structures and don't change the envelope much. */
  ignoreHydrogens: boolean
  /** Reference SASA value (Å²) used to normalize for the glass pass. */
  sasaReference: number
  /** Max triangles MarchingCubes will emit. */
  maxPolyCount: number
}

export const DEFAULT_GAUSSIAN_OPTIONS: GaussianSurfaceOptions = {
  resolution: 80,
  probeRadius: 1.4,
  sigmaFactor: 1.0,
  isolation: 0.5,
  padding: 6.0,
  ignoreHydrogens: true,
  sasaReference: 80.0,
  maxPolyCount: 250000,
}

export interface GaussianSurfaceResources {
  geometry: BufferGeometry
  vertexCount: number
  dispose: () => void
}

export function buildGaussianSurface(
  scene: MoleroScene,
  partial?: Partial<GaussianSurfaceOptions>,
): GaussianSurfaceResources {
  const opts = { ...DEFAULT_GAUSSIAN_OPTIONS, ...partial }
  const A = scene.attrs.count

  // ── Bbox + cubic grid setup ──────────────────────────────────────────
  // Symmetric cube — MarchingCubes' internal grid is cubic and we want
  // uniform voxel size in every axis so the SAS distance check is true
  // Euclidean. `half` = max-axis half-extent + padding.
  const { min, max } = scene.bbox
  const cx = (min[0] + max[0]) * 0.5
  const cy = (min[1] + max[1]) * 0.5
  const cz = (min[2] + max[2]) * 0.5
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  const half = span * 0.5 + opts.padding
  const size = opts.resolution
  // Voxel center positions: cell (i, j, k) → world (cx + (2i/size - 1)·half, …)
  // Use field[ix + size·iy + size²·iz] addressing (MarchingCubes convention).
  const cellWorld = (2 * half) / size

  // ── Allocate MarchingCubes ───────────────────────────────────────────
  const throwawayMaterial = new MeshBasicMaterial()
  const mc = new MarchingCubes(size, throwawayMaterial, false, false, opts.maxPolyCount)
  mc.isolation = opts.isolation
  mc.position.set(cx, cy, cz)
  mc.scale.set(half, half, half)

  // ── Field rasterization ──────────────────────────────────────────────
  // Gaussians sum from baseline zero. Default-empty cells stay at zero
  // and lie below iso (which is positive), so they read as "outside".
  const field = mc.field as Float32Array
  field.fill(0)

  const position = scene.attrs.position
  const atomicNumber = scene.attrs.atomicNumber
  const probe = opts.probeRadius
  const sigmaFactor = opts.sigmaFactor
  let rasterizedAtoms = 0

  // ~3σ contains 99.7% of a Gaussian's energy. Anything further
  // contributes < 0.011 — negligible against iso 0.5.
  const CULL_SIGMA = 3.0

  for (let i = 0; i < A; i++) {
    if (opts.ignoreHydrogens && isHydrogen(atomicNumber[i])) continue
    const vdw = VDW_RADII[atomicNumber[i]] ?? VDW_RADII[0]
    // Per-atom σ. At isolation=0.5, surface lands at d = σ·√(2·ln 2) ≈
    // 1.177·σ; with sigmaFactor=1.0 (default) σ = (vdW + probe), so
    // isolated-atom surface = 1.177·(vdW + probe). That's slightly fat
    // — most atoms have nearby neighbors that pull the surface in via
    // the additive Gaussian overlap. Drop sigmaFactor to ~0.85 for a
    // tighter isolated-atom surface; raise to ~1.2 for extra smoothing.
    const sigma = (vdw + probe) * sigmaFactor
    const reach = sigma * CULL_SIGMA
    const inv2Sigma2 = 1 / (2 * sigma * sigma)

    const ax = position[i * 3]
    const ay = position[i * 3 + 1]
    const az = position[i * 3 + 2]

    // worldX = cx + (2·ix/size − 1)·half → ix = ((worldX − cx)/half + 1)·size/2
    const ixMin = Math.max(0, Math.floor(((ax - reach - cx) / half + 1) * size * 0.5))
    const ixMax = Math.min(size - 1, Math.ceil (((ax + reach - cx) / half + 1) * size * 0.5))
    const iyMin = Math.max(0, Math.floor(((ay - reach - cy) / half + 1) * size * 0.5))
    const iyMax = Math.min(size - 1, Math.ceil (((ay + reach - cy) / half + 1) * size * 0.5))
    const izMin = Math.max(0, Math.floor(((az - reach - cz) / half + 1) * size * 0.5))
    const izMax = Math.min(size - 1, Math.ceil (((az + reach - cz) / half + 1) * size * 0.5))

    rasterizedAtoms++
    const reach2 = reach * reach
    for (let iz = izMin; iz <= izMax; iz++) {
      const vz = cz + (2 * iz / size - 1) * half
      const dz = vz - az
      const dz2 = dz * dz
      if (dz2 > reach2) continue
      const zoff = size * size * iz
      for (let iy = iyMin; iy <= iyMax; iy++) {
        const vy = cy + (2 * iy / size - 1) * half
        const dy = vy - ay
        const dy2dz2 = dy * dy + dz2
        if (dy2dz2 > reach2) continue
        const yoff = size * iy + zoff
        for (let ix = ixMin; ix <= ixMax; ix++) {
          const vx = cx + (2 * ix / size - 1) * half
          const dx = vx - ax
          const d2 = dx * dx + dy2dz2
          if (d2 > reach2) continue
          // Additive Gaussian — atoms reinforce in dense regions, giving
          // a smooth merged surface without per-atom bumps.
          field[ix + yoff] += Math.exp(-d2 * inv2Sigma2)
        }
      }
    }
  }
  void cellWorld // (was used in the SAS bbox check; kept var to silence
                 // the unused warning if we ever revive that path)

  // ── March ────────────────────────────────────────────────────────────
  mc.update()
  const mcAny = mc as any
  const vertexCount = mcAny.count as number
  if (vertexCount === 0) {
    throwawayMaterial.dispose()
    mc.reset()
    throw new Error(
      `Gaussian surface produced no geometry — rasterized ${rasterizedAtoms} atoms but no field cell crossed iso=${opts.isolation}. Try lowering isolation or raising sigmaFactor.`,
    )
  }

  // ── Snapshot positions + normals; bake mc.matrixWorld ────────────────
  const positionSlice = (mcAny.positionArray as Float32Array).slice(0, vertexCount * 3)
  const normalSlice = (mcAny.normalArray as Float32Array).slice(0, vertexCount * 3)
  mc.updateMatrixWorld(true)
  const M = mc.matrixWorld.elements
  for (let i = 0; i < vertexCount; i++) {
    const lx = positionSlice[i * 3]
    const ly = positionSlice[i * 3 + 1]
    const lz = positionSlice[i * 3 + 2]
    positionSlice[i * 3]     = M[0] * lx + M[4] * ly + M[8]  * lz + M[12]
    positionSlice[i * 3 + 1] = M[1] * lx + M[5] * ly + M[9]  * lz + M[13]
    positionSlice[i * 3 + 2] = M[2] * lx + M[6] * ly + M[10] * lz + M[14]
  }

  const surfaceGeom = new BufferGeometry()
  surfaceGeom.setAttribute('position', new Float32BufferAttribute(positionSlice, 3))
  surfaceGeom.setAttribute('normal', new Float32BufferAttribute(normalSlice, 3))

  // ── Per-vertex chemistry attribute ────────────────────────────────────
  const sasa = computeSASA(scene.attrs, scene.bbox)
  const hydrophobicity = computeHydrophobicity(scene)
  const glassAttr = buildPerVertexAttribute(
    scene,
    surfaceGeom,
    vertexCount,
    sasa,
    hydrophobicity,
    opts.sasaReference,
  )
  surfaceGeom.setAttribute('aGlass', new Float32BufferAttribute(glassAttr, 4))
  surfaceGeom.computeBoundingSphere()

  return {
    geometry: surfaceGeom,
    vertexCount,
    dispose: () => {
      surfaceGeom.dispose()
      throwawayMaterial.dispose()
      mc.reset()
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-vertex chemistry — Gaussian-weighted blend over nearby atoms.
//
// The old nearest-atom assignment produced visible voronoi cells on the
// surface where each vertex picked exactly one atom's properties. The
// transitions read as patches of color (one atom per cell, sharp seams).
// Smooth blending fixes this: each vertex sums contributions from every
// atom within ~3σ, weighted by exp(−d²/2σ²). σ is per-atom so larger
// atoms (Fe, Zn) contribute over a wider radius than C/N/O.
//
// `aGlass.w` now carries a continuous *aromaticness fraction* (0..1)
// rather than raw flag bits — fraction of nearby weight that comes from
// aromatic atoms. The glass material reads it as a scalar metalness
// boost. Same channel size (vec4), smoother shader inputs.

const BLEND_GRID_CELL = 5.0 // Å — larger than vdW so 27-cell sweep covers ~3σ
const BLEND_SIGMA_FACTOR = 0.9 // multiplier on (vdW + 0.6) for per-atom σ

function buildPerVertexAttribute(
  scene: MoleroScene,
  geom: BufferGeometry,
  vertexCount: number,
  sasa: Float32Array,
  hydrophobicity: Float32Array,
  sasaRef: number,
): Float32Array {
  const A = scene.attrs.count
  const position = scene.attrs.position
  const atomicNumber = scene.attrs.atomicNumber
  const { bbox } = scene
  const ox = bbox.min[0] - BLEND_GRID_CELL
  const oy = bbox.min[1] - BLEND_GRID_CELL
  const oz = bbox.min[2] - BLEND_GRID_CELL
  const nx = Math.max(1, Math.ceil((bbox.max[0] - ox) / BLEND_GRID_CELL) + 1)
  const ny = Math.max(1, Math.ceil((bbox.max[1] - oy) / BLEND_GRID_CELL) + 1)
  const nz = Math.max(1, Math.ceil((bbox.max[2] - oz) / BLEND_GRID_CELL) + 1)

  const counts = new Uint32Array(nx * ny * nz)
  const cellOf = new Uint32Array(A)
  for (let i = 0; i < A; i++) {
    const cx = Math.min(nx - 1, Math.max(0, ((position[i * 3]     - ox) / BLEND_GRID_CELL) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((position[i * 3 + 1] - oy) / BLEND_GRID_CELL) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((position[i * 3 + 2] - oz) / BLEND_GRID_CELL) | 0))
    const ci = cx + nx * (cy + ny * cz)
    cellOf[i] = ci
    counts[ci]++
  }
  const cells: (Uint32Array | null)[] = new Array(nx * ny * nz).fill(null)
  const cursors = new Uint32Array(nx * ny * nz)
  for (let ci = 0; ci < counts.length; ci++) {
    if (counts[ci] > 0) cells[ci] = new Uint32Array(counts[ci])
  }
  for (let i = 0; i < A; i++) {
    const ci = cellOf[i]
    cells[ci]![cursors[ci]++] = i
  }

  // Precompute per-atom σ² and 1/(2σ²) so the per-vertex loop is just
  // distance squared + multiply + exp.
  const sigma2 = new Float32Array(A)
  const inv2Sigma2 = new Float32Array(A)
  for (let i = 0; i < A; i++) {
    const vdw = VDW_RADII[atomicNumber[i]] ?? VDW_RADII[0]
    const s = (vdw + 0.6) * BLEND_SIGMA_FACTOR
    sigma2[i] = s * s
    inv2Sigma2[i] = 1 / (2 * s * s)
  }

  const positions = geom.getAttribute('position').array as Float32Array
  const out = new Float32Array(vertexCount * 4)
  const charges = scene.attrs.formalCharge
  const flags = scene.attrs.flags

  // Cap search distance per atom at 3σ — anything past that contributes
  // < 0.011 of the peak and is irrelevant.
  const SIGMA_CUTOFF = 3.0
  const sigmaCutoff2 = new Float32Array(A)
  for (let i = 0; i < A; i++) {
    sigmaCutoff2[i] = sigma2[i] * SIGMA_CUTOFF * SIGMA_CUTOFF
  }

  for (let v = 0; v < vertexCount; v++) {
    const vx = positions[v * 3]
    const vy = positions[v * 3 + 1]
    const vz = positions[v * 3 + 2]
    const cx = Math.min(nx - 1, Math.max(0, ((vx - ox) / BLEND_GRID_CELL) | 0))
    const cy = Math.min(ny - 1, Math.max(0, ((vy - oy) / BLEND_GRID_CELL) | 0))
    const cz = Math.min(nz - 1, Math.max(0, ((vz - oz) / BLEND_GRID_CELL) | 0))

    let wSum = 0
    let sasaAcc = 0
    let hydroAcc = 0
    let chargeAcc = 0
    let aromaticAcc = 0

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
            const ddx = position[j * 3]     - vx
            const ddy = position[j * 3 + 1] - vy
            const ddz = position[j * 3 + 2] - vz
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz
            if (d2 > sigmaCutoff2[j]) continue
            const w = Math.exp(-d2 * inv2Sigma2[j])
            wSum += w
            sasaAcc += w * sasa[j]
            hydroAcc += w * hydrophobicity[j]
            chargeAcc += w * charges[j]
            if (flags[j] & AtomFlag.AromaticRing) aromaticAcc += w
            // TransitionMetal flag mapped into aromaticness too — same
            // visual channel (metalness boost), shader doesn't need to
            // distinguish for the glass pass.
            if (flags[j] & AtomFlag.TransitionMetal) aromaticAcc += w * 2.5
          }
        }
      }
    }

    if (wSum < 1e-8) {
      out[v * 4]     = 0
      out[v * 4 + 1] = 0.5
      out[v * 4 + 2] = 0.5
      out[v * 4 + 3] = 0
      continue
    }
    const invW = 1 / wSum
    const s = Math.min(1, (sasaAcc * invW) / sasaRef)
    const h = hydrophobicityNorm(hydroAcc * invW)
    const c = 0.5 + Math.max(-1, Math.min(1, chargeAcc * invW)) * 0.5
    // aromaticness already weighted-summed — divide by total weight
    // gives the in-neighborhood fraction. Clamp to [0, 1] (metal boost
    // can push the unclamped value above 1).
    const ar = Math.min(1, aromaticAcc * invW)
    out[v * 4]     = s
    out[v * 4 + 1] = h
    out[v * 4 + 2] = c
    out[v * 4 + 3] = ar
  }
  return out
}
