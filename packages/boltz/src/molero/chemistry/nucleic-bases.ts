/**
 * Nucleic base geometry — perimeter-ordered ring atom positions for the
 * filled-polygon cartoon pass.
 *
 * For each RNA/DNA residue we walk the outer perimeter of the base ring
 * in CCW order:
 *   Pyrimidine (6-ring):     N1, C2, N3, C4, C5, C6
 *   Purine (fused 5+6):      N1, C2, N3, C4, N9, C8, N7, C5, C6
 * The purine path traces the outer edges of both rings, skipping the
 * internal C4-C5 shared edge — this gives the classic dumbbell silhouette.
 *
 * Plane normal is computed from a SVD-ish cross product of two well-
 * separated chord vectors from the centroid. Atoms missing from a
 * partial-resolution structure get skipped from the perimeter; bases
 * with fewer than 3 perimeter atoms left are discarded.
 *
 * The pass renders each base as a thin prism — top face fan-triangulated
 * from the centroid, bottom face mirrored, side walls per edge.
 */
import { internAtomName } from '../parsers/mmcif'
import type { Scene as MoleroScene } from '../scene/scene'

export interface NucleicBaseGeom {
  /** Index into scene.residues. */
  residueIndex: number
  chainIndex: number
  /** Sugar C1' position. */
  c1: [number, number, number]
  /** Ring perimeter, packed [x0, y0, z0, x1, y1, z1, ...] in CCW order.
   *  Length = 3 × number-of-perimeter-atoms-found (6 for pyrimidine,
   *  9 for full purine, fewer if atoms missing). */
  perimeter: Float32Array
  /** Centroid of the perimeter atoms. */
  centroid: [number, number, number]
  /** Unit normal to the ring plane. */
  normal: [number, number, number]
  isPurine: boolean
  /** One-letter base code, uppercase (A/G/C/T/U/I/N). */
  baseChar: string
}

const PURINE_RESIDUES = new Set(['A', 'G', 'I', 'DA', 'DG', 'DI'])
const PYRIMIDINE_RESIDUES = new Set(['C', 'T', 'U', 'DC', 'DT', 'DU', 'N', 'DN'])

// Perimeter walks in CCW order. The CCW orientation is enforced by the
// plane-normal sign check at the end of extraction — if a particular
// PDB structure happens to mirror our convention, we flip the normal.
const PYRIMIDINE_PERIMETER = ['N1', 'C2', 'N3', 'C4', 'C5', 'C6']
const PURINE_PERIMETER = ['N1', 'C2', 'N3', 'C4', 'N9', 'C8', 'N7', 'C5', 'C6']

const PYRIMIDINE_PERIMETER_IDS = PYRIMIDINE_PERIMETER.map(internAtomName)
const PURINE_PERIMETER_IDS = PURINE_PERIMETER.map(internAtomName)
const C1_PRIME_ID = internAtomName("C1'")

export function extractNucleicBases(scene: MoleroScene): NucleicBaseGeom[] {
  const out: NucleicBaseGeom[] = []
  const positions = scene.attrs.position
  const atomNameId = scene.attrs.atomNameId

  for (const chain of scene.chains) {
    if (chain.entityType !== 'rna' && chain.entityType !== 'dna') continue

    for (let r = chain.residueStart; r < chain.residueEnd; r++) {
      const res = scene.residues[r]
      const isPurine = PURINE_RESIDUES.has(res.compId)
      const isPyrimidine = PYRIMIDINE_RESIDUES.has(res.compId)
      if (!isPurine && !isPyrimidine) continue
      const perimeterIds = isPurine ? PURINE_PERIMETER_IDS : PYRIMIDINE_PERIMETER_IDS

      // Walk this residue's atoms once into a name-id → atom-index map,
      // then index by perimeter order. Lots faster than O(perim × atoms)
      // for the rare 9-atom purines.
      let c1Idx = -1
      const localMap = new Map<number, number>()
      for (let a = res.atomStart; a < res.atomEnd; a++) {
        const nameId = atomNameId[a]
        if (nameId === C1_PRIME_ID) c1Idx = a
        localMap.set(nameId, a)
      }
      if (c1Idx < 0) continue

      // Collect perimeter atoms in order, skipping any missing.
      const perimPos: number[] = []
      let cx = 0, cy = 0, cz = 0
      for (const want of perimeterIds) {
        const aIdx = localMap.get(want)
        if (aIdx === undefined) continue
        const x = positions[aIdx * 3]
        const y = positions[aIdx * 3 + 1]
        const z = positions[aIdx * 3 + 2]
        perimPos.push(x, y, z)
        cx += x; cy += y; cz += z
      }
      const perimCount = perimPos.length / 3
      if (perimCount < 3) continue
      const inv = 1 / perimCount
      cx *= inv; cy *= inv; cz *= inv

      // Plane normal — cross of two well-separated chord vectors from
      // the centroid. Indices 0 and ⌈N/3⌉ guarantee non-antipodal.
      const i0 = 0
      const i1 = Math.max(1, Math.floor(perimCount / 3))
      const ax = perimPos[i0 * 3]     - cx
      const ay = perimPos[i0 * 3 + 1] - cy
      const az = perimPos[i0 * 3 + 2] - cz
      const bx = perimPos[i1 * 3]     - cx
      const by = perimPos[i1 * 3 + 1] - cy
      const bz = perimPos[i1 * 3 + 2] - cz
      let nx = ay * bz - az * by
      let ny = az * bx - ax * bz
      let nz = ax * by - ay * bx
      const nl = Math.hypot(nx, ny, nz)
      if (nl < 1e-6) continue // degenerate
      nx /= nl; ny /= nl; nz /= nl

      // Orient the normal so the perimeter winds CCW when viewed from
      // the normal direction. Check the signed area via the shoelace
      // formula in the plane (using two basis vectors derived from the
      // normal). If negative, flip the normal so top-face fan
      // triangulation produces outward-facing triangles.
      if (signedPlanarArea(perimPos, perimCount, cx, cy, cz, nx, ny, nz) < 0) {
        nx = -nx; ny = -ny; nz = -nz
      }

      const compId = res.compId
      const baseChar = compId.length === 2 && compId[0] === 'D' ? compId[1] : compId[0]

      out.push({
        residueIndex: r,
        chainIndex: chain.index,
        c1: [positions[c1Idx * 3], positions[c1Idx * 3 + 1], positions[c1Idx * 3 + 2]],
        perimeter: Float32Array.from(perimPos),
        centroid: [cx, cy, cz],
        normal: [nx, ny, nz],
        isPurine,
        baseChar,
      })
    }
  }
  return out
}

/** Signed area of the perimeter projected onto the plane (normal n).
 *  Positive = CCW when viewed from +n; negative = CW. */
function signedPlanarArea(
  perim: number[],
  count: number,
  cx: number, cy: number, cz: number,
  nx: number, ny: number, nz: number,
): number {
  // Build an in-plane orthonormal basis (e1, e2) from the normal.
  let e1x, e1y, e1z
  if (Math.abs(nx) < 0.9) {
    // Cross with world X.
    e1x = 0
    e1y = nz
    e1z = -ny
  } else {
    e1x = -nz
    e1y = 0
    e1z = nx
  }
  const e1l = Math.hypot(e1x, e1y, e1z) || 1
  e1x /= e1l; e1y /= e1l; e1z /= e1l
  // e2 = n × e1
  const e2x = ny * e1z - nz * e1y
  const e2y = nz * e1x - nx * e1z
  const e2z = nx * e1y - ny * e1x
  // 2D shoelace.
  let area = 0
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count
    const dxA = perim[i * 3] - cx,     dyA = perim[i * 3 + 1] - cy,     dzA = perim[i * 3 + 2] - cz
    const dxB = perim[j * 3] - cx,     dyB = perim[j * 3 + 1] - cy,     dzB = perim[j * 3 + 2] - cz
    const uA = dxA * e1x + dyA * e1y + dzA * e1z
    const vA = dxA * e2x + dyA * e2y + dzA * e2z
    const uB = dxB * e1x + dyB * e1y + dzB * e1z
    const vB = dxB * e2x + dyB * e2y + dzB * e2z
    area += uA * vB - uB * vA
  }
  return area
}

// ─────────────────────────────────────────────────────────────────────────────
// Base color palette — five distinct hues.

const BASE_COLORS: Record<string, number> = {
  A: 0xff6b6b, // coral red
  G: 0xffd166, // honey gold
  C: 0x4ecdc4, // teal
  T: 0x8be78b, // mint green
  U: 0xc77dff, // amethyst
  I: 0xffa07a, // peach (inosine)
  N: 0xb0b0b0, // gray (unknown)
}

export function baseColorHex(baseChar: string): number {
  return BASE_COLORS[baseChar] ?? BASE_COLORS.N
}
