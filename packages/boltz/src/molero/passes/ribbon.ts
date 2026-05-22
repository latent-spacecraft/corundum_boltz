/**
 * Ribbon pass — SS-aware putty tube along the polymer backbone.
 *
 * For each `BackboneSegment` from `extractBackbones`:
 *   1. Compute per-residue secondary structure (helix / sheet / coil) via
 *      `detectSecondaryStructure`. Nucleic-acid chains skip SS and use
 *      the coil profile uniformly.
 *   2. Compute per-residue radius:
 *        base = { helix: 0.55, sheet: 0.65, coil: 0.30 } [Å]
 *        radius = base × (0.40 + 0.60 × pLDDT/100)
 *      so low-confidence residues taper while SS atoms stay readable.
 *   3. Compute per-residue color: chain hue blended 55 % toward an SS
 *      tint (helix warm orange, sheet cool blue, coil keeps chain hue).
 *   4. Resample those per-residue values onto the high-resolution tube
 *      sample grid (linear interpolation for both radius and color).
 *   5. Extrude via `buildVariableRadiusTube` (Frenet-frame, variable
 *      radius, per-vertex color).
 *
 * One mesh per backbone segment so per-chain material edits stay
 * independent and chain-break gaps are real (no spurious tube spanning
 * a missing-residue void).
 *
 * Slice-1.5 plate: flat-ribbon profile for helix / arrow profile for
 * sheet (right now everything is a circular cross-section, just with
 * SS-modulated thickness).
 */
import {
  CatmullRomCurve3,
  Color,
  Group,
  Mesh,
  MeshPhysicalNodeMaterial,
  Vector3,
  type BufferGeometry,
} from 'three/webgpu'
import {
  extractBackbones,
  type BackboneOptions,
  type BackboneSegment,
} from '../chemistry/backbone'
import {
  detectSecondaryStructure,
  SecondaryStructure,
} from '../chemistry/secondary-structure'
import { CHAIN_PALETTE_HEX } from '../chemistry/chain-palette'
import type { Scene as MoleroScene } from '../scene/scene'
import { buildVariableRadiusTube } from './putty-tube'

/**
 * Cross-section per SS — (u = binormal axis, v = normal axis).
 *   Coil:  equal radii, reads as a round tube.
 *   Helix: U > V — flat ribbon in the binormal direction.
 *   Sheet: U ≫ V — wider, thinner flat ribbon (perpendicular ovoid).
 *
 * Both U and V are scaled by pLDDT modulation; low-confidence regions
 * taper symmetrically so a flat ribbon still reads as a flat ribbon
 * (just thinner overall) rather than degenerating into a noodle.
 */
export interface SSRadii {
  u: number
  v: number
}

export interface RibbonPassOptions {
  /** Per-SS cross-section (Å). u = binormal-axis radius, v = normal-axis
   *  radius. Equal → round; u > v → flat ribbon. `nucleic` is used for
   *  RNA/DNA backbones; the rest are for proteins. */
  radii: { helix: SSRadii; sheet: SSRadii; coil: SSRadii; nucleic: SSRadii }
  /** pLDDT modulation: radius = base × (floor + (1-floor) × pLDDT/100). */
  pLDDTFloor: number
  /** Tube radial segments — 16 = hexadecagonal, gives clean ovoid silhouette. */
  radialSegments: number
  /** Spline samples per trace residue. Higher = smoother bends + SS edges. */
  tubularSegmentsPerResidue: number
  /** Backbone-extraction tuning. */
  backbone: Partial<BackboneOptions>
  /** SS-tint blend amount — 0 = pure chain hue, 1 = pure SS tint. */
  ssTintStrength: number
  /** Material parameters. */
  material: {
    metalness: number
    roughness: number
    clearcoat: number
    clearcoatRoughness: number
    envMapIntensity: number
  }
}

export const DEFAULT_RIBBON_OPTIONS: RibbonPassOptions = {
  radii: {
    helix:   { u: 0.95, v: 0.30 }, // flat ribbon
    sheet:   { u: 1.20, v: 0.22 }, // wider perpendicular ovoid
    coil:    { u: 0.32, v: 0.32 }, // round tube
    nucleic: { u: 0.60, v: 0.60 }, // round tube, ~2× protein coil
  },
  pLDDTFloor: 0.40,
  radialSegments: 16,
  tubularSegmentsPerResidue: 8,
  backbone: {},
  ssTintStrength: 0.55,
  material: {
    metalness: 0.05,
    roughness: 0.35,
    clearcoat: 0.6,
    clearcoatRoughness: 0.15,
    envMapIntensity: 1.0,
  },
}

// Chain palette pulled from the shared module so the glass surface
// (gaussian-surface.ts → aChainTint) shows the same hues.
const CHAIN_PALETTE = CHAIN_PALETTE_HEX

// SS tints — warm helix, cool sheet, neutral coil.
const SS_TINTS: Record<number, [number, number, number]> = {
  [SecondaryStructure.Helix]: srgbToLinearTriple(1.00, 0.45, 0.20),
  [SecondaryStructure.Sheet]: srgbToLinearTriple(0.20, 0.55, 1.00),
  [SecondaryStructure.Coil]:  srgbToLinearTriple(0.50, 0.50, 0.50),
}

export interface RibbonPassResources {
  group: Group
  meshes: Mesh[]
  geometries: BufferGeometry[]
  materials: MeshPhysicalNodeMaterial[]
  dispose: () => void
}

export function createRibbonPass(
  scene: MoleroScene,
  partial?: Partial<RibbonPassOptions>,
): RibbonPassResources {
  const opts = mergeOpts(DEFAULT_RIBBON_OPTIONS, partial)
  const segments = extractBackbones(scene, opts.backbone)
  const ss = detectSecondaryStructure(scene)
  const bfactor = scene.attrs.bfactor

  const group = new Group()
  const meshes: Mesh[] = []
  const geometries: BufferGeometry[] = []
  const materials: MeshPhysicalNodeMaterial[] = []

  for (const seg of segments) {
    const built = buildSegmentGeometry(seg, ss, bfactor, opts)
    if (!built) continue
    // NodeMaterial picks up the `color` BufferAttribute via vertexColors,
    // multiplying it against `material.color` (which we leave white so
    // the per-vertex SS+chain blend renders directly).
    const material = new MeshPhysicalNodeMaterial({
      color: new Color(0xffffff),
      vertexColors: true,
      metalness: opts.material.metalness,
      roughness: opts.material.roughness,
      clearcoat: opts.material.clearcoat,
      clearcoatRoughness: opts.material.clearcoatRoughness,
      envMapIntensity: opts.material.envMapIntensity,
    })
    const mesh = new Mesh(built, material)
    mesh.frustumCulled = false
    group.add(mesh)
    meshes.push(mesh)
    geometries.push(built)
    materials.push(material)
  }

  return {
    group,
    meshes,
    geometries,
    materials,
    dispose: () => {
      for (const g of geometries) g.dispose()
      for (const m of materials) m.dispose()
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-segment geometry assembly.

function buildSegmentGeometry(
  seg: BackboneSegment,
  ss: Uint8Array,
  bfactor: Float32Array,
  opts: RibbonPassOptions,
): BufferGeometry | null {
  const N = seg.atomIndex.length
  if (N < 2) return null

  // Curve control points.
  const points: Vector3[] = new Array(N)
  for (let i = 0; i < N; i++) {
    points[i] = new Vector3(
      seg.positions[i * 3],
      seg.positions[i * 3 + 1],
      seg.positions[i * 3 + 2],
    )
  }
  const curve = new CatmullRomCurve3(points, false, 'catmullrom', 0.5)
  const tubularSegments = Math.max(8, (N - 1) * opts.tubularSegmentsPerResidue)
  const ringCount = tubularSegments + 1

  // Per-residue base radii (U + V) and color.
  const isProtein = seg.entityType === 'protein'
  const baseRadiusU = new Float32Array(N)
  const baseRadiusV = new Float32Array(N)
  const baseColor = new Float32Array(N * 3)
  const chainHex = CHAIN_PALETTE[seg.chainIndex % CHAIN_PALETTE.length]
  const chainRGB = hexToLinearRGB(chainHex)

  for (let i = 0; i < N; i++) {
    const ssCode = isProtein ? ss[seg.residueIndex[i]] : SecondaryStructure.Coil
    const base = isProtein
      ? (ssCode === SecondaryStructure.Helix ? opts.radii.helix
        : ssCode === SecondaryStructure.Sheet ? opts.radii.sheet
        : opts.radii.coil)
      : opts.radii.nucleic
    const plddt = clamp01(bfactor[seg.atomIndex[i]] / 100)
    const mod = opts.pLDDTFloor + (1 - opts.pLDDTFloor) * plddt
    baseRadiusU[i] = base.u * mod
    baseRadiusV[i] = base.v * mod

    if (ssCode === SecondaryStructure.Coil || !isProtein) {
      baseColor[i * 3]     = chainRGB[0]
      baseColor[i * 3 + 1] = chainRGB[1]
      baseColor[i * 3 + 2] = chainRGB[2]
    } else {
      const tint = SS_TINTS[ssCode] ?? SS_TINTS[SecondaryStructure.Coil]
      const a = opts.ssTintStrength
      baseColor[i * 3]     = chainRGB[0] * (1 - a) + tint[0] * a
      baseColor[i * 3 + 1] = chainRGB[1] * (1 - a) + tint[1] * a
      baseColor[i * 3 + 2] = chainRGB[2] * (1 - a) + tint[2] * a
    }
  }

  // Resample onto tube ring samples.
  const radiiU = new Float32Array(ringCount)
  const radiiV = new Float32Array(ringCount)
  const colors = new Float32Array(ringCount * 3)
  for (let r = 0; r < ringCount; r++) {
    const residueFloat = (r / tubularSegments) * (N - 1)
    const a = Math.floor(residueFloat)
    const b = Math.min(N - 1, a + 1)
    const t = residueFloat - a
    radiiU[r] = baseRadiusU[a] * (1 - t) + baseRadiusU[b] * t
    radiiV[r] = baseRadiusV[a] * (1 - t) + baseRadiusV[b] * t
    colors[r * 3]     = baseColor[a * 3]     * (1 - t) + baseColor[b * 3]     * t
    colors[r * 3 + 1] = baseColor[a * 3 + 1] * (1 - t) + baseColor[b * 3 + 1] * t
    colors[r * 3 + 2] = baseColor[a * 3 + 2] * (1 - t) + baseColor[b * 3 + 2] * t
  }

  return buildVariableRadiusTube({
    curve,
    tubularSegments,
    radialSegments: opts.radialSegments,
    radiiU,
    radiiV,
    colors,
    closed: false,
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers.

function hexToLinearRGB(hex: number): [number, number, number] {
  const r = ((hex >> 16) & 0xff) / 255
  const g = ((hex >> 8) & 0xff) / 255
  const b = (hex & 0xff) / 255
  return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)]
}
function srgbToLinearTriple(r: number, g: number, b: number): [number, number, number] {
  return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)]
}
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}
function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

function mergeOpts(
  base: RibbonPassOptions,
  partial?: Partial<RibbonPassOptions>,
): RibbonPassOptions {
  if (!partial) return base
  return {
    radii: { ...base.radii, ...(partial.radii ?? {}) },
    pLDDTFloor: partial.pLDDTFloor ?? base.pLDDTFloor,
    radialSegments: partial.radialSegments ?? base.radialSegments,
    tubularSegmentsPerResidue: partial.tubularSegmentsPerResidue ?? base.tubularSegmentsPerResidue,
    backbone: { ...base.backbone, ...(partial.backbone ?? {}) },
    ssTintStrength: partial.ssTintStrength ?? base.ssTintStrength,
    material: { ...base.material, ...(partial.material ?? {}) },
  }
}
