/**
 * Hydrogen-bond glow pass — faint golden lines between donors and
 * acceptors.
 *
 * Same architecture as the salt-bridge pass (curved Bezier arcs, core +
 * halo layers, additive blending, depthTest off) but tuned for the H-bond
 * register: thinner radii, single warm-gold color (no donor/acceptor
 * gradient — the convention isn't as strong as for salt bridges), and a
 * gentle distance-driven intensity falloff (3.0 Å bonds blaze, 3.5 Å
 * bonds barely shimmer).
 *
 * Proteins typically have several hundred H-bonds. Each contributes
 * ~25 tube vertices × 2 layers × 8 radials → ~400 vertices per bond.
 * A 300-residue protein with ~600 H-bonds is ~250 k vertices across
 * core + halo. Still tiny.
 *
 * For DNA the result is striking: every Watson-Crick base pair has 2-3
 * H-bonds connecting the two strands across the helix — the golden
 * ladder of base pairing literally lights up between the colored base
 * polygons.
 */
import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  Group,
  Mesh,
  MeshBasicMaterial,
  QuadraticBezierCurve3,
  TubeGeometry,
  Vector3,
} from 'three/webgpu'
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'
import type { Scene as MoleroScene } from '../scene/scene'
import type { HBond } from '../chemistry/h-bonds'

export interface HBondPassOptions {
  /** Bright core radius (Å). */
  coreRadius: number
  /** Soft halo radius (Å). */
  haloRadius: number
  radialSegments: number
  tubularSegments: number
  /** Arc bow fraction; H-bonds are short so we bow less than salt bridges. */
  bowFraction: number
  /** Single golden color for the whole pass. */
  color: number
  coreIntensity: number
  haloIntensity: number
  /** Distance at which intensity hits 1× (closer = brighter). 3.0 Å
   *  matches typical N-H⋯O hydrogen-bond geometry. */
  referenceDistance: number
}

export const DEFAULT_HBOND_PASS_OPTIONS: HBondPassOptions = {
  coreRadius: 0.055,
  haloRadius: 0.16,
  radialSegments: 6,
  tubularSegments: 12,
  bowFraction: 0.08,
  color: 0xffd58a,   // warm gold
  coreIntensity: 1.2,
  haloIntensity: 0.35,
  referenceDistance: 3.0,
}

export interface HBondPassResources {
  group: Group
  geometries: BufferGeometry[]
  materials: MeshBasicMaterial[]
  dispose: () => void
}

export function createHBondPass(
  scene: MoleroScene,
  bonds: HBond[],
  partial?: Partial<HBondPassOptions>,
): HBondPassResources {
  const opts = { ...DEFAULT_HBOND_PASS_OPTIONS, ...partial }
  const position = scene.attrs.position
  const center = new Vector3(...scene.center)

  const a = new Vector3()
  const b = new Vector3()
  const mid = new Vector3()
  const axis = new Vector3()
  const outward = new Vector3()
  const perp = new Vector3()
  const control = new Vector3()
  const tmpFallback = new Vector3()
  const colLin = hexToLinear(opts.color)

  const coreGeoms: BufferGeometry[] = []
  const haloGeoms: BufferGeometry[] = []

  for (const hb of bonds) {
    a.set(
      position[hb.atomA * 3],
      position[hb.atomA * 3 + 1],
      position[hb.atomA * 3 + 2],
    )
    b.set(
      position[hb.atomB * 3],
      position[hb.atomB * 3 + 1],
      position[hb.atomB * 3 + 2],
    )
    mid.addVectors(a, b).multiplyScalar(0.5)
    axis.subVectors(b, a)
    const len = axis.length()
    if (len < 1e-6) continue
    axis.divideScalar(len)

    // Same bow direction logic as salt bridges — flare outward from
    // structure centroid for visibility.
    outward.subVectors(mid, center)
    const outLen = outward.length()
    if (outLen > 1e-6) outward.divideScalar(outLen)
    else outward.set(0, 1, 0)
    perp.copy(outward).addScaledVector(axis, -outward.dot(axis))
    if (perp.lengthSq() < 1e-8) {
      tmpFallback.set(0, 1, 0)
      if (Math.abs(axis.y) > 0.95) tmpFallback.set(1, 0, 0)
      perp.copy(tmpFallback).addScaledVector(axis, -tmpFallback.dot(axis))
    }
    perp.normalize()
    control.copy(mid).addScaledVector(perp, len * opts.bowFraction)

    const curve = new QuadraticBezierCurve3(a.clone(), control.clone(), b.clone())

    // Distance falloff: (referenceDistance / distance)². 3.0 Å reference
    // means a 2.8 Å bond is ~1.15×, a 3.5 Å bond is ~0.73×.
    const distFactor = Math.min(
      1.5,
      Math.pow(opts.referenceDistance / hb.distance, 2),
    )

    coreGeoms.push(
      buildArcTube(
        curve,
        opts.tubularSegments,
        opts.coreRadius,
        opts.radialSegments,
        colLin,
        opts.coreIntensity * distFactor,
      ),
    )
    haloGeoms.push(
      buildArcTube(
        curve,
        opts.tubularSegments,
        opts.haloRadius,
        opts.radialSegments,
        colLin,
        opts.haloIntensity * distFactor,
      ),
    )
  }

  const group = new Group()
  const geometries: BufferGeometry[] = []
  const materials: MeshBasicMaterial[] = []

  // depthTest:false matches salt-bridges — emissive lattice should
  // show through the gem shell in `all` mode.
  if (coreGeoms.length > 0) {
    const coreMerged = mergeGeometries(coreGeoms, false) as BufferGeometry
    for (const g of coreGeoms) g.dispose()
    const coreMaterial = new MeshBasicMaterial({
      vertexColors: true,
      blending: AdditiveBlending,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    })
    const coreMesh = new Mesh(coreMerged, coreMaterial)
    coreMesh.frustumCulled = false
    // Renders behind salt bridges (which use 20/21) so salt bridges
    // pop on top when both networks overlap a region.
    coreMesh.renderOrder = 16
    group.add(coreMesh)
    geometries.push(coreMerged)
    materials.push(coreMaterial)
  }
  if (haloGeoms.length > 0) {
    const haloMerged = mergeGeometries(haloGeoms, false) as BufferGeometry
    for (const g of haloGeoms) g.dispose()
    const haloMaterial = new MeshBasicMaterial({
      vertexColors: true,
      blending: AdditiveBlending,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    })
    const haloMesh = new Mesh(haloMerged, haloMaterial)
    haloMesh.frustumCulled = false
    haloMesh.renderOrder = 15
    group.add(haloMesh)
    geometries.push(haloMerged)
    materials.push(haloMaterial)
  }

  return {
    group,
    geometries,
    materials,
    dispose: () => {
      for (const g of geometries) g.dispose()
      for (const m of materials) m.dispose()
    },
  }
}

function buildArcTube(
  curve: QuadraticBezierCurve3,
  tubularSegments: number,
  radius: number,
  radialSegments: number,
  colLin: [number, number, number],
  intensity: number,
): BufferGeometry {
  const geom = new TubeGeometry(curve, tubularSegments, radius, radialSegments, false)
  const vertexCount = geom.getAttribute('position').count
  const colors = new Float32Array(vertexCount * 3)
  const r = colLin[0] * intensity
  const g = colLin[1] * intensity
  const b = colLin[2] * intensity
  for (let v = 0; v < vertexCount; v++) {
    colors[v * 3]     = r
    colors[v * 3 + 1] = g
    colors[v * 3 + 2] = b
  }
  geom.setAttribute('color', new BufferAttribute(colors, 3))
  return geom
}

function hexToLinear(hex: number): [number, number, number] {
  const r = ((hex >> 16) & 0xff) / 255
  const g = ((hex >> 8) & 0xff) / 255
  const b = (hex & 0xff) / 255
  return [srgbToLinear(r), srgbToLinear(g), srgbToLinear(b)]
}
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}
