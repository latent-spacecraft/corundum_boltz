/**
 * Salt bridge pass — curved emissive arcs between cation/anion pairs.
 *
 * Each bridge is a quadratic Bezier curve bowed outward from the
 * structure centroid (so arcs flare *away* from the protein body
 * rather than cutting straight through it). The curve is extruded via
 * `TubeGeometry` and merged into a single mesh per pass.
 *
 * Material is opaque `MeshBasicMaterial` with vertex-color gradient
 * (cation = blue at +Y end, anion = red at −Y end). Opaque-and-emissive
 * rather than additive: this lets the arcs render in Three.js's main
 * opaque pass, which the gem material's `transmission` backend samples
 * into its refraction backbuffer. Result: in `all` mode the arcs sit
 * *inside* the glass and refract correctly, reading as glowing filaments
 * embedded in the gem rather than flat overlay on top of it. In cartoon
 * mode they read as saturated colored arcs against the ribbon.
 *
 * Trade-off: no additive bloom-halo without a postprocess bloom pass.
 * For pure beauty + glow halo we'd want a postprocess bloom; for now
 * the radius + saturation reads cleanly across both representations.
 *
 * Per-bridge intensity scales with `|chargeA × chargeB| / distance²`
 * (Coulomb-like proxy with `^strengthExp` softening), so tight Arg-Asp
 * pairs blaze while looser Lys…Glu interactions pulse softly.
 */
import {
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
import type { SaltBridge } from '../chemistry/salt-bridges'

export interface SaltBridgePassOptions {
  /** Cylinder radius (Å). Single layer — opaque emissive doesn't
   *  benefit from a wider halo (which would just z-fight the core). */
  radius: number
  /** Radial segments per tube. */
  radialSegments: number
  /** Bezier tubular segments — controls arc smoothness. */
  tubularSegments: number
  /** Arc bow magnitude as a fraction of bridge length. 0 = straight
   *  cylinder, 0.2 = comfortable arc, 0.4 = dramatic bow. */
  bowFraction: number
  /** Cation-end color (hex). */
  positiveColor: number
  /** Anion-end color (hex). */
  negativeColor: number
  /** Intensity multiplier baked into vertex colors. Above 1 clamps to
   *  saturated channel max in the MeshBasic shader (no HDR bloom), but
   *  bumping it still pulls greys toward the saturated hue earlier. */
  intensity: number
  /** Strength scale exponent. The Coulomb-like proxy
   *  `|chargeA × chargeB| / distance²` is raised to this power. 0.5 is
   *  a soft square-root falloff (most bridges visible); 1.0 is linear
   *  (strong bridges blaze, weak ones recede). */
  strengthExp: number
}

export const DEFAULT_SALT_BRIDGE_PASS_OPTIONS: SaltBridgePassOptions = {
  radius: 0.20,
  radialSegments: 10,
  tubularSegments: 24,
  bowFraction: 0.18,
  positiveColor: 0x5598ff,
  negativeColor: 0xff4d4d,
  intensity: 1.8,
  strengthExp: 0.5,
}

export interface SaltBridgePassResources {
  group: Group
  geometries: BufferGeometry[]
  materials: MeshBasicMaterial[]
  dispose: () => void
}

export function createSaltBridgePass(
  scene: MoleroScene,
  bridges: SaltBridge[],
  partial?: Partial<SaltBridgePassOptions>,
): SaltBridgePassResources {
  const opts = { ...DEFAULT_SALT_BRIDGE_PASS_OPTIONS, ...partial }
  const position = scene.attrs.position
  const center = new Vector3(...scene.center)

  // Scratch vectors — avoid per-bridge allocation churn.
  const a = new Vector3()
  const b = new Vector3()
  const mid = new Vector3()
  const axis = new Vector3()
  const outward = new Vector3()
  const perp = new Vector3()
  const control = new Vector3()
  const tmpFallback = new Vector3()

  const posLin = hexToLinear(opts.positiveColor)
  const negLin = hexToLinear(opts.negativeColor)

  // Single per-bridge tube — opaque emissive, gets sampled by the gem
  // material's transmission backbuffer so the arcs refract correctly
  // when rendered inside the shell.
  const tubeGeoms: BufferGeometry[] = []

  for (const br of bridges) {
    a.set(
      position[br.atomA * 3],
      position[br.atomA * 3 + 1],
      position[br.atomA * 3 + 2],
    )
    b.set(
      position[br.atomB * 3],
      position[br.atomB * 3 + 1],
      position[br.atomB * 3 + 2],
    )
    mid.addVectors(a, b).multiplyScalar(0.5)

    // Arc bow: control point = midpoint + perpendicular offset.
    // Direction = midpoint-from-centroid, projected perpendicular to the
    // bridge axis. Result: arcs flare outward from the protein body.
    axis.subVectors(b, a)
    const len = axis.length()
    if (len < 1e-6) continue
    axis.divideScalar(len)
    outward.subVectors(mid, center)
    const outLen = outward.length()
    if (outLen > 1e-6) {
      outward.divideScalar(outLen)
    } else {
      // Bridge midpoint sits exactly on the centroid — pick any direction.
      outward.set(0, 1, 0)
    }
    // perp = outward - (outward·axis) axis  (Gram-Schmidt)
    perp.copy(outward).addScaledVector(axis, -outward.dot(axis))
    if (perp.lengthSq() < 1e-8) {
      // Outward was parallel to axis — fall back to a world-up-derived perp.
      tmpFallback.set(0, 1, 0)
      if (Math.abs(axis.y) > 0.95) tmpFallback.set(1, 0, 0)
      perp.copy(tmpFallback).addScaledVector(axis, -tmpFallback.dot(axis))
    }
    perp.normalize()
    control.copy(mid).addScaledVector(perp, len * opts.bowFraction)

    const curve = new QuadraticBezierCurve3(a.clone(), control.clone(), b.clone())

    // Strength proxy. Cap at 1.0 so a single Arg-Asp pair at 2.5 Å
    // doesn't drown the rest of the network.
    const strength = Math.min(
      1.0,
      Math.pow(
        Math.abs(br.chargeA * br.chargeB) / (br.distance * br.distance),
        opts.strengthExp,
      ),
    )

    tubeGeoms.push(
      buildArcTube(
        curve,
        opts.tubularSegments,
        opts.radius,
        opts.radialSegments,
        posLin,
        negLin,
        opts.intensity * strength,
      ),
    )
  }

  const group = new Group()
  const geometries: BufferGeometry[] = []
  const materials: MeshBasicMaterial[] = []

  // Opaque + standard depth so the transmission pass picks the arcs up
  // into its backbuffer. `renderOrder` left at default 0 to stay in
  // the opaque queue.
  if (tubeGeoms.length > 0) {
    const merged = mergeGeometries(tubeGeoms, false) as BufferGeometry
    for (const g of tubeGeoms) g.dispose()
    const material = new MeshBasicMaterial({
      vertexColors: true,
      transparent: false,
      depthWrite: true,
      depthTest: true,
    })
    const mesh = new Mesh(merged, material)
    mesh.frustumCulled = false
    group.add(mesh)
    geometries.push(merged)
    materials.push(material)
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

// ─────────────────────────────────────────────────────────────────────────────
// Per-bridge tube builder. TubeGeometry orders vertices as
// `(tubularSegments+1) × (radialSegments+1)` rings — the ring index along
// the curve gives us the `t` parameter we use for the cation-blue → anion-red
// color gradient.

function buildArcTube(
  curve: QuadraticBezierCurve3,
  tubularSegments: number,
  radius: number,
  radialSegments: number,
  posLin: [number, number, number],
  negLin: [number, number, number],
  intensity: number,
): BufferGeometry {
  const geom = new TubeGeometry(curve, tubularSegments, radius, radialSegments, false)
  const ringStride = radialSegments + 1
  const vertexCount = geom.getAttribute('position').count
  const colors = new Float32Array(vertexCount * 3)
  for (let v = 0; v < vertexCount; v++) {
    const iRing = Math.floor(v / ringStride)
    const t = iRing / tubularSegments // 0 at cation end, 1 at anion end
    const r = (posLin[0] * (1 - t) + negLin[0] * t) * intensity
    const g = (posLin[1] * (1 - t) + negLin[1] * t) * intensity
    const bC = (posLin[2] * (1 - t) + negLin[2] * t) * intensity
    colors[v * 3]     = r
    colors[v * 3 + 1] = g
    colors[v * 3 + 2] = bC
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
