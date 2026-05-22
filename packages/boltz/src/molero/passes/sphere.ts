/**
 * Sphere pass — instanced atomic spheres with property-channel materials.
 *
 * Geometry: one icosahedron per atom via `InstancedMesh`. Per-instance:
 *   - matrix (position × vdW-radius scale) → instanceMatrix
 *   - color (CPK in linear-RGB)            → instanceColor
 *   - **channels** packed vec4              → custom `aChannels` attr:
 *       .x = formalCharge   (-2 .. +2)
 *       .y = hybridization  (0 .. 4 — see Hybridization enum)
 *       .z = pLDDT          (0 .. 100, encoded in B-factor by our writer)
 *       .w = flagBits       (Uint8 bitfield, see AtomFlag)
 *
 * Material: `MeshPhysicalNodeMaterial`. TSL wires the channels into the
 * PBR slots:
 *   - **emission strength**  ← low pLDDT (1 - smoothstep(0.3, 0.7, p))
 *   - **emission color**     ← formalCharge sign (negative = red,
 *                                                  positive = cool blue)
 *   - **roughness**          ← hybridization (sp² polished, sp³ matte)
 *   - **metalness boost**    ← AromaticRing flag → 0.4, TransitionMetal → 1.0
 *
 * The CPU side packs the vec4 in `buildScene`/`buildChannels`; the GPU
 * never touches the underlying enum arrays directly. Phase 2 channel
 * routing can be reconfigured by editing this file alone — the chemistry
 * layer doesn't change.
 */
import {
  IcosahedronGeometry,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  MeshPhysicalNodeMaterial,
  Color,
} from 'three/webgpu'
import {
  abs,
  bitAnd,
  clamp,
  float,
  instancedBufferAttribute,
  mix,
  nodeObject,
  saturate,
  select,
  smoothstep,
  uint,
  vec3,
} from 'three/tsl'
import { AtomFlag, type Scene as MoleroScene } from '../scene/scene'

export interface SpherePassOptions {
  /** Multiplier on vdW radii. 0.28 ≈ ball-and-stick, 1.0 = space-filling. */
  scale: number
  /** Icosahedron subdivision level. 2 = 320 tris/atom. */
  detail: number
  /** When true, install the TSL property-channel material. When false,
   *  use a plain MeshPhysicalNodeMaterial with the instance color and
   *  no other channel routing. */
  propertyChannels: boolean
  /** Optional per-atom filter — return true to render an instance for
   *  that atom. Used by the cartoon representation to hide backbone
   *  atoms (which the ribbon already shows). */
  atomFilter?: (atomIndex: number) => boolean
}

export const DEFAULT_SPHERE_OPTIONS: SpherePassOptions = {
  scale: 0.42,
  detail: 2,
  propertyChannels: true,
}

export interface SpherePassResources {
  mesh: InstancedMesh
  geometry: IcosahedronGeometry
  material: MeshPhysicalNodeMaterial
  dispose: () => void
}

export function createSpherePass(
  scene: MoleroScene,
  partial?: Partial<SpherePassOptions>,
): SpherePassResources {
  const opts = { ...DEFAULT_SPHERE_OPTIONS, ...partial }
  const geometry = new IcosahedronGeometry(1, opts.detail)

  const totalCount = scene.attrs.count
  const filter = opts.atomFilter
  // First pass — pick out the atoms that pass the filter so we can size
  // the InstancedMesh exactly. Without this an unfiltered tail of the
  // buffer renders at the identity matrix as a sphere stack at origin.
  const indices = filter ? new Int32Array(totalCount) : null
  let count = totalCount
  if (filter && indices) {
    count = 0
    for (let i = 0; i < totalCount; i++) {
      if (filter(i)) indices[count++] = i
    }
  }
  const material = buildMaterial(scene, geometry, opts, indices, count)
  const mesh = new InstancedMesh(geometry, material, count)
  mesh.frustumCulled = false

  // ─── Per-instance matrix + CPK color ──────────────────────────────────
  const tmpMatrix = new Matrix4()
  const tmpColor = new Color()
  const { position, radius, color } = scene.attrs
  const scale = opts.scale

  for (let outI = 0; outI < count; outI++) {
    const i = indices ? indices[outI] : outI
    const r = radius[i] * scale
    tmpMatrix.makeScale(r, r, r)
    tmpMatrix.setPosition(
      position[i * 3],
      position[i * 3 + 1],
      position[i * 3 + 2],
    )
    mesh.setMatrixAt(outI, tmpMatrix)
    tmpColor.setRGB(
      color[i * 3],
      color[i * 3 + 1],
      color[i * 3 + 2],
    )
    mesh.setColorAt(outI, tmpColor)
  }
  mesh.instanceMatrix.needsUpdate = true
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true

  return {
    mesh,
    geometry,
    material,
    dispose: () => {
      geometry.dispose()
      material.dispose()
      mesh.dispose()
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Material assembly — packs the channel vec4 attribute and binds TSL nodes.

function buildMaterial(
  scene: MoleroScene,
  geometry: IcosahedronGeometry,
  opts: SpherePassOptions,
  indices: Int32Array | null,
  instanceCount: number,
): MeshPhysicalNodeMaterial {
  const material = new MeshPhysicalNodeMaterial({
    metalness: 0.0,
    roughness: 0.45,
    clearcoat: 0.4,
    clearcoatRoughness: 0.25,
    envMapIntensity: 1.0,
  })

  if (!opts.propertyChannels) return material

  const { formalCharge, hybridization, bfactor, flags } = scene.attrs
  const packed = new Float32Array(instanceCount * 4)
  for (let outI = 0; outI < instanceCount; outI++) {
    const i = indices ? indices[outI] : outI
    packed[outI * 4]     = formalCharge[i]
    packed[outI * 4 + 1] = hybridization[i]
    packed[outI * 4 + 2] = bfactor[i]
    packed[outI * 4 + 3] = flags[i]
  }
  const channelAttr = new InstancedBufferAttribute(packed, 4)
  channelAttr.setUsage(0x88e4 /* STATIC_DRAW */)
  geometry.setAttribute('aChannels', channelAttr)

  // nodeObject() exposes the swizzle accessors that TSL nodes need to be
  // composable; the bare Node<unknown> returned by instancedBufferAttribute
  // doesn't satisfy the typed-swizzle interface at the TS level.
  const channels = nodeObject(instancedBufferAttribute(channelAttr) as any) as any
  const formalChargeN = channels.x
  const hybridizationN = channels.y
  const plddtN = channels.z.div(100)
  const flagBitsN = channels.w

  // ── Emission ────────────────────────────────────────────────────────────
  // Low pLDDT → soft warm glow (uncertainty pulse).
  // Charge sign → cool blue (positive) or warm red (negative) overlay.
  const lowConfidence = float(1).sub(smoothstep(0.3, 0.7, plddtN))
  const confidenceGlow = vec3(1.0, 0.25, 0.1).mul(lowConfidence).mul(0.35)

  // chargeMagnitude in [0,1.5] roughly; we cap at 1 for safety.
  const chargeMag = saturate(abs(formalChargeN))
  const positiveTint = vec3(0.25, 0.45, 1.0)
  const negativeTint = vec3(1.0, 0.30, 0.20)
  const chargeTint = select(formalChargeN.greaterThan(0), positiveTint, negativeTint)
  const chargeGlow = chargeTint.mul(chargeMag).mul(0.4)

  material.emissiveNode = confidenceGlow.add(chargeGlow)

  // ── Roughness ───────────────────────────────────────────────────────────
  // sp² (=2) → polished (0.15); sp³ (=3) → matte (0.55).
  // Map hybridization linearly with a clamp; sp/Lp default to mid-range.
  const hybNormalized = saturate(hybridizationN.sub(1).div(2))
  // hybNormalized: sp=0, sp²=0.5, sp³=1, lp=1.5→1 (clamped)
  const roughnessByHyb = mix(float(0.15), float(0.55), hybNormalized)
  material.roughnessNode = clamp(roughnessByHyb, float(0.05), float(0.95))

  // ── Metalness ───────────────────────────────────────────────────────────
  // AromaticRing flag (bit 2) → metalness 0.4 (aromatic atoms read polished).
  // TransitionMetal flag (bit 6) → metalness 1.0 (literal metal centers).
  const flagU = uint(flagBitsN)
  const aromaticBit = bitAnd(flagU, uint(AtomFlag.AromaticRing)).greaterThan(uint(0))
  const metalBit    = bitAnd(flagU, uint(AtomFlag.TransitionMetal)).greaterThan(uint(0))
  const metalnessByFlag = select(metalBit, float(1.0),
                                 select(aromaticBit, float(0.4), float(0.0)))
  material.metalnessNode = metalnessByFlag

  return material
}
