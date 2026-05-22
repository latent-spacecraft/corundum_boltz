/**
 * Stick pass — instanced bond cylinders.
 *
 * Each bond renders as one cylinder from atom A to atom B. Per-instance
 * matrix encodes (midpoint position) × (rotation aligning Z-axis to bond
 * direction) × (scale = bondRadius, bondRadius, bondLength). Per-instance
 * color is the midpoint of the two atoms' CPK colors.
 *
 * Two-tone "split" coloring (each half colored by its atom) is a Phase-2
 * material upgrade: NodeMaterial with a varying that interpolates the
 * two endpoint colors based on local cylinder Z. For Phase 1.2 we ship
 * one color per bond and let the sphere pass carry the per-atom contrast.
 *
 * The cylinder geometry is built once (CylinderGeometry along Y-axis
 * by Three.js convention; we rotate the up vector to (0,1,0) when
 * computing instance matrices).
 *
 * Metallic-coordination bonds are visually distinguishable by their
 * thinner radius; this is the cheap-and-readable equivalent of dashed
 * lines without needing per-instance dashes.
 */
import {
  CylinderGeometry,
  InstancedMesh,
  Matrix4,
  MeshPhysicalMaterial,
  Color,
  Quaternion,
  Vector3,
} from 'three/webgpu'
import type { Scene as MoleroScene } from '../scene/scene'
import { BondFlag, type BondData } from '../chemistry/bonds'

export interface StickPassOptions {
  /** Cylinder radius for covalent bonds (Å). */
  radius: number
  /** Radius multiplier for metallic-coordination bonds. */
  metallicRadiusFactor: number
  /** Cylinder radial subdivisions. 8 = octagonal (cheap, octagonal
   *  silhouette is invisible at typical screen sizes). */
  radialSegments: number
  /** Whether to draw bonds beyond a max length (catch atom-clash spurious
   *  edges). Defaults to filtering out anything longer than maxLength. */
  maxLength: number
  material: {
    metalness: number
    roughness: number
    clearcoat: number
    clearcoatRoughness: number
    envMapIntensity: number
  }
}

export const DEFAULT_STICK_OPTIONS: StickPassOptions = {
  radius: 0.18,
  metallicRadiusFactor: 0.55,
  radialSegments: 10,
  maxLength: 3.2,
  material: {
    metalness: 0.0,
    roughness: 0.45,
    clearcoat: 0.4,
    clearcoatRoughness: 0.25,
    envMapIntensity: 1.0,
  },
}

export interface StickPassResources {
  mesh: InstancedMesh
  geometry: CylinderGeometry
  material: MeshPhysicalMaterial
  dispose: () => void
}

export function createStickPass(
  scene: MoleroScene,
  bonds: BondData,
  partial?: Partial<StickPassOptions>,
): StickPassResources {
  const opts = mergeOpts(DEFAULT_STICK_OPTIONS, partial)
  // CylinderGeometry's default axis is +Y; height = 1 so per-instance Y
  // scale is the bond length directly.
  const geometry = new CylinderGeometry(1, 1, 1, opts.radialSegments, 1, false)
  const material = new MeshPhysicalMaterial({
    metalness: opts.material.metalness,
    roughness: opts.material.roughness,
    clearcoat: opts.material.clearcoat,
    clearcoatRoughness: opts.material.clearcoatRoughness,
    envMapIntensity: opts.material.envMapIntensity,
  })

  // First pass — count bonds that pass the maxLength filter so we can
  // size the InstancedMesh exactly.
  const { atomA, atomB, flags } = bonds
  const N = bonds.count
  const position = scene.attrs.position
  const color = scene.attrs.color
  const maxLen2 = opts.maxLength * opts.maxLength

  let kept = 0
  for (let i = 0; i < N; i++) {
    const a = atomA[i], b = atomB[i]
    const dx = position[b * 3]     - position[a * 3]
    const dy = position[b * 3 + 1] - position[a * 3 + 1]
    const dz = position[b * 3 + 2] - position[a * 3 + 2]
    if (dx * dx + dy * dy + dz * dz <= maxLen2) kept++
  }

  const mesh = new InstancedMesh(geometry, material, kept)
  mesh.frustumCulled = false

  const m = new Matrix4()
  const p = new Vector3()
  const dir = new Vector3()
  const yAxis = new Vector3(0, 1, 0)
  const q = new Quaternion()
  const s = new Vector3()
  const c = new Color()

  let out = 0
  for (let i = 0; i < N; i++) {
    const a = atomA[i], b = atomB[i]
    const ax = position[a * 3]
    const ay = position[a * 3 + 1]
    const az = position[a * 3 + 2]
    const bx = position[b * 3]
    const by = position[b * 3 + 1]
    const bz = position[b * 3 + 2]
    const dx = bx - ax
    const dy = by - ay
    const dz = bz - az
    const len2 = dx * dx + dy * dy + dz * dz
    if (len2 > maxLen2) continue
    const len = Math.sqrt(len2)

    p.set((ax + bx) * 0.5, (ay + by) * 0.5, (az + bz) * 0.5)
    dir.set(dx / len, dy / len, dz / len)
    q.setFromUnitVectors(yAxis, dir)
    const r = (flags[i] & BondFlag.Metallic)
      ? opts.radius * opts.metallicRadiusFactor
      : opts.radius
    s.set(r, len, r)
    m.compose(p, q, s)
    mesh.setMatrixAt(out, m)

    c.setRGB(
      0.5 * (color[a * 3]     + color[b * 3]),
      0.5 * (color[a * 3 + 1] + color[b * 3 + 1]),
      0.5 * (color[a * 3 + 2] + color[b * 3 + 2]),
    )
    mesh.setColorAt(out, c)
    out++
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

function mergeOpts(
  base: StickPassOptions,
  partial?: Partial<StickPassOptions>,
): StickPassOptions {
  if (!partial) return base
  return {
    radius: partial.radius ?? base.radius,
    metallicRadiusFactor: partial.metallicRadiusFactor ?? base.metallicRadiusFactor,
    radialSegments: partial.radialSegments ?? base.radialSegments,
    maxLength: partial.maxLength ?? base.maxLength,
    material: { ...base.material, ...(partial.material ?? {}) },
  }
}
