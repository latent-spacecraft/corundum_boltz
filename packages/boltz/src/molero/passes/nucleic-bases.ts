/**
 * Nucleic base ring pass — filled-polygon prisms from real ring atoms.
 *
 * For each nucleic base we build a thin prism using the perimeter atom
 * positions extracted by `extractNucleicBases`. The top + bottom faces
 * are fan-triangulated from the centroid (offset by ±thickness/2 along
 * the ring plane normal); side walls are quads, one per perimeter edge.
 * All vertex blocks are duplicated at face boundaries so flat shading
 * lands cleanly (no normal-averaging across the top/side seam).
 *
 * All bases for the whole scene are merged into one `BufferGeometry`
 * with `vertexColors` so the renderer is a single draw call. Pyrimidines
 * contribute 38 vertices + 24 tris; purines contribute 56 vertices + 36
 * tris. A 292-bp DNA gets ~14 k vertices and ~9 k triangles — trivial.
 *
 * Material is `MeshPhysicalNodeMaterial`, polished-but-matte like the
 * protein ribbon. Color comes from the per-base palette via
 * `vertexColors`.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Mesh,
  MeshPhysicalNodeMaterial,
} from 'three/webgpu'
import { baseColorHex, type NucleicBaseGeom } from '../chemistry/nucleic-bases'

export interface NucleicBasePassOptions {
  /** Prism thickness along the ring plane normal (Å). */
  thickness: number
  material: {
    metalness: number
    roughness: number
    clearcoat: number
    clearcoatRoughness: number
    envMapIntensity: number
  }
}

export const DEFAULT_NUCLEIC_BASE_OPTIONS: NucleicBasePassOptions = {
  thickness: 0.45,
  material: {
    metalness: 0.05,
    roughness: 0.35,
    clearcoat: 0.5,
    clearcoatRoughness: 0.15,
    envMapIntensity: 1.0,
  },
}

export interface NucleicBasePassResources {
  mesh: Mesh
  geometry: BufferGeometry
  material: MeshPhysicalNodeMaterial
  dispose: () => void
}

// Helpers — sRGB → linear for vertex color storage.
function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

export function createNucleicBasePass(
  bases: NucleicBaseGeom[],
  partial?: Partial<NucleicBasePassOptions>,
): NucleicBasePassResources {
  const opts = { ...DEFAULT_NUCLEIC_BASE_OPTIONS, ...partial }

  // ── First pass — count vertices + triangles ──────────────────────────
  let vCount = 0
  let tCount = 0
  for (const b of bases) {
    const N = b.perimeter.length / 3
    if (N < 3) continue
    // Top: N+1 vertices (perimeter + centroid), N triangles.
    // Bottom: N+1 vertices, N triangles.
    // Side walls: 4N vertices (per-edge quad), 2N triangles.
    vCount += 6 * N + 2
    tCount += 4 * N
  }

  const positions = new Float32Array(vCount * 3)
  const normals = new Float32Array(vCount * 3)
  const colors = new Float32Array(vCount * 3)
  const indices = vCount > 65535 ? new Uint32Array(tCount * 3) : new Uint16Array(tCount * 3)

  let vOff = 0 // vertex cursor (in vertex units, multiply by 3 for float index)
  let iOff = 0 // triangle index cursor

  const tmpColor = new Color()
  const tinyTip = opts.thickness * 0.5

  for (const b of bases) {
    const N = b.perimeter.length / 3
    if (N < 3) continue

    const cx = b.centroid[0], cy = b.centroid[1], cz = b.centroid[2]
    const nx = b.normal[0], ny = b.normal[1], nz = b.normal[2]
    const ox = tinyTip * nx, oy = tinyTip * ny, oz = tinyTip * nz

    // Per-base color (linear-RGB).
    const hex = baseColorHex(b.baseChar)
    tmpColor.setRGB(
      srgbToLinear(((hex >> 16) & 0xff) / 255),
      srgbToLinear(((hex >> 8) & 0xff) / 255),
      srgbToLinear((hex & 0xff) / 255),
    )
    const cr = tmpColor.r, cg = tmpColor.g, cb = tmpColor.b

    // Convenience writers — bumps vOff after each call.
    const writeV = (x: number, y: number, z: number, nxV: number, nyV: number, nzV: number) => {
      const off = vOff * 3
      positions[off]     = x
      positions[off + 1] = y
      positions[off + 2] = z
      normals[off]       = nxV
      normals[off + 1]   = nyV
      normals[off + 2]   = nzV
      colors[off]        = cr
      colors[off + 1]    = cg
      colors[off + 2]    = cb
      vOff++
    }
    const writeTri = (a: number, c2: number, b2: number) => {
      // Note arg order matches "a, b, c" with CCW winding when viewed
      // from the side the normal points.
      indices[iOff]     = a
      indices[iOff + 1] = c2
      indices[iOff + 2] = b2
      iOff += 3
    }

    // ── Top face ──────────────────────────────────────────────────────
    // Centroid first, then N perimeter atoms, all with +normal.
    const topCentroidV = vOff
    writeV(cx + ox, cy + oy, cz + oz, nx, ny, nz)
    const topPerimStart = vOff
    for (let i = 0; i < N; i++) {
      writeV(
        b.perimeter[i * 3]     + ox,
        b.perimeter[i * 3 + 1] + oy,
        b.perimeter[i * 3 + 2] + oz,
        nx, ny, nz,
      )
    }
    // N triangles (centroid → perim[i] → perim[i+1]). CCW from +normal.
    for (let i = 0; i < N; i++) {
      const a = topPerimStart + i
      const c2 = topPerimStart + ((i + 1) % N)
      writeTri(topCentroidV, a, c2)
    }

    // ── Bottom face ───────────────────────────────────────────────────
    // Mirror of top, with -normal and reversed winding (so triangles
    // face -normal direction).
    const botCentroidV = vOff
    writeV(cx - ox, cy - oy, cz - oz, -nx, -ny, -nz)
    const botPerimStart = vOff
    for (let i = 0; i < N; i++) {
      writeV(
        b.perimeter[i * 3]     - ox,
        b.perimeter[i * 3 + 1] - oy,
        b.perimeter[i * 3 + 2] - oz,
        -nx, -ny, -nz,
      )
    }
    for (let i = 0; i < N; i++) {
      const a = botPerimStart + i
      const c2 = botPerimStart + ((i + 1) % N)
      // Reversed: centroid → perim[i+1] → perim[i]
      writeTri(botCentroidV, c2, a)
    }

    // ── Side walls — one quad per perimeter edge ──────────────────────
    // Each quad gets its own 4 vertices with a face-aligned normal so
    // flat shading is clean. For edge i → (i+1)%N:
    //   normal_side = normalize((perim[i+1] − perim[i]) × normal)
    // Two triangles per quad.
    for (let i = 0; i < N; i++) {
      const j = (i + 1) % N
      const ax = b.perimeter[i * 3],     ay = b.perimeter[i * 3 + 1],     az = b.perimeter[i * 3 + 2]
      const bx = b.perimeter[j * 3],     by = b.perimeter[j * 3 + 1],     bz = b.perimeter[j * 3 + 2]
      // Edge direction.
      const ex = bx - ax, ey = by - ay, ez = bz - az
      // Outward normal = edge × planeNormal (right-hand rule with CCW
      // perimeter winding gives outward).
      let snx = ey * nz - ez * ny
      let sny = ez * nx - ex * nz
      let snz = ex * ny - ey * nx
      const sl = Math.hypot(snx, sny, snz) || 1
      snx /= sl; sny /= sl; snz /= sl

      const topA = vOff
      writeV(ax + ox, ay + oy, az + oz, snx, sny, snz)
      const botA = vOff
      writeV(ax - ox, ay - oy, az - oz, snx, sny, snz)
      const topB = vOff
      writeV(bx + ox, by + oy, bz + oz, snx, sny, snz)
      const botB = vOff
      writeV(bx - ox, by - oy, bz - oz, snx, sny, snz)
      // Two triangles (CCW from outward normal):
      //   (topA, botA, topB) and (topB, botA, botB)
      writeTri(topA, botA, topB)
      writeTri(topB, botA, botB)
    }
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(positions, 3))
  geometry.setAttribute('normal', new BufferAttribute(normals, 3))
  geometry.setAttribute('color', new BufferAttribute(colors, 3))
  geometry.setIndex(new BufferAttribute(indices, 1))
  geometry.computeBoundingSphere()

  const material = new MeshPhysicalNodeMaterial({
    color: 0xffffff,
    vertexColors: true,
    metalness: opts.material.metalness,
    roughness: opts.material.roughness,
    clearcoat: opts.material.clearcoat,
    clearcoatRoughness: opts.material.clearcoatRoughness,
    envMapIntensity: opts.material.envMapIntensity,
  })

  const mesh = new Mesh(geometry, material)
  mesh.frustumCulled = false

  return {
    mesh,
    geometry,
    material,
    dispose: () => {
      geometry.dispose()
      material.dispose()
    },
  }
}
