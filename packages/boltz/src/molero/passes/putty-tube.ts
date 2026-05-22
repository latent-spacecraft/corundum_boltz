/**
 * Variable-radius tube builder — extrudes a circular cross-section along
 * a curve with per-sample radius and per-sample color.
 *
 * Three.js's stock `TubeGeometry` is constant-radius. We need
 * per-residue radius for the putty effect (thick at confident residues,
 * thin at low pLDDT) and per-residue color for SS tinting. Same Frenet-
 * frame parallel-transport math as `TubeGeometry`, just with the radius
 * and color closed over each tube sample.
 *
 * The geometry layout is:
 *   - `(tubularSegments + 1) × (radialSegments + 1)` vertices
 *   - one ring per tube sample; the +1 radial vertex duplicates the
 *     seam so smooth-shaded normals don't twist at the boundary
 *   - indices stitch each quad into two triangles
 *
 * Inputs `radii[]` and `colors[]` are sampled at the same set of `t`s
 * the geometry is built on: `i / tubularSegments` for `i ∈ [0, N]`. The
 * caller is responsible for resampling per-residue values onto that
 * grid.
 */
import {
  BufferAttribute,
  BufferGeometry,
  Float32BufferAttribute,
  Uint32BufferAttribute,
  Vector3,
  type CatmullRomCurve3,
} from 'three/webgpu'

export interface VariableTubeOptions {
  /** Curve to extrude along. */
  curve: CatmullRomCurve3
  /** Number of tube samples (rings). The resulting geometry has
   *  `tubularSegments + 1` rings; `radii.length` must match that. */
  tubularSegments: number
  /** Radial segments per ring (e.g. 10 = decagonal). */
  radialSegments: number
  /** Per-sample radius, length = `tubularSegments + 1`. */
  radii: Float32Array
  /** Per-sample linear-RGB color, length = `3 * (tubularSegments + 1)`.
   *  Pass `null` to omit the `color` attribute. */
  colors: Float32Array | null
  /** Close the tube into a loop. Almost never wanted for ribbons. */
  closed: boolean
}

export function buildVariableRadiusTube(opts: VariableTubeOptions): BufferGeometry {
  const { curve, tubularSegments, radialSegments, radii, colors, closed } = opts
  const ringCount = tubularSegments + 1
  if (radii.length !== ringCount) {
    throw new Error(`radii length ${radii.length} must equal tubularSegments+1 (${ringCount})`)
  }
  if (colors && colors.length !== ringCount * 3) {
    throw new Error(`colors length ${colors.length} must equal 3*(tubularSegments+1) (${ringCount * 3})`)
  }

  const frames = curve.computeFrenetFrames(tubularSegments, closed)
  const ringStride = radialSegments + 1

  // Output buffers — sized exactly to avoid push-array overhead.
  const vertexCount = ringCount * ringStride
  const positions = new Float32Array(vertexCount * 3)
  const normals = new Float32Array(vertexCount * 3)
  const colorBuf = colors ? new Float32Array(vertexCount * 3) : null
  // Index count: tubularSegments * radialSegments * 6 (two tris per quad).
  const indexCount = tubularSegments * radialSegments * 6
  const indices = vertexCount > 65535
    ? new Uint32Array(indexCount)
    : new Uint16Array(indexCount)

  const P = new Vector3()
  let vi = 0 // vertex-write cursor in floats (positions[vi])

  for (let i = 0; i < ringCount; i++) {
    const t = i / tubularSegments
    curve.getPointAt(t, P)
    const N = frames.normals[i]
    const B = frames.binormals[i]
    const r = radii[i]
    const cr = colors
      ? [colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]]
      : null

    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * Math.PI * 2
      const sinV = Math.sin(v)
      const cosV = -Math.cos(v) // matches TubeGeometry's outward-facing convention
      let nx = cosV * N.x + sinV * B.x
      let ny = cosV * N.y + sinV * B.y
      let nz = cosV * N.z + sinV * B.z
      // Normalize defensively; degenerate frames at flat spline segments
      // can produce zero-length tangents.
      const nl = Math.hypot(nx, ny, nz) || 1
      nx /= nl; ny /= nl; nz /= nl
      normals[vi]     = nx
      normals[vi + 1] = ny
      normals[vi + 2] = nz
      positions[vi]     = P.x + r * nx
      positions[vi + 1] = P.y + r * ny
      positions[vi + 2] = P.z + r * nz
      if (colorBuf && cr) {
        colorBuf[vi]     = cr[0]
        colorBuf[vi + 1] = cr[1]
        colorBuf[vi + 2] = cr[2]
      }
      vi += 3
    }
  }

  // Generate triangle indices ring-by-ring.
  let ii = 0
  for (let j = 1; j <= tubularSegments; j++) {
    for (let i = 1; i <= radialSegments; i++) {
      const a = ringStride * (j - 1) + (i - 1)
      const b = ringStride * j + (i - 1)
      const c = ringStride * j + i
      const d = ringStride * (j - 1) + i
      indices[ii    ] = a
      indices[ii + 1] = b
      indices[ii + 2] = d
      indices[ii + 3] = b
      indices[ii + 4] = c
      indices[ii + 5] = d
      ii += 6
    }
  }

  const geom = new BufferGeometry()
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geom.setAttribute('normal', new Float32BufferAttribute(normals, 3))
  if (colorBuf) geom.setAttribute('color', new Float32BufferAttribute(colorBuf, 3))
  // Index attribute — keep the typed array shape that fits the vertex count.
  const indexAttr = indices instanceof Uint32Array
    ? new Uint32BufferAttribute(indices, 1)
    : new BufferAttribute(indices, 1)
  geom.setIndex(indexAttr)
  geom.computeBoundingSphere()
  return geom
}
