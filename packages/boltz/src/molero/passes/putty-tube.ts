/**
 * Variable-cross-section tube builder — extrudes an *elliptical* cross-
 * section along a curve, with per-sample (radiusU, radiusV) and
 * per-sample color.
 *
 * Two perpendicular radii distinguish ribbon styles by SS:
 *   - radiusU = radiusV → circle (coil / nucleic)
 *   - radiusU > radiusV → flat ribbon in the binormal axis (helix)
 *   - radiusU ≫ radiusV → very thin flat ribbon (sheet body)
 * U corresponds to the curve's binormal axis; V to its normal axis.
 *
 * Three.js's stock `TubeGeometry` is constant circular. We sweep our
 * own polygon using the same Frenet-frame parallel-transport math,
 * with per-sample radii/colors closed over each ring.
 *
 * Layout:
 *   - `(tubularSegments + 1) × (radialSegments + 1)` vertices
 *   - one ring per tube sample; the +1 radial vertex duplicates the
 *     seam so smooth-shaded normals don't twist at the boundary
 *   - indices stitch each quad into two triangles
 *
 * Inputs (`radiiU`, `radiiV`, `colors`) are sampled at the same set of
 * `t`s the geometry is built on: `i / tubularSegments` for `i ∈ [0, N]`.
 * Caller is responsible for resampling per-residue values onto that grid.
 *
 * Surface normals are computed correctly for the elliptical shape via
 * `(x_v × x_u_perp)` — for a circle this collapses to the radial vector;
 * for elliptical, it accounts for the per-axis scale so highlights land
 * where they should on the flat ribbon edges.
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
   *  `tubularSegments + 1` rings; `radiiU.length` must match that. */
  tubularSegments: number
  /** Radial segments per ring (e.g. 12 = dodecagonal). */
  radialSegments: number
  /** Per-sample radius along the curve binormal axis. Length =
   *  `tubularSegments + 1`. The "wide" axis of a flat ribbon. */
  radiiU: Float32Array
  /** Per-sample radius along the curve normal axis. Length =
   *  `tubularSegments + 1`. The "thin" axis of a flat ribbon. */
  radiiV: Float32Array
  /** Per-sample linear-RGB color, length = `3 * (tubularSegments + 1)`.
   *  Pass `null` to omit the `color` attribute. */
  colors: Float32Array | null
  /** Close the tube into a loop. Almost never wanted for ribbons. */
  closed: boolean
}

export function buildVariableRadiusTube(opts: VariableTubeOptions): BufferGeometry {
  const { curve, tubularSegments, radialSegments, radiiU, radiiV, colors, closed } = opts
  const ringCount = tubularSegments + 1
  if (radiiU.length !== ringCount) {
    throw new Error(`radiiU length ${radiiU.length} must equal tubularSegments+1 (${ringCount})`)
  }
  if (radiiV.length !== ringCount) {
    throw new Error(`radiiV length ${radiiV.length} must equal tubularSegments+1 (${ringCount})`)
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
    const rU = radiiU[i]
    const rV = radiiV[i]
    const cr = colors
      ? [colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]]
      : null

    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * Math.PI * 2
      const sinV = Math.sin(v)
      const cosV = -Math.cos(v) // matches TubeGeometry's outward-facing convention
      // Position: ellipse parameterized as (rU·cos·B + rV·sin·N).
      // U is the binormal axis ("wide" axis of a flat ribbon — the
      // direction perpendicular to the curve's center-of-curvature
      // pull). V is the normal axis (toward the helix axis for a
      // helix — the "thin" axis of a flat ribbon). This puts the wide
      // face of helix/sheet ribbons perpendicular to the direction of
      // travel: looking at a helix from outside the cylinder, you see
      // the ribbon flat-on as it sweeps.
      const px = P.x + rU * cosV * B.x + rV * sinV * N.x
      const py = P.y + rU * cosV * B.y + rV * sinV * N.y
      const pz = P.z + rU * cosV * B.z + rV * sinV * N.z
      positions[vi]     = px
      positions[vi + 1] = py
      positions[vi + 2] = pz
      // Normal: outward-pointing normal of the elliptical cross-section.
      // For an ellipse x²/a² + y²/b² = 1, the outward normal at angle θ is
      // (b·cosθ, a·sinθ) (un-normalized). So in our (B, N) UV frame:
      //   n_uv = (rV·cosV along B, rU·sinV along N)
      // (note the swap: the wider axis has the smaller normal component
      // — highlights land where curvature is highest, on the narrow ends).
      const nuRaw = rV * cosV
      const nvRaw = rU * sinV
      let nx = nuRaw * B.x + nvRaw * N.x
      let ny = nuRaw * B.y + nvRaw * N.y
      let nz = nuRaw * B.z + nvRaw * N.z
      const nl = Math.hypot(nx, ny, nz) || 1
      nx /= nl; ny /= nl; nz /= nl
      normals[vi]     = nx
      normals[vi + 1] = ny
      normals[vi + 2] = nz
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
