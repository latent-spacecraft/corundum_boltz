/**
 * LiquidGlass — CSS `backdrop-filter` clipped to the atom silhouette.
 *
 * The browser does the heavy lifting: a single `<div>` overlay sets
 * `backdrop-filter: blur() brightness() saturate()` and `clip-path:
 * polygon(...)`. The clip path is the convex hull of the projected atom
 * positions, smoothed by two passes of Chaikin to round the corners
 * into a liquid-glass blob.
 *
 * No canvas pixel copying, no destination-in masking — `backdrop-filter`
 * is the proper primitive for "blur whatever is behind me", and it
 * picks up the live Mol* canvas pixels without us touching them.
 *
 *   backdrop-filter ── runs once per frame, GPU-accelerated, hits the
 *                      composited layer beneath this div.
 *   clip-path       ── restricts the filtered area to the silhouette.
 *   box-shadow inset ─ inner rim highlight, follows the clip boundary.
 *   background       ── flat tint over the blurred backdrop.
 *
 * The animation loop only does:
 *   - poll Mol* camera
 *   - project atoms (~3-5 mat-vec mults per atom)
 *   - convex hull + Chaikin smoothing (~1ms for 1000 atoms)
 *   - write polygon points back to inline style
 *
 * No React re-renders inside the loop; the div ref is mutated directly.
 */
import { useEffect, useRef } from 'react'

export interface LiquidGlassParams {
  /** CSS-pixel blur on the backdrop. */
  blur: number
  /** Brightness multiplier on the blurred backdrop (1.0 neutral). */
  brightness: number
  /** Saturation multiplier (1.0 neutral). */
  saturation: number
  /** Pixels to inflate the convex hull outward, so the silhouette wraps
   *  the atoms cleanly instead of clipping at their centers. */
  envelopePad: number
  /** Chaikin smoothing iterations on the hull (0 = sharp polygon). */
  smoothIterations: number
  /** Tint color (hex int). 0 = none. */
  tintColor: number
  /** 0-1 strength of the tint overlay. */
  tintAmount: number
  /** 0-1 strength of the inner rim highlight (inset box-shadow alpha). */
  edgeHighlight: number
  /** Pixels of the inner-rim blur radius. */
  edgeWidth: number
}

export interface CameraSnapshot {
  /** Field of view in radians (Mol* convention). */
  fov: number
  position: [number, number, number]
  up: [number, number, number]
  target: [number, number, number]
}

export interface LiquidGlassProps {
  /** Atom XYZ positions packed [x0,y0,z0,...]. Null hides the overlay. */
  atomPositions: Float32Array | null
  cameraSnapshot: () => CameraSnapshot | null
  params: LiquidGlassParams
  width: number
  height: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline matrix math — column-major OpenGL convention. Avoids pulling
// gl-matrix or three.js for this overlay.

function lookAt(
  out: Float32Array,
  eye: readonly [number, number, number],
  target: readonly [number, number, number],
  up: readonly [number, number, number],
) {
  let zx = eye[0] - target[0]
  let zy = eye[1] - target[1]
  let zz = eye[2] - target[2]
  let zl = Math.hypot(zx, zy, zz) || 1
  zx /= zl; zy /= zl; zz /= zl
  let xx = up[1] * zz - up[2] * zy
  let xy = up[2] * zx - up[0] * zz
  let xz = up[0] * zy - up[1] * zx
  let xl = Math.hypot(xx, xy, xz) || 1
  xx /= xl; xy /= xl; xz /= xl
  const yx = zy * xz - zz * xy
  const yy = zz * xx - zx * xz
  const yz = zx * xy - zy * xx
  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2])
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2])
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2])
  out[15] = 1
}

function perspective(out: Float32Array, fovRad: number, aspect: number, near: number, far: number) {
  const f = 1 / Math.tan(fovRad / 2)
  const nf = 1 / (near - far)
  out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
  out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
  out[8] = 0; out[9] = 0; out[10] = (far + near) * nf; out[11] = -1
  out[12] = 0; out[13] = 0; out[14] = 2 * far * near * nf; out[15] = 0
}

/**
 * Computes `out = a * b` in column-major flat-array layout. (The prior
 * implementation in this file had the operand order reversed — which
 * silently broke atom projection because vp came out as view*proj
 * instead of proj*view.)
 */
function multMat4(out: Float32Array, a: Float32Array, b: Float32Array) {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4]
    const b1 = b[c * 4 + 1]
    const b2 = b[c * 4 + 2]
    const b3 = b[c * 4 + 3]
    out[c * 4]     = a[0] * b0 + a[4] * b1 + a[8]  * b2 + a[12] * b3
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9]  * b2 + a[13] * b3
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convex hull (Andrew's monotone chain) — sorts by x, then walks lower/upper.
// Returns the hull points in CCW order.

interface P { x: number; y: number }

function convexHull(pts: P[]): P[] {
  const n = pts.length
  if (n < 3) return pts.slice()
  const sorted = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y)
  const cross = (O: P, A: P, B: P) =>
    (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
  const lower: P[] = []
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop()
    }
    lower.push(p)
  }
  const upper: P[] = []
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop()
    }
    upper.push(p)
  }
  lower.pop()
  upper.pop()
  return lower.concat(upper)
}

/** Inflate a polygon outward from its centroid by `pad` pixels. */
function padHull(hull: P[], pad: number): P[] {
  if (hull.length < 3 || pad === 0) return hull
  let cx = 0, cy = 0
  for (const p of hull) { cx += p.x; cy += p.y }
  cx /= hull.length; cy /= hull.length
  return hull.map((p) => {
    const dx = p.x - cx
    const dy = p.y - cy
    const d = Math.hypot(dx, dy) || 1
    return { x: p.x + (dx / d) * pad, y: p.y + (dy / d) * pad }
  })
}

/** Chaikin corner-cutting — each iteration replaces every vertex with two
 *  points at 1/4 and 3/4 along each edge. Two iterations produce a
 *  visually smooth blob (4× the vertex count). */
function chaikin(pts: P[], iterations: number): P[] {
  let out = pts
  for (let it = 0; it < iterations; it++) {
    const n = out.length
    if (n < 3) break
    const next: P[] = new Array(n * 2)
    for (let i = 0; i < n; i++) {
      const a = out[i]
      const b = out[(i + 1) % n]
      next[i * 2]     = { x: a.x * 0.75 + b.x * 0.25, y: a.y * 0.75 + b.y * 0.25 }
      next[i * 2 + 1] = { x: a.x * 0.25 + b.x * 0.75, y: a.y * 0.25 + b.y * 0.75 }
    }
    out = next
  }
  return out
}

function hexCssRgba(hex: number, a: number): string {
  const r = (hex >> 16) & 0xff
  const g = (hex >> 8) & 0xff
  const b = hex & 0xff
  return `rgba(${r}, ${g}, ${b}, ${a})`
}

export function LiquidGlass({
  atomPositions,
  cameraSnapshot,
  params,
  width,
  height,
}: LiquidGlassProps) {
  const divRef = useRef<HTMLDivElement | null>(null)

  // Hot-path refs — tick reads through these so prop changes don't
  // restart the animation loop or churn React.
  const paramsRef = useRef(params)
  const atomsRef = useRef(atomPositions)
  const cameraRef = useRef(cameraSnapshot)
  useEffect(() => { paramsRef.current = params }, [params])
  useEffect(() => { atomsRef.current = atomPositions }, [atomPositions])
  useEffect(() => { cameraRef.current = cameraSnapshot }, [cameraSnapshot])

  useEffect(() => {
    const div = divRef.current
    if (!div) return

    const view = new Float32Array(16)
    const proj = new Float32Array(16)
    const vp = new Float32Array(16)
    const projectedScratch: P[] = []

    let raf = 0
    let disposed = false

    const tick = () => {
      if (disposed) return
      raf = requestAnimationFrame(tick)

      const snap = cameraRef.current()
      const atoms = atomsRef.current
      const p = paramsRef.current
      const W = width
      const H = height

      if (!snap || !atoms || atoms.length === 0 || W < 2 || H < 2) {
        // Empty path → div fully clipped out → no backdrop effect, but
        // also no flash of unmasked blur.
        div.style.clipPath = 'polygon(0 0, 0 0, 0 0)'
        return
      }

      perspective(proj, snap.fov, W / H, 0.1, 10000)
      lookAt(view, snap.position, snap.target, snap.up)
      multMat4(vp, proj, view)

      // Project atoms to screen pixel coords. Reuse the scratch array.
      projectedScratch.length = 0
      for (let i = 0; i < atoms.length; i += 3) {
        const x = atoms[i]
        const y = atoms[i + 1]
        const z = atoms[i + 2]
        const cx = vp[0] * x + vp[4] * y + vp[8]  * z + vp[12]
        const cy = vp[1] * x + vp[5] * y + vp[9]  * z + vp[13]
        const cw = vp[3] * x + vp[7] * y + vp[11] * z + vp[15]
        if (cw <= 0) continue
        const ndcX = cx / cw
        const ndcY = cy / cw
        const px = (ndcX * 0.5 + 0.5) * W
        const py = (1 - (ndcY * 0.5 + 0.5)) * H
        projectedScratch.push({ x: px, y: py })
      }

      if (projectedScratch.length < 3) {
        div.style.clipPath = 'polygon(0 0, 0 0, 0 0)'
        return
      }

      const hull = convexHull(projectedScratch)
      const padded = padHull(hull, p.envelopePad)
      const smoothed = chaikin(padded, Math.max(0, p.smoothIterations | 0))

      // Build the polygon string for clip-path. We use absolute pixels
      // since the div fills the container exactly (inset: 0).
      let poly = 'polygon('
      for (let i = 0; i < smoothed.length; i++) {
        const pt = smoothed[i]
        if (i > 0) poly += ', '
        poly += pt.x.toFixed(1) + 'px ' + pt.y.toFixed(1) + 'px'
      }
      poly += ')'

      div.style.clipPath = poly
      // Some browsers still want the -webkit prefix for clip-path
      // animation smoothness on certain GPUs.
      ;(div.style as any).webkitClipPath = poly

      div.style.backdropFilter =
        `blur(${p.blur}px) brightness(${p.brightness}) saturate(${p.saturation})`
      ;(div.style as any).webkitBackdropFilter =
        `blur(${p.blur}px) brightness(${p.brightness}) saturate(${p.saturation})`
      div.style.background =
        p.tintAmount > 0 ? hexCssRgba(p.tintColor, p.tintAmount) : 'transparent'
      // Inner rim glow — inset box-shadow follows the clip-path boundary,
      // so we get a free edge highlight that traces the silhouette.
      div.style.boxShadow =
        p.edgeHighlight > 0
          ? `inset 0 0 ${p.edgeWidth}px ${p.edgeWidth * 0.4}px rgba(255,255,255,${p.edgeHighlight})`
          : 'none'
    }

    raf = requestAnimationFrame(tick)
    return () => {
      disposed = true
      cancelAnimationFrame(raf)
    }
    // Mount/unmount only — props read through refs above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, height])

  return (
    <div
      ref={divRef}
      style={{
        position: 'absolute',
        inset: 0,
        // Mol* needs the orbit interactions.
        pointerEvents: 'none',
        // Initial clip — empty so we never flash an unmasked blurred
        // backdrop on first paint before tick() runs.
        clipPath: 'polygon(0 0, 0 0, 0 0)',
      }}
    />
  )
}
