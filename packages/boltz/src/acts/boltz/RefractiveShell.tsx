/**
 * RefractiveShell — true Three.js gem shell wrapping the molecular structure.
 *
 * Sits on top of the Mol* canvas as a transparent WebGL overlay. The shell
 * geometry is the convex hull of the atom positions (with a small outward
 * inflate so it wraps cleanly rather than touching atom centres), rendered
 * with `createGemMaterial()` — same factory the corner logo uses, so the
 * structure and the wordmark share one visual register.
 *
 * Camera mirror: each frame we poll Mol*'s live camera (via the
 * `cameraSnapshot` callback also feeding LiquidGlass) and set our
 * PerspectiveCamera to the same fov / position / target / up. Orbiting
 * Mol* drags the gem shell along with it pixel-for-pixel.
 *
 * Material attenuation is scaled to the structure's spatial extent so the
 * tint reads on a 30 Å protein the same way it reads on a 1-unit logo gem.
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js'
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createGemMaterial, GEM_PRESETS, type GemMaterialOpts } from '@/components/gemMaterial'
import type { CameraSnapshot } from './LiquidGlass'

export type GemPreset = keyof typeof GEM_PRESETS

/** Geometry style for the shell. */
export type ShellShape = 'faceted' | 'smooth'

export interface RefractiveShellProps {
  /** Atom XYZ positions packed [x0,y0,z0,...] in the same coord space as
   *  the Mol* camera reports (Å). Null hides the overlay. */
  atomPositions: Float32Array | null
  cameraSnapshot: () => CameraSnapshot | null
  width: number
  height: number
  /**
   * Underlying Mol* canvas. When supplied, its live pixels become the
   * Three.js scene background, so the gem's transmission shader refracts
   * the actual chain rendering instead of empty space. The Mol* DOM
   * canvas is still drawn by Mol* — we just sample it as a texture every
   * frame and let the Three.js layer be the only visible one.
   */
  backdropCanvas?: HTMLCanvasElement | null
  /**
   * 'faceted' = convex hull of atom positions (crystalline, sharp facets).
   * 'smooth'  = bbox-aligned ellipsoid (polished lens; reads as quartz/glass
   *             and lets the chain refract through cleanly).
   */
  shape?: ShellShape
  /** Outward inflate in Å — gives the shell some headroom over atom centres. */
  padding?: number
  /** Pre-tuned material preset. */
  preset?: GemPreset
  /**
   * Multiplier on the structure's spatial radius for the attenuation path
   * length. Higher = lighter tint / more chain visibility. Default 1.0 —
   * gives a balanced look on most molecules.
   */
  attenuationFactor?: number
  /** Per-call overrides on the resolved material spec (preset + factor). */
  materialOverrides?: Partial<GemMaterialOpts>
}

/** Pack atom positions into a Three.js Vector3 array. */
function toVec3Array(positions: Float32Array): THREE.Vector3[] {
  const out: THREE.Vector3[] = []
  for (let i = 0; i + 2 < positions.length; i += 3) {
    out.push(new THREE.Vector3(positions[i], positions[i + 1], positions[i + 2]))
  }
  return out
}

/** Centroid of a point set. */
function centroidOf(pts: THREE.Vector3[]): THREE.Vector3 {
  const c = new THREE.Vector3()
  for (const p of pts) c.add(p)
  if (pts.length) c.divideScalar(pts.length)
  return c
}

/** Maximum distance from centroid — useful for scaling attenuation to scene. */
function radiusOf(pts: THREE.Vector3[], centroid: THREE.Vector3): number {
  let max = 0
  for (const p of pts) {
    const d = p.distanceTo(centroid)
    if (d > max) max = d
  }
  return max
}

export function RefractiveShell({
  atomPositions,
  cameraSnapshot,
  width,
  height,
  backdropCanvas = null,
  shape = 'faceted',
  padding = 1.5,
  preset = 'ruby',
  attenuationFactor = 1.0,
  materialOverrides,
}: RefractiveShellProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Hold the latest props in refs so the animation loop sees them without
  // restarting on every re-render. Geometry / material rebuilds key off
  // identity changes inside the tick.
  const propsRef = useRef({
    atomPositions,
    shape,
    padding,
    preset,
    attenuationFactor,
    materialOverrides,
    cameraSnapshot,
    width,
    height,
    backdropCanvas,
  })
  propsRef.current = {
    atomPositions,
    shape,
    padding,
    preset,
    attenuationFactor,
    materialOverrides,
    cameraSnapshot,
    width,
    height,
    backdropCanvas,
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      premultipliedAlpha: true,
    })
    renderer.setPixelRatio(dpr)
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    renderer.outputColorSpace = THREE.SRGBColorSpace

    const scene = new THREE.Scene()

    // PMREM-baked room env so transmission has IBL highlights. The chain
    // refraction itself comes from `scene.background` below.
    const pmrem = new THREE.PMREMGenerator(renderer)
    pmrem.compileEquirectangularShader()
    const envScene = new RoomEnvironment()
    const envRT = pmrem.fromScene(envScene, 0.04)
    scene.environment = envRT.texture

    // Use the live Mol* canvas as the scene background. Three.js's
    // MeshPhysicalMaterial transmission samples the scene (including the
    // background) when refracting, so this is how the chain ends up visible
    // *through* the gem. We mark needsUpdate every frame so the texture
    // keeps in step with Mol*'s own redraws.
    let backdropTexture: THREE.CanvasTexture | null = null
    const initialBackdrop = propsRef.current.backdropCanvas
    if (initialBackdrop) {
      backdropTexture = new THREE.CanvasTexture(initialBackdrop)
      backdropTexture.colorSpace = THREE.SRGBColorSpace
      // Mol*'s canvas Y matches CSS Y (top-down). Three.js's default flipY
      // sends rows the right way around for that source.
      scene.background = backdropTexture
    }

    const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 10000)

    // Shell state. Separate slots so material-only edits (slider drags) can
    // swap the material without rebuilding the geometry — marching cubes
    // takes ~15-30 ms on a typical protein and we don't want that latency
    // on every IOR keystroke.
    let shellMesh: THREE.Mesh | null = null
    let material: THREE.MeshPhysicalMaterial | null = null
    let currentRadius = 0 // structure radius, cached for material scaling

    const disposeShell = () => {
      if (shellMesh) {
        scene.remove(shellMesh)
        shellMesh.geometry.dispose()
        shellMesh = null
      }
      if (material) {
        material.dispose()
        material = null
      }
    }

    /** Faceted: convex hull of inflated atom positions. World-space verts. */
    const buildFacetedMesh = (
      pts: THREE.Vector3[],
      c: THREE.Vector3,
      pad: number,
      mat: THREE.MeshPhysicalMaterial,
    ): THREE.Mesh | null => {
      const inflated = pts.map((p) => {
        const dir = p.clone().sub(c)
        const len = dir.length()
        if (len < 1e-6) return p.clone()
        return p.clone().add(dir.multiplyScalar(pad / len))
      })
      let geo: THREE.BufferGeometry
      try {
        geo = new ConvexGeometry(inflated)
      } catch (e) {
        console.warn('[RefractiveShell] convex hull failed:', e)
        return null
      }
      return new THREE.Mesh(geo, mat)
    }

    /**
     * Smooth: marching-cubes iso-surface of the atomic Gaussian field.
     *
     * Each atom contributes a metaball to a regular 3D grid; the iso-surface
     * is extracted at a tuned threshold to give a smooth envelope that
     * follows the chain's actual topology (lobes, concavities, the gap
     * between the helix and turn on a Trp-cage, …). This is the molecular-
     * surface look — a true SES would require atomic radii and probe
     * rolling, but for a refractive shell the metaball envelope reads the
     * same and is much cheaper.
     *
     * Resolution 48 → ≈ 20-40k triangles, builds in ~15 ms on a 50-aa
     * protein. The MarchingCubes object renders directly as a Mesh so we
     * use it in-place rather than extracting a static BufferGeometry.
     */
    const buildSmoothMesh = (
      pts: THREE.Vector3[],
      pad: number,
      mat: THREE.MeshPhysicalMaterial,
    ): THREE.Mesh => {
      let xMin = Infinity, yMin = Infinity, zMin = Infinity
      let xMax = -Infinity, yMax = -Infinity, zMax = -Infinity
      for (const p of pts) {
        if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x
        if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y
        if (p.z < zMin) zMin = p.z; if (p.z > zMax) zMax = p.z
      }
      // Pad on all sides so the iso-surface has headroom around the outer
      // atoms — otherwise the marching-cubes grid clips the metaballs at
      // the silhouette and we get flat patches there.
      const cubeMargin = pad + 3 // extra ~3 Å for metaball field falloff
      xMin -= cubeMargin; xMax += cubeMargin
      yMin -= cubeMargin; yMax += cubeMargin
      zMin -= cubeMargin; zMax += cubeMargin

      // Uniform cube — MarchingCubes operates on [-1, 1]^3 internally and
      // mesh.scale is a single THREE.Vector3, so anisotropic dimensions
      // would distort metaball shapes. Use the longest axis as the cube
      // size; non-cubic molecules waste some grid cells but render fine.
      const cubeHalf = Math.max(xMax - xMin, yMax - yMin, zMax - zMin) / 2
      const cubeCx = (xMin + xMax) / 2
      const cubeCy = (yMin + yMax) / 2
      const cubeCz = (zMin + zMax) / 2

      const resolution = 48
      const maxPoly = 100_000
      const mc = new MarchingCubes(resolution, mat, true, false, maxPoly)
      // Iso threshold — lower = surface farther from atoms (chunky blob).
      // 80 is the addon's default; 60-100 is the useful range here.
      mc.isolation = 80

      mc.reset()
      // Metaball params. Each ball's effective radius in normalized [0, 1]
      // cube units is ~sqrt(strength/subtract). With strength 0.6 / subtract
      // 10 → radius 0.245 cube units. For a ~30 Å molecule with cubeHalf
      // ~15-18 Å, that's ~3.7-4.4 Å in atom space — close to vdW radius for
      // the canonical "fat tube" molecular surface.
      const strength = 0.6
      const subtract = 10
      for (const p of pts) {
        const bx = (p.x - cubeCx) / (2 * cubeHalf) + 0.5
        const by = (p.y - cubeCy) / (2 * cubeHalf) + 0.5
        const bz = (p.z - cubeCz) / (2 * cubeHalf) + 0.5
        if (bx < 0 || bx > 1 || by < 0 || by > 1 || bz < 0 || bz > 1) continue
        mc.addBall(bx, by, bz, strength, subtract)
      }
      mc.update()

      // Position + scale the [-1, 1] cube into world coordinates so the
      // surface aligns with the atoms.
      mc.position.set(cubeCx, cubeCy, cubeCz)
      mc.scale.setScalar(cubeHalf)
      return mc
    }

    /** Compose preset + scene-scale (thickness + scaled distance) + user overrides. */
    const buildMaterial = (
      presetName: GemPreset,
      attFactor: number,
      r: number,
      overrides: Partial<GemMaterialOpts> | undefined,
    ): THREE.MeshPhysicalMaterial => {
      const baseOpts = GEM_PRESETS[presetName]
      const scaledDistance = (baseOpts.attenuationDistance ?? 0.5) * attFactor * r * 2
      return createGemMaterial({
        ...baseOpts,
        attenuationDistance: Math.max(0.1, scaledDistance),
        thickness: r,
        ...(overrides ?? {}),
      })
    }

    /** Full rebuild: geometry + material. ~15-30 ms for smooth, <5 ms for faceted. */
    const rebuildShell = (
      positions: Float32Array,
      shapeName: ShellShape,
      pad: number,
      presetName: GemPreset,
      attFactor: number,
      overrides: Partial<GemMaterialOpts> | undefined,
    ) => {
      disposeShell()
      const pts = toVec3Array(positions)
      if (pts.length < 4) return
      const c = centroidOf(pts)
      const r = radiusOf(pts, c)
      currentRadius = r

      material = buildMaterial(presetName, attFactor, r, overrides)
      const mesh =
        shapeName === 'smooth'
          ? buildSmoothMesh(pts, pad, material)
          : buildFacetedMesh(pts, c, pad, material)
      if (!mesh) {
        material.dispose()
        material = null
        return
      }
      mesh.renderOrder = 1
      shellMesh = mesh
      scene.add(shellMesh)
    }

    /** Cheap: swap just the material on the existing shell mesh. */
    const swapMaterial = (
      presetName: GemPreset,
      attFactor: number,
      overrides: Partial<GemMaterialOpts> | undefined,
    ) => {
      if (!shellMesh) return
      const newMat = buildMaterial(presetName, attFactor, currentRadius, overrides)
      material?.dispose()
      material = newMat
      shellMesh.material = newMat
    }

    // Geometry fingerprint: changes force a full marching-cubes / hull rebuild.
    // Material fingerprint: changes only swap the material instance.
    let lastGeoFp: string | null = null
    let lastMatFp: string | null = null
    const geoFingerprintOf = (p: typeof propsRef.current) =>
      `${p.atomPositions ? p.atomPositions.byteLength : 'null'}|${p.shape}|${p.padding}`
    const matFingerprintOf = (p: typeof propsRef.current) =>
      `${p.preset}|${p.attenuationFactor}|${JSON.stringify(p.materialOverrides ?? null)}`

    if (atomPositions) {
      rebuildShell(atomPositions, shape, padding, preset, attenuationFactor, materialOverrides)
      lastGeoFp = geoFingerprintOf(propsRef.current)
      lastMatFp = matFingerprintOf(propsRef.current)
    }

    let raf = 0
    let lastW = 0, lastH = 0

    const tick = () => {
      raf = requestAnimationFrame(tick)
      const p = propsRef.current

      // Resize the renderer if the container changed.
      if (p.width !== lastW || p.height !== lastH) {
        renderer.setSize(p.width, p.height, false)
        camera.aspect = p.width / p.height
        camera.updateProjectionMatrix()
        lastW = p.width
        lastH = p.height
      }

      // Two-tier rebuild check. Geometry rebuilds (marching cubes / hull)
      // are expensive; material swaps are cheap.
      const geoFp = geoFingerprintOf(p)
      const matFp = matFingerprintOf(p)
      if (geoFp !== lastGeoFp) {
        if (p.atomPositions) {
          rebuildShell(
            p.atomPositions,
            p.shape,
            p.padding,
            p.preset,
            p.attenuationFactor,
            p.materialOverrides,
          )
        } else {
          disposeShell()
        }
        lastGeoFp = geoFp
        lastMatFp = matFp
      } else if (matFp !== lastMatFp) {
        swapMaterial(p.preset, p.attenuationFactor, p.materialOverrides)
        lastMatFp = matFp
      }

      // Mirror Mol* camera. Skip frame when Mol* isn't ready yet.
      const snap = p.cameraSnapshot()
      if (!snap || !shellMesh) {
        renderer.clear()
        return
      }
      camera.fov = THREE.MathUtils.radToDeg(snap.fov)
      camera.position.set(snap.position[0], snap.position[1], snap.position[2])
      camera.up.set(snap.up[0], snap.up[1], snap.up[2])
      camera.lookAt(snap.target[0], snap.target[1], snap.target[2])
      camera.updateProjectionMatrix()

      // Push the latest Mol* pixels into the backdrop texture so the gem
      // refracts the current frame (camera drag, streaming intermediates,
      // etc).
      if (backdropTexture) backdropTexture.needsUpdate = true

      renderer.render(scene, camera)
    }
    raf = requestAnimationFrame(tick)

    return () => {
      cancelAnimationFrame(raf)
      disposeShell()
      backdropTexture?.dispose()
      envRT.dispose?.()
      pmrem.dispose()
      renderer.dispose()
    }
    // Mount once and live for the lifetime of the canvas. Prop changes are
    // observed via propsRef inside the animation loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none', // pass mouse drags through to Mol*
        display: 'block',
      }}
    />
  )
}
