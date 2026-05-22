/**
 * MoleroViewer — Molero's first React shell.
 *
 * Mounts a `<canvas>`, boots Three.js's `WebGPURenderer`, builds a scene
 * from the structure text via Molero's own parser → entity graph →
 * sphere-pass pipeline, and drives an OrbitControls + PMREM environment.
 *
 * No Mol* dependency in this path. WebGPU is required; if the browser
 * doesn't have it, we surface a clear error rather than falling back —
 * the user can flip the engine toggle back to Mol* until they upgrade.
 *
 * Per-property material channels (Phase 2) layer on by swapping the
 * pass's MeshPhysicalMaterial for a MeshPhysicalNodeMaterial with TSL
 * nodes reading `instancedAttribute('aFormalCharge')`, etc. The
 * `PropertyAttributes` table built by `buildScene` already carries
 * everything those nodes will need.
 */
import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import {
  ACESFilmicToneMapping,
  PerspectiveCamera,
  PMREMGenerator,
  Scene as ThreeScene,
  SRGBColorSpace,
  WebGPURenderer,
} from 'three/webgpu'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'

import { parseMmcif } from './parsers/mmcif'
import { parsePdb } from './parsers/pdb'
import { buildScene } from './scene/scene'
import { createSpherePass, type SpherePassResources } from './passes/sphere'
import { createStickPass, type StickPassResources } from './passes/stick'
import { computeBonds } from './chemistry/bonds'

export interface MoleroStructure {
  data: string
  format: 'pdb' | 'mmcif'
  id: string
}

export interface MoleroViewerProps {
  structure: MoleroStructure | null
  className?: string
}

interface RendererState {
  renderer: WebGPURenderer
  scene: ThreeScene
  camera: PerspectiveCamera
  controls: OrbitControls
  pmrem: PMREMGenerator
  raf: number
  disposed: boolean
  spherePass: SpherePassResources | null
  stickPass: StickPassResources | null
}

export function MoleroViewer({ structure, className }: MoleroViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const stateRef = useRef<RendererState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

  // Effect 1 — boot the renderer once on mount.
  useEffect(() => {
    let cancelled = false
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    async function init() {
      try {
        if (!('gpu' in navigator)) {
          throw new Error('WebGPU not available in this browser')
        }
        const renderer = new WebGPURenderer({
          canvas: canvas!,
          antialias: true,
          alpha: false,
        })
        renderer.toneMapping = ACESFilmicToneMapping
        renderer.toneMappingExposure = 1.1
        renderer.outputColorSpace = SRGBColorSpace
        const dpr = Math.min(window.devicePixelRatio || 1, 2)
        renderer.setPixelRatio(dpr)
        const rect = container!.getBoundingClientRect()
        renderer.setSize(rect.width, rect.height, false)
        await renderer.init()
        if (cancelled) {
          renderer.dispose()
          return
        }

        const scene = new ThreeScene()
        // PMREM-baked room environment for IBL — metals + clearcoat need
        // *something* to reflect or they read as matte plastic.
        const pmrem = new PMREMGenerator(renderer)
        const envScene = new RoomEnvironment()
        // PMREMGenerator in three/webgpu is async-via-promise on .fromScene
        // in newer versions, but the WebGPURenderer path still returns a
        // RenderTarget synchronously.
        const envRT = pmrem.fromScene(envScene, 0.04)
        scene.environment = envRT.texture

        const camera = new PerspectiveCamera(
          35,
          rect.width / Math.max(1, rect.height),
          0.5,
          5000,
        )
        camera.position.set(0, 0, 50)

        const controls = new OrbitControls(camera, canvas!)
        controls.enableDamping = true
        controls.dampingFactor = 0.08

        const s: RendererState = {
          renderer,
          scene,
          camera,
          controls,
          pmrem,
          raf: 0,
          disposed: false,
          spherePass: null,
          stickPass: null,
        }
        stateRef.current = s

        const tick = () => {
          if (s.disposed) return
          s.raf = requestAnimationFrame(tick)
          controls.update()
          renderer.renderAsync(scene, camera)
        }
        s.raf = requestAnimationFrame(tick)
        setReady(true)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    }
    void init()

    return () => {
      cancelled = true
      const s = stateRef.current
      stateRef.current = null
      setReady(false)
      if (s) {
        s.disposed = true
        cancelAnimationFrame(s.raf)
        if (s.spherePass) s.spherePass.dispose()
        if (s.stickPass) s.stickPass.dispose()
        s.controls.dispose()
        s.pmrem.dispose()
        s.renderer.dispose()
      }
    }
  }, [])

  // Effect 2 — handle resize.
  useLayoutEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => {
      const s = stateRef.current
      if (!s) return
      const r = container.getBoundingClientRect()
      const w = Math.max(1, r.width)
      const h = Math.max(1, r.height)
      s.renderer.setSize(w, h, false)
      s.camera.aspect = w / h
      s.camera.updateProjectionMatrix()
    })
    ro.observe(container)
    return () => ro.disconnect()
  }, [])

  // Effect 3 — load / replace structure.
  useEffect(() => {
    const s = stateRef.current
    if (!s || !ready) return
    if (s.spherePass) {
      s.scene.remove(s.spherePass.mesh)
      s.spherePass.dispose()
      s.spherePass = null
    }
    if (s.stickPass) {
      s.scene.remove(s.stickPass.mesh)
      s.stickPass.dispose()
      s.stickPass = null
    }
    if (!structure) return

    try {
      const t0 = performance.now()
      const parsed = structure.format === 'pdb'
        ? parsePdb(structure.data)
        : parseMmcif(structure.data)
      const built = buildScene(parsed)
      const tParse = performance.now() - t0

      const tBonds0 = performance.now()
      const bonds = computeBonds(built.attrs, built.bbox)
      const tBonds = performance.now() - tBonds0

      // Spheres slightly smaller when sticks are present — gives the
      // ball-and-stick aesthetic instead of fused vdW blobs.
      const sphere = createSpherePass(built, { scale: 0.28 })
      s.scene.add(sphere.mesh)
      s.spherePass = sphere

      const stick = createStickPass(built, bonds)
      s.scene.add(stick.mesh)
      s.stickPass = stick

      // Frame the camera around the structure.
      const [cx, cy, cz] = built.center
      const r = Math.max(built.radius, 5)
      s.controls.target.set(cx, cy, cz)
      s.camera.position.set(cx, cy, cz + r * 2.5)
      s.camera.near = Math.max(0.1, r * 0.05)
      s.camera.far = r * 50
      s.camera.updateProjectionMatrix()
      s.controls.update()

      console.log(
        '[Molero] %d atoms / %d residues / %d chains (parse %sms) / %d bonds (perceive %sms)',
        built.attrs.count,
        built.residues.length,
        built.chains.length,
        tParse.toFixed(1),
        bonds.count,
        tBonds.toFixed(1),
      )
    } catch (e) {
      console.error('[Molero] structure load failed:', e)
      setError((e as Error).message)
    }
  }, [structure, ready])

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '4 / 3',
        minHeight: 320,
        border: '1px solid var(--rule)',
        background: '#0a0a0a',
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          display: 'block',
        }}
      />
      {error && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--destructive)',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            background: 'var(--paper)',
            padding: 24,
            textAlign: 'center',
          }}
        >
          Molero: {error}
          {error.includes('WebGPU') && (
            <span style={{ display: 'block', marginTop: 8, color: 'var(--ink-faded)' }}>
              Flip the engine toggle back to Mol* to keep viewing while you upgrade.
            </span>
          )}
        </div>
      )}
    </div>
  )
}
