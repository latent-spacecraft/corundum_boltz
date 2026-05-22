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
import { createRibbonPass, type RibbonPassResources } from './passes/ribbon'
import {
  buildGaussianSurface,
  type GaussianSurfaceResources,
} from './passes/gaussian-surface'
import { createGlassPass, type GlassPassResources } from './passes/glass'
import { computeBonds } from './chemistry/bonds'
import { BUNDLED_GLASS_PRESET, splitPreset, type GlassPreset } from './glass-preset'
import { AtomFlag } from './scene/scene'

export interface MoleroStructure {
  data: string
  format: 'pdb' | 'mmcif'
  id: string
}

/**
 * Which render passes Molero mounts.
 *   - 'ball-stick': every atom + every bond. Full atomic detail; the
 *                   inspection view.
 *   - 'cartoon':    ribbon (tube) + sidechain ball-and-stick. Backbone
 *                   atoms and backbone-only bonds are hidden — the
 *                   ribbon already carries that signal.
 *   - 'glass':      SASA-modulated refractive shell only.
 *   - 'all':        cartoon + glass (composited via Three.js transmission).
 */
export type MoleroRepresentation = 'ball-stick' | 'cartoon' | 'glass' | 'all'

export interface MoleroViewerProps {
  structure: MoleroStructure | null
  representation?: MoleroRepresentation
  /** Override the bundled glass preset (from a tuning panel etc.). Only
   *  consumed when representation includes the glass pass. */
  glassPreset?: GlassPreset
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
  ribbonPass: RibbonPassResources | null
  surface: GaussianSurfaceResources | null
  glassPass: GlassPassResources | null
}

export function MoleroViewer({
  structure,
  representation = 'cartoon',
  glassPreset = BUNDLED_GLASS_PRESET,
  className,
}: MoleroViewerProps) {
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
          ribbonPass: null,
          surface: null,
          glassPass: null,
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
        if (s.ribbonPass) s.ribbonPass.dispose()
        if (s.glassPass) s.glassPass.dispose()
        if (s.surface) s.surface.dispose()
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

  // Effect 3 — load / replace structure / swap representation.
  useEffect(() => {
    const s = stateRef.current
    if (!s || !ready) return
    // Tear down current passes.
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
    if (s.ribbonPass) {
      s.scene.remove(s.ribbonPass.group)
      s.ribbonPass.dispose()
      s.ribbonPass = null
    }
    if (s.glassPass) {
      s.scene.remove(s.glassPass.mesh)
      s.glassPass.dispose()
      s.glassPass = null
    }
    if (s.surface) {
      s.surface.dispose()
      s.surface = null
    }
    if (!structure) return

    try {
      const t0 = performance.now()
      const parsed = structure.format === 'pdb'
        ? parsePdb(structure.data)
        : parseMmcif(structure.data)
      const built = buildScene(parsed)
      const tParse = performance.now() - t0

      // ball-stick: full atomic detail (every atom + every bond).
      // cartoon:    ribbon + ball-and-stick on sidechain atoms only.
      // glass:      gem shell only.
      // all:        cartoon + glass (composited via transmission).
      const wantSpheresOrSticks =
        representation === 'ball-stick' ||
        representation === 'cartoon' ||
        representation === 'all'
      const wantCartoon = representation === 'cartoon' || representation === 'all'
      const wantGlass = representation === 'glass' || representation === 'all'
      const hideBackbone = wantCartoon // cartoon hides backbone atoms / backbone-only bonds

      let bondCount = 0
      let tBonds = 0
      if (wantSpheresOrSticks) {
        const tBonds0 = performance.now()
        const bonds = computeBonds(built.attrs, built.bbox)
        tBonds = performance.now() - tBonds0
        bondCount = bonds.count

        const flagsArr = built.attrs.flags
        const isBackbone = (i: number) => (flagsArr[i] & AtomFlag.Backbone) !== 0
        // Cartoon: hide pure-backbone atoms (N, Cα, C, O — the ribbon
        // already carries those). Sidechain atoms render.
        const atomFilter = hideBackbone
          ? (i: number) => !isBackbone(i)
          : undefined
        // Cartoon: drop bonds where both endpoints are backbone (Cα-C,
        // C-N, etc — also covered by the ribbon). Keep bonds where at
        // least one end is sidechain (Cα-Cβ et al — the attachment).
        const bondFilter = hideBackbone
          ? (a: number, b: number) => !(isBackbone(a) && isBackbone(b))
          : undefined

        // Spheres shrink when sticks are present (ball-and-stick look).
        const sphere = createSpherePass(built, { scale: 0.28, atomFilter })
        s.scene.add(sphere.mesh)
        s.spherePass = sphere

        const stick = createStickPass(built, bonds, { bondFilter })
        s.scene.add(stick.mesh)
        s.stickPass = stick
      }

      if (wantCartoon) {
        const ribbon = createRibbonPass(built)
        s.scene.add(ribbon.group)
        s.ribbonPass = ribbon
      }

      let tGlass = 0
      let surfaceVerts = 0
      if (wantGlass) {
        const tG0 = performance.now()
        const { surface: surfaceOpts, material: materialOpts } = splitPreset(glassPreset)
        const surf = buildGaussianSurface(built, surfaceOpts)
        s.surface = surf
        surfaceVerts = surf.vertexCount
        const glass = createGlassPass(surf.geometry, materialOpts)
        s.scene.add(glass.mesh)
        s.glassPass = glass
        tGlass = performance.now() - tG0
      }

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
        '[Molero] %d atoms / %d residues / %d chains (parse %sms) / %d bonds (%sms) / %s%s / rep=%s',
        built.attrs.count,
        built.residues.length,
        built.chains.length,
        tParse.toFixed(1),
        bondCount,
        tBonds.toFixed(1),
        wantGlass ? `${surfaceVerts} surf verts (${tGlass.toFixed(0)}ms)` : 'no glass',
        '',
        representation,
      )
    } catch (e) {
      console.error('[Molero] structure load failed:', e)
      setError((e as Error).message)
    }
  }, [structure, ready, representation, glassPreset])

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
