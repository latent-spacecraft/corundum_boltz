/**
 * MolViewer — headless Mol* embed, jewelry register.
 *
 * Three Mol*-rendered representations on the metal armature:
 *   - putty wire (backbone, pLDDT-thickened metal filigree)
 *   - ball-and-stick on polymer (side chains as articulated metalwork)
 *   - ball-and-stick on ligand (geometric metal centerpiece)
 *
 * The gem shell is rendered by a separate Three.js overlay
 * (`RefractiveShell`) stacked on top of the Mol* canvas, because Mol*'s
 * material model has no IOR / transmission — it can't do real refraction.
 * The overlay polls Mol*'s camera state every frame and mirrors it onto
 * its own PerspectiveCamera, so orbiting Mol* drags the gem with it.
 *
 * The full parameter bundle for each metal lives in jewelry-presets.json
 * so the user can tweak in-app and write the final look back to that file
 * via the dev-only Vite middleware (see vite.config.ts).
 *
 * Two effects on the Mol* side:
 *   - Effect 1 rebuilds reps on structure/metal/rep-affecting param changes.
 *   - Effect 2 applies cheap canvas3d-only props (bg, exposure, bloom,
 *     lighting) for smooth slider drag without geometry rebuild.
 *
 * Strict-mode safe: `cancelled` flag prevents orphan plugins on the
 * second mount-then-unmount cycle.
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'

import 'molstar/build/viewer/molstar.css'

import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec'
import { PluginContext } from 'molstar/lib/mol-plugin/context'
import { Color } from 'molstar/lib/mol-util/color'

import defaultPresets from './jewelry-presets.json'
import { extractAtomPositions } from './atomParser'
import {
  LiquidGlass,
  type CameraSnapshot,
  type LiquidGlassParams,
} from './LiquidGlass'
import { RefractiveShell, type GemPreset } from './RefractiveShell'
import { useGemShellStore } from './GemShellDrawer'
import type { GemMaterialOpts } from '@/components/gemMaterial'

// `sourceCanvas` is no longer needed — backdrop-filter reads the live
// Mol* layer beneath this overlay automatically. Keeping the import path
// stable in case we need the canvas ref for a future feature.

export type StructureFormat = 'pdb' | 'mmcif'

export interface StructurePayload {
  data: string
  format: StructureFormat
  id: string
}

export type Metal = 'gold' | 'silver' | 'copper'
export const METALS: readonly Metal[] = ['gold', 'silver', 'copper'] as const

export interface JewelryPreset {
  // Metal armature
  color: number
  metalness: number
  roughness: number
  emissive: number
  // Scene (cheap — canvas3d only)
  background: number
  exposure: number
  bloomStrength: number
  bloomRadius: number
  bloomThreshold: number
  // Studio rig — five directional lights + ambient. Positions and
  // colors are baked into STUDIO_LIGHT_LAYOUT below; only intensities
  // are per-metal so you can soften silver, push gold, etc.
  ambientIntensity: number
  keyIntensity: number
  fillIntensity: number
  rimIntensity: number
  topIntensity: number
  bounceIntensity: number
  // Wire (putty)
  wireSizeFactor: number
  wireBaseSize: number
  wireBfactorFactor: number
  // Side chains (ball-and-stick on polymer)
  sideChainSizeFactor: number
  sideChainAspectRatio: number
  sideChainBondScale: number
  // Ligand (ball-and-stick)
  ligandSizeFactor: number
  ligandAspectRatio: number
  ligandBondScale: number
  // Liquid-glass overlay — HTML element with CSS backdrop-filter clipped
  // to the convex-hull silhouette of projected atoms.
  shellBlur: number              // CSS px backdrop blur
  shellBrightness: number        // 1.0 = neutral
  shellSaturation: number        // 1.0 = neutral
  shellEnvelopePad: number       // px to inflate the hull outward
  shellSmoothIterations: number  // Chaikin smoothing passes (0 = sharp)
  /** 0 = use metal color as tint. */
  shellTintColor: number
  shellTintAmount: number        // 0-1
  shellEdgeHighlight: number     // 0-1 inner-rim alpha
  shellEdgeWidth: number         // px box-shadow inner-rim radius
}

export type JewelryPresets = Record<Metal, JewelryPreset>

export const BUNDLED_PRESETS: JewelryPresets = defaultPresets as JewelryPresets

/**
 * Studio rig — 5 directional lights arranged for jewelry-case drama. The
 * metal armature is full PBR but Mol* has no environment map (verified:
 * no IBL in mol-gl/shader), so reflections only catch the lights we put
 * in the scene. A single key light gives one hot spot and dead matte
 * everywhere else; a multi-light rig fakes a polished-display case by
 * giving every facet a highlight to catch.
 *
 * Inclination is the angle from zenith (0° = directly above, 90° =
 * horizon, 180° = directly below). Azimuth rotates around the vertical
 * axis (0° = front).
 *
 *   key        — warm hot spot, upper-front-right; main facet highlight.
 *   fill       — cool soft light, upper-back-left; opens up shadows.
 *   rim        — neutral back-top; halo edge-light that separates the
 *                piece from the background.
 *   top        — neutral overhead; sparkle on horizontal facets.
 *   bounce     — warm under-glow; subtle metal warmth from below.
 *
 * Ambient is pulled to near zero so the rig does the work — jewelry
 * cases live or die by contrast.
 */
type LightSlot = 'key' | 'fill' | 'rim' | 'top' | 'bounce'
const STUDIO_LIGHT_LAYOUT: Record<
  LightSlot,
  { inclination: number; azimuth: number; color: number }
> = {
  key:    { inclination: 38,  azimuth: 35,  color: 0xfff0d0 },
  fill:   { inclination: 62,  azimuth: 235, color: 0xcad8ff },
  rim:    { inclination: 22,  azimuth: 175, color: 0xffffff },
  top:    { inclination: 5,   azimuth: 0,   color: 0xfff5e6 },
  bounce: { inclination: 148, azimuth: 0,   color: 0xffc18a },
}
const AMBIENT_COLOR = 0xffffff

interface Props {
  structure: StructurePayload | null
  metal?: Metal
  /** Override the bundled presets (e.g. from a settings panel). */
  presets?: JewelryPresets
  /**
   * Hide the (expensive) gem shell so per-frame rebuilds only redraw the
   * cheap metal reps. Wire/sidechain/ligand tessellate in ~10-30 ms each
   * for typical sizes, so the armature condenses at near-real-time.
   * Shell crystallizes around the final structure once this flips false.
   */
  streaming?: boolean
  /**
   * Which gem-shell renderer to use on top of the Mol* canvas:
   *   - 'glass' : CSS backdrop-filter overlay (LiquidGlass). Fast, no extra
   *               WebGL context, but no real refraction.
   *   - 'gem'   : Three.js MeshPhysicalMaterial shell (RefractiveShell).
   *               Real transmission/IOR/dispersion, dedicated WebGL canvas
   *               mirroring Mol*'s camera. Slightly heavier; matches the
   *               wordmark logo's material register exactly.
   *   - 'none'  : no shell overlay; only the Mol* metal armature renders.
   * Defaults to 'gem' since we built the new register specifically for this.
   */
  shellMode?: 'glass' | 'gem' | 'none'
  /** Gem preset for the 'gem' shellMode. Ignored otherwise. */
  gemPreset?: GemPreset
  className?: string
}

function applyCanvas3d(plugin: PluginContext, p: JewelryPreset) {
  const c3d = plugin.canvas3d
  if (!c3d) return
  c3d.setProps({
    renderer: {
      backgroundColor: Color(p.background),
      exposure: p.exposure,
      ambientColor: Color(AMBIENT_COLOR),
      ambientIntensity: p.ambientIntensity,
      light: (['key', 'fill', 'rim', 'top', 'bounce'] as const).map((slot) => {
        const layout = STUDIO_LIGHT_LAYOUT[slot]
        const intensityKey =
          (slot + 'Intensity') as `${LightSlot}Intensity`
        return {
          inclination: layout.inclination,
          azimuth: layout.azimuth,
          color: Color(layout.color),
          intensity: p[intensityKey],
        }
      }),
    },
    postprocessing: {
      outline: { name: 'off', params: {} },
      bloom: {
        name: 'on',
        params: {
          strength: p.bloomStrength,
          radius: p.bloomRadius,
          threshold: p.bloomThreshold,
          mode: 'emissive',
        },
      },
    },
  })
}

/**
 * Adapter: subscribes to the gem-shell tuning store and threads the resolved
 * params into RefractiveShell. Kept as a separate component so MolViewer's
 * own re-renders aren't triggered by every slider drag — only this thin
 * wrapper re-renders, and the shell's prop diff handles the rebuild.
 */
function RefractiveShellFromStore({
  atomPositions,
  cameraSnapshot,
  backdropCanvas,
  width,
  height,
  fallbackPreset,
}: {
  atomPositions: Float32Array | null
  cameraSnapshot: () => CameraSnapshot | null
  backdropCanvas: HTMLCanvasElement | null
  width: number
  height: number
  fallbackPreset: GemPreset
}) {
  const shape = useGemShellStore((s) => s.shape)
  const presetFromStore = useGemShellStore((s) => s.preset)
  const attenuationFactor = useGemShellStore((s) => s.attenuationFactor)
  const padding = useGemShellStore((s) => s.padding)
  const ior = useGemShellStore((s) => s.iorOverride)
  const transmission = useGemShellStore((s) => s.transmissionOverride)
  const roughness = useGemShellStore((s) => s.roughnessOverride)
  const dispersion = useGemShellStore((s) => s.dispersionOverride)
  const overrides: Partial<GemMaterialOpts> = {}
  if (ior !== null) overrides.ior = ior
  if (transmission !== null) overrides.transmission = transmission
  if (roughness !== null) overrides.roughness = roughness
  if (dispersion !== null) overrides.dispersion = dispersion
  return (
    <RefractiveShell
      atomPositions={atomPositions}
      cameraSnapshot={cameraSnapshot}
      backdropCanvas={backdropCanvas}
      width={width}
      height={height}
      shape={shape}
      preset={presetFromStore ?? fallbackPreset}
      attenuationFactor={attenuationFactor}
      padding={padding}
      materialOverrides={Object.keys(overrides).length ? overrides : undefined}
    />
  )
}

export function MolViewer({
  structure,
  metal = 'gold',
  presets = BUNDLED_PRESETS,
  streaming = false,
  shellMode = 'gem',
  gemPreset = 'ruby',
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const pluginRef = useRef<PluginContext | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  // Pixel size of the overlay canvas — measured from the container via
  // ResizeObserver so the Three.js canvas always matches Mol*'s viewport.
  const [overlaySize, setOverlaySize] = useState({ width: 0, height: 0 })

  useEffect(() => {
    let cancelled = false
    async function init() {
      const container = containerRef.current
      const canvas = canvasRef.current
      if (!container || !canvas) return
      try {
        const plugin = new PluginContext(DefaultPluginSpec())
        await plugin.init()
        if (cancelled) {
          plugin.dispose()
          return
        }
        const ok = await plugin.initViewerAsync(canvas, container)
        if (!ok) throw new Error('Mol* initViewerAsync returned false')
        pluginRef.current = plugin
        setReady(true)
      } catch (e) {
        setError((e as Error).message)
      }
    }
    void init()
    return () => {
      cancelled = true
      const p = pluginRef.current
      pluginRef.current = null
      setReady(false)
      if (p) p.dispose()
    }
  }, [])

  const preset = presets[metal]

  // Effect 1 — structure + representations. Rebuilds on structure change,
  // metal change, or any representation-affecting param change. Bloom /
  // exposure / background ride on Effect 2 to keep drag smooth.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !ready || !structure) return
    let cancelled = false
    async function load() {
      try {
        await plugin!.clear()
        applyCanvas3d(plugin!, preset)

        const data = await plugin!.builders.data.rawData({
          data: structure!.data,
          label: structure!.id,
        })
        const trajectory = await plugin!.builders.structure.parseTrajectory(
          data,
          structure!.format === 'mmcif' ? 'mmcif' : 'pdb',
        )
        if (cancelled) return

        const model = await plugin!.builders.structure.createModel(trajectory)
        if (cancelled) return
        const struct = await plugin!.builders.structure.createStructure(model)
        if (cancelled) return
        const polymer = await plugin!.builders.structure.tryCreateComponentStatic(
          struct,
          'polymer',
        )
        if (cancelled || !polymer) return

        // Metal armature — putty wire. Thickness tracks pLDDT (B-factor).
        await plugin!.builders.structure.representation.addRepresentation(
          polymer,
          {
            type: 'putty',
            typeParams: {
              alpha: 1.0,
              emissive: preset.emissive,
              quality: 'high',
              material: {
                metalness: preset.metalness,
                roughness: preset.roughness,
                bumpiness: 0,
              },
              sizeFactor: preset.wireSizeFactor,
            },
            color: 'uniform',
            colorParams: { value: Color(preset.color) } as any,
            size: 'uncertainty',
            sizeParams: {
              bfactorFactor: preset.wireBfactorFactor,
              baseSize: preset.wireBaseSize,
            } as any,
          },
          { tag: 'jewel-wire' },
        )
        if (cancelled) return

        // Side chains as fine articulated metalwork (covers backbone too;
        // the putty wire dominates that visually).
        await plugin!.builders.structure.representation.addRepresentation(
          polymer,
          {
            type: 'ball-and-stick',
            typeParams: {
              alpha: 1.0,
              emissive: preset.emissive,
              quality: 'high',
              material: {
                metalness: preset.metalness,
                roughness: preset.roughness,
                bumpiness: 0,
              },
              sizeFactor: preset.sideChainSizeFactor,
              sizeAspectRatio: preset.sideChainAspectRatio,
              bondScale: preset.sideChainBondScale,
            },
            color: 'uniform',
            colorParams: { value: Color(preset.color) } as any,
          },
          { tag: 'jewel-side-chains' },
        )
        if (cancelled) return

        // Ligands — slightly heavier so cofactors read as centerpiece.
        const ligand = await plugin!.builders.structure.tryCreateComponentStatic(
          struct,
          'ligand',
        )
        if (cancelled) return
        if (ligand) {
          await plugin!.builders.structure.representation.addRepresentation(
            ligand,
            {
              type: 'ball-and-stick',
              typeParams: {
                alpha: 1.0,
                emissive: preset.emissive,
                quality: 'high',
                material: {
                  metalness: preset.metalness,
                  roughness: preset.roughness,
                  bumpiness: 0,
                },
                sizeFactor: preset.ligandSizeFactor,
                sizeAspectRatio: preset.ligandAspectRatio,
                bondScale: preset.ligandBondScale,
              },
              color: 'uniform',
              colorParams: { value: Color(preset.color) } as any,
            },
            { tag: 'jewel-ligand' },
          )
          if (cancelled) return
        }

        // Gem shell is handled by the RefractiveShell overlay below — Mol*
        // has no IOR / transmission so it can't render real refraction.
      } catch (e) {
        console.error('[MolViewer] structure load failed:', e)
        if (e instanceof Error) console.error('[MolViewer] stack:', e.stack)
        if (!cancelled) setError((e as Error).message)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [
    structure,
    ready,
    metal,
    streaming,
    preset.color,
    preset.metalness,
    preset.roughness,
    preset.emissive,
    preset.wireSizeFactor,
    preset.wireBaseSize,
    preset.wireBfactorFactor,
    preset.sideChainSizeFactor,
    preset.sideChainAspectRatio,
    preset.sideChainBondScale,
    preset.ligandSizeFactor,
    preset.ligandAspectRatio,
    preset.ligandBondScale,
  ])

  // Effect 2 — cheap canvas3d updates. Smooth slider drag.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !ready) return
    applyCanvas3d(plugin, preset)
  }, [
    ready,
    preset.background,
    preset.exposure,
    preset.bloomStrength,
    preset.bloomRadius,
    preset.bloomThreshold,
    preset.ambientIntensity,
    preset.keyIntensity,
    preset.fillIntensity,
    preset.rimIntensity,
    preset.topIntensity,
    preset.bounceIntensity,
  ])

  // Track container size so the Three.js overlay canvas matches the Mol*
  // canvas pixel-for-pixel. ResizeObserver fires on parent layout changes
  // (window resize, panel collapse, drawer open) without polling.
  useLayoutEffect(() => {
    const container = containerRef.current
    if (!container) return
    const measure = () => {
      const r = container.getBoundingClientRect()
      setOverlaySize({ width: Math.max(1, r.width), height: Math.max(1, r.height) })
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(container)
    return () => ro.disconnect()
  }, [])

  // Atom positions for the refractive shell — parsed once per structure
  // change. ~50-200μs for typical sizes; memoization avoids re-parsing
  // when only material/lighting sliders move.
  const atomPositions = useMemo(() => {
    if (!structure) return null
    return extractAtomPositions(structure.data, structure.format)
  }, [structure])

  // Camera snapshot getter — stable reference, called by the overlay's
  // animation loop each frame. Reads Mol*'s live camera state.
  const cameraSnapshot = useCallback((): CameraSnapshot | null => {
    const cam = pluginRef.current?.canvas3d?.camera
    if (!cam) return null
    const s = cam.state
    return {
      fov: s.fov,
      position: [s.position[0], s.position[1], s.position[2]],
      up: [s.up[0], s.up[1], s.up[2]],
      target: [s.target[0], s.target[1], s.target[2]],
    }
  }, [])

  const glassParams: LiquidGlassParams = useMemo(
    () => ({
      blur: preset.shellBlur,
      brightness: preset.shellBrightness,
      saturation: preset.shellSaturation,
      envelopePad: preset.shellEnvelopePad,
      smoothIterations: preset.shellSmoothIterations,
      // 0 is the sentinel for "tint by metal color" so the silhouette
      // always reads as the current metal unless overridden explicitly.
      tintColor: preset.shellTintColor || preset.color,
      tintAmount: preset.shellTintAmount,
      edgeHighlight: preset.shellEdgeHighlight,
      edgeWidth: preset.shellEdgeWidth,
    }),
    [
      preset.shellBlur,
      preset.shellBrightness,
      preset.shellSaturation,
      preset.shellEnvelopePad,
      preset.shellSmoothIterations,
      preset.shellTintColor,
      preset.shellTintAmount,
      preset.shellEdgeHighlight,
      preset.shellEdgeWidth,
      preset.color,
    ],
  )

  // Hide the glass during streaming so the metal armature condensation
  // shows clean. Reappears when `streaming` flips false on the last frame.
  const shellAtoms = streaming ? null : atomPositions

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
        background: 'var(--paper-mottle)',
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
      {overlaySize.width > 0 && overlaySize.height > 0 && shellMode === 'glass' && (
        <LiquidGlass
          atomPositions={shellAtoms}
          cameraSnapshot={cameraSnapshot}
          params={glassParams}
          width={overlaySize.width}
          height={overlaySize.height}
        />
      )}
      {overlaySize.width > 0 && overlaySize.height > 0 && shellMode === 'gem' && (
        <RefractiveShellFromStore
          atomPositions={shellAtoms}
          cameraSnapshot={cameraSnapshot}
          backdropCanvas={canvasRef.current}
          width={overlaySize.width}
          height={overlaySize.height}
          fallbackPreset={gemPreset}
        />
      )}
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
          }}
        >
          Viewer: {error}
        </div>
      )}
    </div>
  )
}
