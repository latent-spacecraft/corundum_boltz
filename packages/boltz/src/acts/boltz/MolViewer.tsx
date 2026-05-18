/**
 * MolViewer — headless Mol* embed.
 *
 * We deliberately avoid `createPluginUI` because Mol*'s default layout
 * paints sequence / log / structure-tools panels at fixed positions that
 * escape our card. The headless `PluginContext` route gives us just the
 * 3D canvas; the field guide owns the surrounding chrome.
 *
 * Two render modes live here:
 *   - 'cartoon' — Mol*'s `default` preset (ribbon/planks/arrows by
 *     secondary structure, chain-id colors). The conventional view.
 *   - 'glass' — a putty (smooth-tube) representation with a translucent
 *     low-roughness material and B-factor-driven size & color themes.
 *     Because the mmCIF writer stuffs pLDDT × 100 into B_iso_or_equiv,
 *     the tube naturally thickens and saturates at high-confidence
 *     residues — the "gem with veins" effect, free, from data we
 *     already wrote.
 *
 * Glass-mode tuning is split across two effects so that postprocessing
 * sliders (bloom/exposure) don't trigger a full structure rebuild and
 * remain smooth under continuous drag.
 *
 * Implementation notes:
 *   - `position: absolute; inset: 0` on the canvas inside an `overflow:
 *     hidden` container so any future popups Mol* draws stay inside the
 *     card.
 *   - Strict-mode safe: we track `cancelled` so the second StrictMode
 *     mount-then-unmount cycle doesn't leave an orphan plugin.
 */
import { useEffect, useRef, useState } from 'react'

import 'molstar/build/viewer/molstar.css'

import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec'
import { PluginContext } from 'molstar/lib/mol-plugin/context'
import { Color } from 'molstar/lib/mol-util/color'

export type StructureFormat = 'pdb' | 'mmcif'
export type ViewMode = 'cartoon' | 'glass' | 'crystal' | 'vacuum'

export interface StructurePayload {
  data: string
  format: StructureFormat
  /** Identifier used for cache-busting / key tracking. */
  id: string
}

// Built-in Mol* color themes available without registering extra
// behaviors. `uncertainty` reads B-factor (= pLDDT × 100 for our writer)
// and is the gem default. The others are useful for sanity-checking the
// geometry.
export const GLASS_COLOR_THEMES = [
  'uncertainty',
  'chain-id',
  'sequence-id',
  'residue-name',
  'hydrophobicity',
  'element-symbol',
] as const
export type GlassColorTheme = (typeof GLASS_COLOR_THEMES)[number]

export interface GlassParams {
  // Geometry — putty tube only (glass mode)
  sizeFactor: number
  baseSize: number
  bfactorFactor: number
  // Surface — gaussian surface only (crystal mode)
  resolution: number
  smoothness: number
  radiusOffset: number
  // Material — both modes
  alpha: number
  emissive: number
  roughness: number
  metalness: number
  bumpiness: number
  // Color — both modes
  colorTheme: GlassColorTheme
  colorReverse: boolean
  // Postprocessing — both modes, canvas3d-only (no rebuild)
  bloomStrength: number
  bloomRadius: number
  bloomThreshold: number
  exposure: number
  backgroundColor: number
}

// Metallic coral putty — confidence-veined polished gemstone tube.
export const DEFAULT_GLASS_PARAMS: GlassParams = {
  sizeFactor: 0.7,
  baseSize: 0,
  bfactorFactor: 0.02,
  resolution: 1,
  smoothness: 1.5,
  radiusOffset: 0,
  alpha: 1,
  emissive: 0.25,
  roughness: 0.1,
  metalness: 1,
  bumpiness: 0,
  colorTheme: 'uncertainty',
  colorReverse: true,
  bloomStrength: 1.4,
  bloomRadius: 0.8,
  bloomThreshold: 0.25,
  exposure: 1.3,
  backgroundColor: 0x0a0a0a,
}

// Vacuum tube — glowing inner wire wrapped in a thin glass shell.
// The wire's emissive is global; pLDDT drives BRIGHTNESS via the
// uncertainty color theme + bloom mode 'emissive', so high-confidence
// residues read as white-hot filament and low-confidence ones stay dim.
export interface VacuumParams {
  // Inner wire — putty
  wireAlpha: number
  wireEmissive: number
  wireRoughness: number
  wireMetalness: number
  wireSizeFactor: number
  wireBaseSize: number
  wireBfactorFactor: number
  // Outer shell — gaussian surface
  shellAlpha: number
  shellEmissive: number
  shellRoughness: number
  shellMetalness: number
  shellResolution: number
  shellSmoothness: number
  shellRadiusOffset: number
  // Color — both layers share the theme; the wire does the work
  colorTheme: GlassColorTheme
  colorReverse: boolean
  // Postprocessing
  bloomStrength: number
  bloomRadius: number
  bloomThreshold: number
  exposure: number
  backgroundColor: number
}

export const DEFAULT_VACUUM_PARAMS: VacuumParams = {
  wireAlpha: 1,
  wireEmissive: 0.6,
  wireRoughness: 0.4,
  wireMetalness: 0,
  wireSizeFactor: 0.5,
  wireBaseSize: 0.1,
  wireBfactorFactor: 0.025,
  shellAlpha: 0.15,
  shellEmissive: 0,
  shellRoughness: 0.1,
  shellMetalness: 0,
  shellResolution: 0.6,
  shellSmoothness: 1,
  shellRadiusOffset: 1.5,
  colorTheme: 'uncertainty',
  colorReverse: true,
  bloomStrength: 1.8,
  bloomRadius: 1,
  bloomThreshold: 0.15,
  exposure: 1.5,
  backgroundColor: 0x050505,
}

// Translucent gaussian surface — protein-in-ice / fossil-in-amber.
// Lower alpha + slight radius puff + smoothest setting; metalness pulled
// back from the putty default so it reads as a translucent material
// rather than a metallic shell.
export const DEFAULT_CRYSTAL_PARAMS: GlassParams = {
  sizeFactor: 0.7,
  baseSize: 0,
  bfactorFactor: 0.02,
  resolution: 0.6,
  smoothness: 1,
  radiusOffset: 0.5,
  alpha: 0.3,
  emissive: 0.1,
  roughness: 0.15,
  metalness: 0.6,
  bumpiness: 0,
  colorTheme: 'uncertainty',
  colorReverse: true,
  bloomStrength: 1.2,
  bloomRadius: 0.6,
  bloomThreshold: 0.4,
  exposure: 1.4,
  backgroundColor: 0x0a0a0a,
}

interface Props {
  structure: StructurePayload | null
  /** Optional pLDDT values per residue; when provided, colours by confidence. */
  plddt?: Float32Array | number[] | null
  viewMode?: ViewMode
  glassParams?: GlassParams
  vacuumParams?: VacuumParams
  className?: string
}

// Apply the canvas3d-only slice of vacuum params.
function applyVacuumCanvas3d(plugin: PluginContext, p: VacuumParams) {
  const c3d = plugin.canvas3d
  if (!c3d) return
  c3d.setProps({
    renderer: {
      backgroundColor: Color(p.backgroundColor),
      exposure: p.exposure,
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

// Apply the canvas3d-only slice of glass params (bloom, exposure,
// background). Cheap — does not rebuild geometry. Called both from the
// load effect and from a separate postprocessing effect.
function applyGlassCanvas3d(plugin: PluginContext, p: GlassParams) {
  const c3d = plugin.canvas3d
  if (!c3d) return
  c3d.setProps({
    renderer: {
      backgroundColor: Color(p.backgroundColor),
      exposure: p.exposure,
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

function applyCartoonCanvas3d(
  plugin: PluginContext,
  defaults: { renderer: any; postprocessing: any } | null,
) {
  const c3d = plugin.canvas3d
  if (!c3d || !defaults) return
  c3d.setProps({
    renderer: {
      backgroundColor: defaults.renderer.backgroundColor,
      exposure: defaults.renderer.exposure,
    },
    postprocessing: {
      outline: defaults.postprocessing.outline,
      bloom: defaults.postprocessing.bloom,
    },
  })
}

export function MolViewer({
  structure,
  viewMode = 'cartoon',
  glassParams = DEFAULT_GLASS_PARAMS,
  vacuumParams = DEFAULT_VACUUM_PARAMS,
  className,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const pluginRef = useRef<PluginContext | null>(null)
  const defaultsRef = useRef<{ renderer: any; postprocessing: any } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)

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
        if (plugin.canvas3d) {
          const p = plugin.canvas3d.props
          defaultsRef.current = {
            renderer: { ...p.renderer },
            postprocessing: { ...p.postprocessing },
          }
        }
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

  // Effect 1 — structure + representation. Rebuilds when anything that
  // affects geometry/material/color theme changes, plus on structure or
  // viewMode change.
  //
  // The deps list pulls the rep-affecting slice of glassParams; the
  // postprocessing slice is intentionally excluded so its sliders don't
  // cause a rebuild.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !ready || !structure) return
    let cancelled = false
    async function load() {
      try {
        await plugin!.clear()
        if (viewMode === 'glass' || viewMode === 'crystal') {
          applyGlassCanvas3d(plugin!, glassParams)
        } else if (viewMode === 'vacuum') {
          applyVacuumCanvas3d(plugin!, vacuumParams)
        } else {
          applyCartoonCanvas3d(plugin!, defaultsRef.current)
        }

        const data = await plugin!.builders.data.rawData({
          data: structure!.data,
          label: structure!.id,
        })
        const trajectory = await plugin!.builders.structure.parseTrajectory(
          data,
          structure!.format === 'mmcif' ? 'mmcif' : 'pdb',
        )
        if (cancelled) return

        if (viewMode === 'cartoon') {
          await plugin!.builders.structure.hierarchy.applyPreset(
            trajectory,
            'default',
          )
        } else {
          // Manual build: model → structure → polymer component → (putty
          // for glass, gaussian-surface for crystal, both for vacuum).
          // We skip the `default` preset because it emits cartoon.
          const model = await plugin!.builders.structure.createModel(trajectory)
          if (cancelled) return
          const struct = await plugin!.builders.structure.createStructure(model)
          if (cancelled) return
          const polymer = await plugin!.builders.structure.tryCreateComponentStatic(
            struct,
            'polymer',
          )
          if (cancelled || !polymer) return

          if (viewMode === 'glass') {
            await plugin!.builders.structure.representation.addRepresentation(
              polymer,
              {
                type: 'putty',
                typeParams: {
                  alpha: glassParams.alpha,
                  emissive: glassParams.emissive,
                  quality: 'high',
                  material: {
                    metalness: glassParams.metalness,
                    roughness: glassParams.roughness,
                    bumpiness: glassParams.bumpiness,
                  },
                  sizeFactor: glassParams.sizeFactor,
                },
                color: glassParams.colorTheme,
                colorParams: { reverse: glassParams.colorReverse } as any,
                size: 'uncertainty',
                sizeParams: {
                  bfactorFactor: glassParams.bfactorFactor,
                  baseSize: glassParams.baseSize,
                } as any,
              },
              { tag: 'glass-ribbon' },
            )
          } else if (viewMode === 'crystal') {
            // crystal: gaussian-surface with the same gem treatment.
            await plugin!.builders.structure.representation.addRepresentation(
              polymer,
              {
                type: 'gaussian-surface',
                typeParams: {
                  alpha: glassParams.alpha,
                  emissive: glassParams.emissive,
                  quality: 'high',
                  material: {
                    metalness: glassParams.metalness,
                    roughness: glassParams.roughness,
                    bumpiness: glassParams.bumpiness,
                  },
                  resolution: glassParams.resolution,
                  smoothness: glassParams.smoothness,
                  radiusOffset: glassParams.radiusOffset,
                },
                color: glassParams.colorTheme,
                colorParams: { reverse: glassParams.colorReverse } as any,
              },
              { tag: 'crystal-surface' },
            )
          } else {
            // vacuum: glowing putty wire INSIDE translucent gaussian-
            // surface shell. Order matters for transparency sorting in
            // some backends, but Mol*'s WBOIT handles the layering.
            //
            // The wire's emissive + high bloom + emissive-mode bloom
            // means brightness scales with the color theme; with
            // `uncertainty` reverse=true, high-pLDDT residues read as
            // white-hot filament.
            await plugin!.builders.structure.representation.addRepresentation(
              polymer,
              {
                type: 'putty',
                typeParams: {
                  alpha: vacuumParams.wireAlpha,
                  emissive: vacuumParams.wireEmissive,
                  quality: 'high',
                  material: {
                    metalness: vacuumParams.wireMetalness,
                    roughness: vacuumParams.wireRoughness,
                    bumpiness: 0,
                  },
                  sizeFactor: vacuumParams.wireSizeFactor,
                },
                color: vacuumParams.colorTheme,
                colorParams: { reverse: vacuumParams.colorReverse } as any,
                size: 'uncertainty',
                sizeParams: {
                  bfactorFactor: vacuumParams.wireBfactorFactor,
                  baseSize: vacuumParams.wireBaseSize,
                } as any,
              },
              { tag: 'vacuum-wire' },
            )
            if (cancelled) return
            await plugin!.builders.structure.representation.addRepresentation(
              polymer,
              {
                type: 'gaussian-surface',
                typeParams: {
                  alpha: vacuumParams.shellAlpha,
                  emissive: vacuumParams.shellEmissive,
                  quality: 'high',
                  material: {
                    metalness: vacuumParams.shellMetalness,
                    roughness: vacuumParams.shellRoughness,
                    bumpiness: 0,
                  },
                  resolution: vacuumParams.shellResolution,
                  smoothness: vacuumParams.shellSmoothness,
                  radiusOffset: vacuumParams.shellRadiusOffset,
                },
                // Shell uses a neutral color; the wire owns the pLDDT
                // narrative. chain-id with a single chain produces an
                // even tint so the shell reads as glass, not as a
                // second confidence indicator.
                color: 'chain-id',
              },
              { tag: 'vacuum-shell' },
            )
          }
        }
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
    viewMode,
    // Rep-affecting params: geometry (glass), surface (crystal), material, color.
    glassParams.sizeFactor,
    glassParams.baseSize,
    glassParams.bfactorFactor,
    glassParams.resolution,
    glassParams.smoothness,
    glassParams.radiusOffset,
    glassParams.alpha,
    glassParams.emissive,
    glassParams.roughness,
    glassParams.metalness,
    glassParams.bumpiness,
    glassParams.colorTheme,
    glassParams.colorReverse,
    // Rep-affecting vacuum params (wire + shell).
    vacuumParams.wireAlpha,
    vacuumParams.wireEmissive,
    vacuumParams.wireRoughness,
    vacuumParams.wireMetalness,
    vacuumParams.wireSizeFactor,
    vacuumParams.wireBaseSize,
    vacuumParams.wireBfactorFactor,
    vacuumParams.shellAlpha,
    vacuumParams.shellEmissive,
    vacuumParams.shellRoughness,
    vacuumParams.shellMetalness,
    vacuumParams.shellResolution,
    vacuumParams.shellSmoothness,
    vacuumParams.shellRadiusOffset,
    vacuumParams.colorTheme,
    vacuumParams.colorReverse,
  ])

  // Effect 2 — canvas3d-only updates. Cheap, no rebuild. Smooth slider
  // drag for bloom/exposure/background.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !ready) return
    if (viewMode === 'glass' || viewMode === 'crystal') {
      applyGlassCanvas3d(plugin, glassParams)
    } else if (viewMode === 'vacuum') {
      applyVacuumCanvas3d(plugin, vacuumParams)
    }
  }, [
    ready,
    viewMode,
    glassParams.bloomStrength,
    glassParams.bloomRadius,
    glassParams.bloomThreshold,
    glassParams.exposure,
    glassParams.backgroundColor,
    vacuumParams.bloomStrength,
    vacuumParams.bloomRadius,
    vacuumParams.bloomThreshold,
    vacuumParams.exposure,
    vacuumParams.backgroundColor,
  ])

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        position: 'relative',
        width: '100%',
        minHeight: 380,
        height: 380,
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
