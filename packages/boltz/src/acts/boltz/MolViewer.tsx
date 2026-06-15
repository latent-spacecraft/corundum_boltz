/**
 * MolViewer — headless Mol\* embed, default register.
 *
 * Strips the alpha-era jewelry treatment (gold putty + refractive gem
 * shell). The canvas now shows Mol\*'s standard look — cartoon polymer,
 * ball-and-stick ligands, chain-id coloring — with a compact toolbar
 * exposing the controls users actually reach for:
 *
 *   - Representation : cartoon | ball+stick | surface | spacefill
 *   - Color theme    : chain-id | secondary-structure | b-factor | element
 *   - Reset camera
 *   - Screenshot     : PNG download via Mol\*'s viewport-screenshot helper
 *
 * Mol\*'s built-in left/right control panels would give us many more knobs
 * for free, but they bring in ~150 KB of UI bundle and don't match the
 * Corundum chrome at all. The 30-line toolbar covers the daily-use 80%.
 *
 * The legacy refractive-shell + liquid-glass overlays + jewelry-presets
 * file are kept on disk for now (re-importable if a future Molero-on
 * mode wants them) but no longer wired through this component.
 */
import { useEffect, useRef, useState } from 'react'

import 'molstar/build/viewer/molstar.css'

import { DefaultPluginSpec } from 'molstar/lib/mol-plugin/spec'
import { PluginContext } from 'molstar/lib/mol-plugin/context'
import { PluginCommands } from 'molstar/lib/mol-plugin/commands'
import { to_mmCIF } from 'molstar/lib/mol-model/structure/export/mmcif'

import { downloadText, mmcifToPdb } from './structureExport'

export type StructureFormat = 'pdb' | 'mmcif'

export interface StructurePayload {
  data: string
  format: StructureFormat
  id: string
}

interface Props {
  structure: StructurePayload | null
  className?: string
}

type Representation = 'cartoon' | 'ball-and-stick' | 'gaussian-surface' | 'spacefill'
type ColorTheme = 'chain-id' | 'secondary-structure' | 'uncertainty' | 'element-symbol'

const REPRESENTATION_LABEL: Record<Representation, string> = {
  cartoon: 'Cartoon',
  'ball-and-stick': 'Ball+stick',
  'gaussian-surface': 'Surface',
  spacefill: 'Spacefill',
}

const COLOR_LABEL: Record<ColorTheme, string> = {
  'chain-id': 'Chain',
  'secondary-structure': 'Secondary',
  uncertainty: 'pLDDT',
  'element-symbol': 'Element',
}

export function MolViewer({ structure, className }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const pluginRef = useRef<PluginContext | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [representation, setRepresentation] = useState<Representation>('cartoon')
  const [colorTheme, setColorTheme] = useState<ColorTheme>('chain-id')

  // Initialize Mol\* once. Strict-mode-safe via `cancelled` flag.
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

  // (Re)build the structure + representations on structure / rep / theme change.
  useEffect(() => {
    const plugin = pluginRef.current
    if (!plugin || !ready || !structure) return
    let cancelled = false
    async function load() {
      try {
        await plugin!.clear()

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

        // Polymer rep — user-chosen type + theme.
        const polymer = await plugin!.builders.structure.tryCreateComponentStatic(
          struct,
          'polymer',
        )
        if (polymer && !cancelled) {
          await plugin!.builders.structure.representation.addRepresentation(
            polymer,
            {
              type: representation,
              color: colorTheme,
            },
          )
        }

        // Ligand rep — always ball-and-stick (cartoon doesn't apply to
        // small molecules), color by element by default so cofactors
        // read as chemical structure rather than a chain-tinted blob.
        const ligand = await plugin!.builders.structure.tryCreateComponentStatic(
          struct,
          'ligand',
        )
        if (ligand && !cancelled) {
          await plugin!.builders.structure.representation.addRepresentation(
            ligand,
            {
              type: 'ball-and-stick',
              color: 'element-symbol',
            },
          )
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
  }, [structure, ready, representation, colorTheme])

  const resetCamera = () => {
    const plugin = pluginRef.current
    if (!plugin) return
    void PluginCommands.Camera.Reset(plugin, {})
  }

  /**
   * Pull a fresh mmCIF from the current viewer state. Falls back to the
   * untouched source string when it's already mmCIF — round-tripping
   * through Mol*'s exporter would needlessly re-encode and could drop the
   * pLDDT B-factor column Boltz writes directly.
   */
  const currentMmcif = (): string => {
    if (!structure) throw new Error('No structure loaded')
    if (structure.format === 'mmcif') return structure.data
    const plugin = pluginRef.current
    if (!plugin) throw new Error('Viewer not ready')
    const loaded = plugin.managers.structure.hierarchy.current.structures[0]
    const s = loaded?.cell.obj?.data
    if (!s) throw new Error('No structure loaded')
    const result = to_mmCIF(structure.id, s, false, { copyAllCategories: true })
    return typeof result === 'string' ? result : new TextDecoder().decode(result)
  }

  const saveMmcif = () => {
    try {
      const cif = currentMmcif()
      downloadText(cif, `${structure?.id ?? 'structure'}.cif`, 'chemical/x-cif')
    } catch (e) {
      console.error('[MolViewer] mmCIF export failed:', e)
    }
  }

  const savePdb = () => {
    try {
      const cif = currentMmcif()
      const pdb = mmcifToPdb(cif, structure?.id ?? 'STRUCTURE')
      downloadText(pdb, `${structure?.id ?? 'structure'}.pdb`, 'chemical/x-pdb')
    } catch (e) {
      console.error('[MolViewer] PDB export failed:', e)
    }
  }

  const screenshot = async () => {
    const plugin = pluginRef.current
    if (!plugin) return
    try {
      const helper = plugin.helpers.viewportScreenshot
      if (!helper) {
        console.warn('[MolViewer] no viewportScreenshot helper available')
        return
      }
      const uri = await helper.getImageDataUri()
      const a = document.createElement('a')
      a.href = uri
      a.download = `${structure?.id ?? 'structure'}.png`
      a.click()
    } catch (e) {
      console.error('[MolViewer] screenshot failed:', e)
    }
  }

  return (
    <div
      className={className}
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: 0,
      }}
    >
      <Toolbar
        representation={representation}
        onRepresentationChange={setRepresentation}
        colorTheme={colorTheme}
        onColorThemeChange={setColorTheme}
        onResetCamera={resetCamera}
        onScreenshot={screenshot}
        onSaveMmcif={saveMmcif}
        onSavePdb={savePdb}
        disabled={!structure}
      />
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          flex: '1 1 0',
          minHeight: 0,
          width: '100%',
          background: 'var(--background)',
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
    </div>
  )
}

/**
 * Compact controls strip above the viewport. Two segmented switchers
 * (representation, color theme) + two icon-like buttons (reset, PNG).
 * Disabled until a structure is loaded so users don't fire commands
 * into an empty scene.
 */
function Toolbar({
  representation,
  onRepresentationChange,
  colorTheme,
  onColorThemeChange,
  onResetCamera,
  onScreenshot,
  onSaveMmcif,
  onSavePdb,
  disabled,
}: {
  representation: Representation
  onRepresentationChange: (r: Representation) => void
  colorTheme: ColorTheme
  onColorThemeChange: (c: ColorTheme) => void
  onResetCamera: () => void
  onScreenshot: () => void
  onSaveMmcif: () => void
  onSavePdb: () => void
  disabled: boolean
}) {
  return (
    <div
      className="flex flex-wrap items-center gap-2 border-b px-3 py-2 text-xs"
      style={{
        borderColor: 'var(--rule)',
        background: 'var(--card)',
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <span style={{ color: 'var(--ink-faded)' }}>Representation</span>
      <Segmented
        value={representation}
        options={Object.keys(REPRESENTATION_LABEL) as Representation[]}
        label={(v) => REPRESENTATION_LABEL[v]}
        onChange={onRepresentationChange}
        disabled={disabled}
      />
      <span style={{ color: 'var(--rule)' }}>│</span>
      <span style={{ color: 'var(--ink-faded)' }}>Color</span>
      <Segmented
        value={colorTheme}
        options={Object.keys(COLOR_LABEL) as ColorTheme[]}
        label={(v) => COLOR_LABEL[v]}
        onChange={onColorThemeChange}
        disabled={disabled}
      />
      <span style={{ flex: '1 0 auto' }} />
      <button
        type="button"
        onClick={onResetCamera}
        disabled={disabled}
        className="border px-2 py-1 text-xs transition-colors"
        style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
        title="Reset camera"
      >
        Reset view
      </button>
      <button
        type="button"
        onClick={onSaveMmcif}
        disabled={disabled}
        className="border px-2 py-1 text-xs transition-colors"
        style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
        title="Download mmCIF (.cif)"
      >
        mmCIF
      </button>
      <button
        type="button"
        onClick={onSavePdb}
        disabled={disabled}
        className="border px-2 py-1 text-xs transition-colors"
        style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
        title="Download PDB (.pdb)"
      >
        PDB
      </button>
      <button
        type="button"
        onClick={onScreenshot}
        disabled={disabled}
        className="border px-2 py-1 text-xs transition-colors"
        style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
        title="Download PNG"
      >
        PNG
      </button>
    </div>
  )
}

function Segmented<T extends string>({
  value,
  options,
  label,
  onChange,
  disabled,
}: {
  value: T
  options: T[]
  label: (v: T) => string
  onChange: (v: T) => void
  disabled: boolean
}) {
  return (
    <div className="flex">
      {options.map((opt) => {
        const active = opt === value
        return (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            disabled={disabled}
            className="border px-2 py-1 text-xs"
            style={{
              borderColor: active ? 'var(--oxblood)' : 'var(--rule)',
              background: active ? 'var(--paper-mottle)' : 'transparent',
              color: active ? 'var(--ink)' : 'var(--ink-faded)',
              marginLeft: -1,
            }}
          >
            {label(opt)}
          </button>
        )
      })}
    </div>
  )
}
