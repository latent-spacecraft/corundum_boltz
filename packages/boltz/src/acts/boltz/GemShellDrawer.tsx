/**
 * GemShellDrawer — live-tuning controls for the RefractiveShell overlay.
 *
 * Surfaces every knob the shell exposes (shape, preset, IOR, transmission,
 * roughness, dispersion, attenuation factor, padding) as a compact drawer at
 * the bottom of the app, matching the pattern of WebGpuDebug and LigandDrawer.
 *
 * Settings flow:
 *   GemShellDrawer ──writes──▶ useGemShellStore ──read──▶ MolViewer prop
 *                                                       ──passed──▶ RefractiveShell
 *
 * Drawer state is stored in zustand so:
 *   - the drawer doesn't have to be open for the shell to use the params
 *   - the values persist across pane re-renders
 *   - we can later add presets-on-disk via the same Vite middleware pattern
 *     used for jewelry-presets.json
 */
import { create } from 'zustand'
import type { GemPreset, ShellShape } from './RefractiveShell'

export interface GemShellParams {
  shape: ShellShape
  preset: GemPreset
  /** Multiplier on per-preset attenuation distance (0.1–5). Higher = clearer. */
  attenuationFactor: number
  /** Outward inflate of the shell in Å (0–4). */
  padding: number
  /** Override IOR. null = use preset default. */
  iorOverride: number | null
  /** Override transmission. null = preset default. */
  transmissionOverride: number | null
  /** Override roughness. null = preset default. */
  roughnessOverride: number | null
  /** Override dispersion. null = preset default. */
  dispersionOverride: number | null
}

export interface GemShellState extends GemShellParams {
  setShape: (s: ShellShape) => void
  setPreset: (p: GemPreset) => void
  setAttenuationFactor: (v: number) => void
  setPadding: (v: number) => void
  setIor: (v: number | null) => void
  setTransmission: (v: number | null) => void
  setRoughness: (v: number | null) => void
  setDispersion: (v: number | null) => void
  reset: () => void
}

const DEFAULTS: GemShellParams = {
  shape: 'smooth',
  preset: 'quartz',
  attenuationFactor: 1.0,
  padding: 1.5,
  iorOverride: null,
  transmissionOverride: null,
  roughnessOverride: null,
  dispersionOverride: null,
}

export const useGemShellStore = create<GemShellState>((set) => ({
  ...DEFAULTS,
  setShape: (shape) => set({ shape }),
  setPreset: (preset) => set({ preset }),
  setAttenuationFactor: (attenuationFactor) => set({ attenuationFactor }),
  setPadding: (padding) => set({ padding }),
  setIor: (iorOverride) => set({ iorOverride }),
  setTransmission: (transmissionOverride) => set({ transmissionOverride }),
  setRoughness: (roughnessOverride) => set({ roughnessOverride }),
  setDispersion: (dispersionOverride) => set({ dispersionOverride }),
  reset: () => set(DEFAULTS),
}))

// ─────────────────────────────────────────────────────────────────────────────
// Drawer UI

const SHAPES: { value: ShellShape; label: string; hint: string }[] = [
  { value: 'smooth', label: 'Smooth', hint: 'Polished ellipsoid — reads as quartz/glass.' },
  { value: 'faceted', label: 'Faceted', hint: 'Convex hull — crystalline cut-gem look.' },
]

const PRESETS: { value: GemPreset; label: string }[] = [
  { value: 'quartz', label: 'Quartz' },
  { value: 'diamond', label: 'Diamond' },
  { value: 'ruby', label: 'Ruby' },
  { value: 'sapphire', label: 'Sapphire' },
  { value: 'emerald', label: 'Emerald' },
]

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  override = false,
  onClearOverride,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  override?: boolean
  onClearOverride?: () => void
}) {
  return (
    <label
      className="flex flex-col gap-1"
      style={{ color: 'var(--ink-faded)' }}
    >
      <div className="flex items-baseline justify-between font-mono text-[10px] uppercase tracking-widest">
        <span>
          {label}
          {override && (
            <span style={{ color: 'var(--oxblood)' }}> · custom</span>
          )}
        </span>
        <span style={{ color: 'var(--ink)' }}>
          {value.toFixed(step < 0.1 ? 3 : step < 1 ? 2 : 1)}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="flex-1"
        />
        {override && onClearOverride && (
          <button
            type="button"
            onClick={onClearOverride}
            className="border px-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Restore preset default"
          >
            ×
          </button>
        )}
      </div>
    </label>
  )
}

export function GemShellDrawer() {
  const s = useGemShellStore()

  // Helper to convert a "null-or-number" override into a slider-friendly
  // (defaultValue, hasOverride). Sliders always read a real number; clicking
  // them implicitly enables the override.
  const iorVal = s.iorOverride ?? 1.5
  const transVal = s.transmissionOverride ?? 1.0
  const roughVal = s.roughnessOverride ?? 0.02
  const dispVal = s.dispersionOverride ?? 0.2

  return (
    <details
      className="border-t"
      style={{ borderColor: 'var(--rule)', background: 'var(--card)' }}
    >
      <summary
        className="flex cursor-pointer select-none items-center justify-between px-6 py-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        <span>Gem shell</span>
        <span>
          {s.shape} · {s.preset}
        </span>
      </summary>
      <div className="grid grid-cols-1 gap-x-6 gap-y-4 px-6 py-4 md:grid-cols-3">
        {/* ── Shape ────────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-2">
          <span
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: 'var(--ink-faded)' }}
          >
            Shape
          </span>
          <div className="flex gap-1">
            {SHAPES.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => s.setShape(opt.value)}
                title={opt.hint}
                className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
                style={{
                  borderColor:
                    s.shape === opt.value ? 'var(--oxblood)' : 'var(--rule)',
                  background:
                    s.shape === opt.value ? 'var(--paper-mottle)' : 'transparent',
                  color: 'var(--ink)',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Preset ───────────────────────────────────────────────────── */}
        <div className="flex flex-col gap-2">
          <span
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: 'var(--ink-faded)' }}
          >
            Preset
          </span>
          <div className="flex flex-wrap gap-1">
            {PRESETS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => s.setPreset(opt.value)}
                className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
                style={{
                  borderColor:
                    s.preset === opt.value ? 'var(--oxblood)' : 'var(--rule)',
                  background:
                    s.preset === opt.value ? 'var(--paper-mottle)' : 'transparent',
                  color: 'var(--ink)',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* ── Reset ────────────────────────────────────────────────────── */}
        <div className="flex flex-col justify-end">
          <button
            type="button"
            onClick={() => s.reset()}
            className="border px-2 py-1.5 font-mono text-[10px] uppercase tracking-widest"
            style={{
              borderColor: 'var(--rule)',
              color: 'var(--ink-faded)',
            }}
          >
            Restore defaults
          </button>
        </div>

        {/* ── Scene knobs ──────────────────────────────────────────────── */}
        <Slider
          label="Attenuation × radius"
          value={s.attenuationFactor}
          onChange={s.setAttenuationFactor}
          min={0.1}
          max={5}
          step={0.05}
        />
        <Slider
          label="Padding (Å)"
          value={s.padding}
          onChange={s.setPadding}
          min={0}
          max={5}
          step={0.1}
        />

        {/* ── Material overrides ───────────────────────────────────────── */}
        <Slider
          label="IOR"
          value={iorVal}
          onChange={s.setIor}
          min={1.0}
          max={2.6}
          step={0.01}
          override={s.iorOverride !== null}
          onClearOverride={() => s.setIor(null)}
        />
        <Slider
          label="Transmission"
          value={transVal}
          onChange={s.setTransmission}
          min={0}
          max={1}
          step={0.01}
          override={s.transmissionOverride !== null}
          onClearOverride={() => s.setTransmission(null)}
        />
        <Slider
          label="Roughness"
          value={roughVal}
          onChange={s.setRoughness}
          min={0}
          max={0.5}
          step={0.01}
          override={s.roughnessOverride !== null}
          onClearOverride={() => s.setRoughness(null)}
        />
        <Slider
          label="Dispersion"
          value={dispVal}
          onChange={s.setDispersion}
          min={0}
          max={2}
          step={0.01}
          override={s.dispersionOverride !== null}
          onClearOverride={() => s.setDispersion(null)}
        />
      </div>
    </details>
  )
}
