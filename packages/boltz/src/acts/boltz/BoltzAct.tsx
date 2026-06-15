/**
 * Boltz act content.
 *
 * Stage slots:
 *   Input   : FASTA textarea + precision picker + "Load engine" trio (with
 *             three stacked progress bars, one per graph) + disabled
 *             "Predict structure" button until the orchestration loop ships
 *             + live "Load .pdb / .mmCIF" file picker + "Load example".
 *   Canvas  : the MolViewer.
 *   Output  : structure header, atom/residue counts, source provenance.
 *
 * Slice-1 scope: the three Boltz-2 ONNX graphs (trunk, diffusion_step,
 * confidence) can be loaded into the engine worker. Once all three are
 * ready, the act surfaces "engine warm — awaiting orchestration logic".
 * Actual prediction is a later slice (Slice 2 = feats pipeline,
 * Slice 3 = orchestration loops).
 */
import { create } from 'zustand'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { MolViewer, type StructurePayload } from './MolViewer'
import { useModelSession } from '@/hooks/useModelSession'
import { formatBytes } from '@/engine/fetcher'
import { detectDevice, type DeviceCapabilities } from '@/engine/device'
import { estimateMemory, type MemoryEstimate } from '@/engine/memory'
import {
  boltzBundle,
  bundleApproxBytes,
  DEFAULT_PRECISION,
  PRECISIONS,
  PRECISION_LABEL,
  type BoltzPrecision,
} from './models'
import { predict, type ProgressEvent } from './orchestrate'
import { writeMmcif } from './mmcif'
import { validateAgainstGolden, type ValidationReport } from './featurizer/validate'
import { featurizeChains, parseFasta, type ParsedChain } from './featurizer'
import { loadLigandBlob, type LigandBlob } from './featurizer/ligand'
import { useLigandInsertSlot, useLigandDrawer } from './LigandDrawer'

interface BoltzActState {
  structure: StructurePayload | null
  source: string
  error: string | null
  fasta: string
  /**
   * True while a diffusion sampling run is streaming intermediate frames
   * into `structure`. The canvas uses this to drop the expensive gem
   * shell during streaming (only the cheap metal armature reps redraw),
   * then restores the full jewelry treatment on the final frame.
   */
  streaming: boolean
  /** Live memory-pressure estimate, published by BoltzInput, read by StatusBar. */
  memoryEstimate: MemoryEstimate | null
  /** Device capabilities snapshot, published once by BoltzInput. */
  device: DeviceCapabilities | null
  /** Currently-selected precision, mirrored from BoltzInput so the status
   *  bar can label "fp32 engine warm" without a prop drill. */
  precision: BoltzPrecision
  setStructure: (payload: StructurePayload | null, source: string) => void
  setStreamingFrame: (payload: StructurePayload) => void
  setStreaming: (s: boolean) => void
  setError: (e: string | null) => void
  setFasta: (text: string) => void
  setMemoryEstimate: (m: MemoryEstimate | null) => void
  setDevice: (d: DeviceCapabilities | null) => void
  setPrecisionMirror: (p: BoltzPrecision) => void
}

export const useBoltz = create<BoltzActState>((set) => ({
  structure: null,
  source: '',
  error: null,
  fasta: '',
  streaming: false,
  memoryEstimate: null,
  device: null,
  precision: DEFAULT_PRECISION,
  setStructure: (payload, source) =>
    set({ structure: payload, source, error: null, streaming: false }),
  // Streaming frames replace structure but keep the source line stable so
  // the output pane doesn't churn the provenance label every frame.
  setStreamingFrame: (payload) => set({ structure: payload, error: null }),
  setStreaming: (s) => set({ streaming: s }),
  setError: (e) => set({ error: e }),
  setFasta: (text) => set({ fasta: text }),
  setMemoryEstimate: (m) => set({ memoryEstimate: m }),
  setDevice: (d) => set({ device: d }),
  setPrecisionMirror: (p) => set({ precision: p }),
}))

function detectFormat(name: string, content: string): 'pdb' | 'mmcif' {
  const lower = name.toLowerCase()
  if (lower.endsWith('.cif') || lower.endsWith('.mmcif') || lower.endsWith('.cif.gz')) {
    return 'mmcif'
  }
  if (lower.endsWith('.pdb') || lower.endsWith('.ent')) return 'pdb'
  const head = content.slice(0, 200).trim()
  if (head.startsWith('data_') || head.includes('_entry.id')) return 'mmcif'
  return 'pdb'
}

// ─────────────────────────────────────────────────────────────────────────────
// Bundled example — Crambin (PDB 1CRN), 46-residue plant protein.

// ─────────────────────────────────────────────────────────────────────────────
// Predict-pipeline helpers

function phaseLabel(e: ProgressEvent | null): string {
  if (!e) return 'Initialising…'
  if (e.phase === 'recycling') return `Recycling ${(e.step ?? 0) + 1}/${e.total ?? 1}`
  if (e.phase === 'sampling') return `Folding ${e.step}/${e.total}`
  if (e.phase === 'confidence') return 'Scoring confidence…'
  return 'Done'
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny presentational helpers used across the input pane sections.

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <label
      className="text-xs font-medium"
      style={{ color: 'var(--ink)' }}
    >
      {children}
    </label>
  )
}

function ChipButton({
  onClick,
  title,
  children,
}: {
  onClick: () => void
  title?: string
  children: React.ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="border px-2 py-1 text-xs transition-colors"
      style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
    >
      {children}
    </button>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequences input — FASTA textarea, chain preview, file open.

function SequencesSection({
  fasta,
  setFasta,
  parseError,
  polymerChainCount,
  ligandChainCount,
  polymerLen,
  tooShort,
  tooLong,
  onLoadFile,
}: {
  fasta: string
  setFasta: (s: string) => void
  parseError: string | null
  polymerChainCount: number
  ligandChainCount: number
  polymerLen: number
  tooShort: boolean
  tooLong: boolean
  onLoadFile: (payload: StructurePayload, source: string) => void
}) {
  const totalChains = polymerChainCount + ligandChainCount
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Sequences</SectionLabel>
      <textarea
        value={fasta}
        onChange={(e) => setFasta(e.target.value)}
        rows={5}
        spellCheck={false}
        placeholder={'>my_protein\nMKLLI…'}
        className="w-full resize-y border p-2 font-mono text-xs leading-relaxed"
        style={{
          borderColor: 'var(--rule)',
          background: 'var(--background)',
          color: 'var(--ink)',
        }}
      />
      <p
        className="text-xs leading-snug"
        style={{ color: 'var(--ink-faded)' }}
      >
        {parseError ? (
          <span style={{ color: 'var(--destructive)' }}>{parseError}</span>
        ) : totalChains === 0 ? (
          <span>Paste FASTA, pick an example, or open a structure file.</span>
        ) : (
          <>
            <span style={{ color: 'var(--ink)' }}>
              {totalChains} chain{totalChains !== 1 ? 's' : ''}
            </span>
            {polymerLen > 0 && <> · {polymerLen} residue{polymerLen !== 1 ? 's' : ''}</>}
            {ligandChainCount > 0 && <> · {ligandChainCount} ligand{ligandChainCount !== 1 ? 's' : ''}</>}
            {tooShort && (
              <span style={{ color: 'var(--destructive)' }}> · under 8 minimum</span>
            )}
            {tooLong && (
              <span style={{ color: 'var(--destructive)' }}> · over 1024 maximum</span>
            )}
          </>
        )}
      </p>
      <label
        className="cursor-pointer border px-3 py-1.5 text-center text-xs transition-colors"
        style={{ borderColor: 'var(--oxblood)', color: 'var(--oxblood)' }}
      >
        Open .pdb / .mmCIF file…
        <input
          type="file"
          accept=".pdb,.ent,.cif,.mmcif"
          className="hidden"
          onChange={async (e) => {
            const file = e.target.files?.[0]
            if (!file) return
            const data = await file.text()
            const format = detectFormat(file.name, data)
            onLoadFile({ data, format, id: file.name }, `local: ${file.name}`)
            e.target.value = ''
          }}
        />
      </label>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Ligands input — text input + chip list, with browse link to the modal.

function LigandsSection({
  fasta,
  setFasta,
  ligandChains,
}: {
  fasta: string
  setFasta: (s: string) => void
  ligandChains: ParsedChain[]
}) {
  const [draft, setDraft] = useState('')
  const setLigandInsert = useLigandInsertSlot((s) => s.setInsert)
  const openDrawer = useLigandDrawer((s) => s.setOpen)

  const addLigand = useCallback(
    (raw: string) => {
      const code = raw.trim().toUpperCase()
      if (!code) return
      // De-dup: skip if the same ligand chain is already in the input.
      const re = new RegExp(`^>\\S+\\s+ligand\\s*\\r?\\n${code}\\b`, 'mi')
      if (re.test(fasta)) return
      const chunk = `>lig_${code} ligand\n${code}`
      const next = fasta.trim() ? `${fasta.trim()}\n${chunk}\n` : `${chunk}\n`
      setFasta(next)
    },
    [fasta, setFasta],
  )

  // Re-register the drawer's pick-a-ligand slot on every fasta change so it
  // always closes over the latest value (avoids stale-closure dedup checks).
  useEffect(() => {
    setLigandInsert((ccd) => addLigand(ccd))
    return () => setLigandInsert(null)
  }, [addLigand, setLigandInsert])

  const removeLigand = (chain: ParsedChain) => {
    const code = chain.sequence.toUpperCase()
    const name = chain.name || `lig_${code}`
    // Remove the `>name ligand\nCODE\n` block. Tolerate optional extra blank lines.
    const re = new RegExp(`>${name}\\s+ligand\\s*\\r?\\n${code}\\s*\\r?\\n?`, 'i')
    setFasta(fasta.replace(re, '').replace(/\n{3,}/g, '\n\n'))
  }

  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Ligands</SectionLabel>
      <div className="flex gap-1">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') {
              e.preventDefault()
              addLigand(draft)
              setDraft('')
            }
          }}
          placeholder="CCD code (e.g. HEM, ATP, ZN)"
          spellCheck={false}
          className="flex-1 border px-2 py-1 font-mono text-xs uppercase tracking-wide"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--background)',
            color: 'var(--ink)',
          }}
        />
        <ChipButton
          onClick={() => {
            addLigand(draft)
            setDraft('')
          }}
        >
          + Add
        </ChipButton>
      </div>
      {ligandChains.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {ligandChains.map((c, i) => (
            <span
              key={`${c.name}-${i}`}
              className="flex items-center gap-1.5 border px-2 py-0.5 text-xs"
              style={{ borderColor: 'var(--rule)', color: 'var(--ink)' }}
            >
              <span className="font-mono uppercase tracking-wide">
                {c.sequence.toUpperCase()}
              </span>
              <button
                type="button"
                onClick={() => removeLigand(c)}
                aria-label={`Remove ${c.sequence}`}
                title="Remove"
                style={{ color: 'var(--ink-faded)', fontSize: 14, lineHeight: 1 }}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <button
        type="button"
        onClick={() => openDrawer(true)}
        className="self-start text-xs underline underline-offset-2 transition-colors"
        style={{ color: 'var(--oxblood)' }}
      >
        Browse cofactor library →
      </button>
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Model selection — precision dropdown with autodetect recommendation.

function ModelSection({
  precision,
  setPrecision,
  device,
}: {
  precision: BoltzPrecision
  setPrecision: (p: BoltzPrecision) => void
  device: DeviceCapabilities | null
}) {
  return (
    <section className="flex flex-col gap-2">
      <SectionLabel>Model</SectionLabel>
      <select
        value={precision}
        onChange={(e) => setPrecision(e.target.value as BoltzPrecision)}
        className="w-full border px-2 py-1.5 font-mono text-xs"
        style={{
          borderColor: 'var(--rule)',
          background: 'var(--background)',
          color: 'var(--ink)',
        }}
      >
        {PRECISIONS.map((p) => (
          <option key={p} value={p}>
            {PRECISION_LABEL[p]}
            {device?.recommendedPrecision === p ? '  ★ recommended' : ''}
          </option>
        ))}
      </select>
      {device && (
        <p
          className="text-xs leading-snug"
          style={{ color: 'var(--ink-faded)' }}
        >
          {device.reason}
        </p>
      )}
    </section>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Run rail — single button that auto-loads engine then predicts. Keyed by
// precision so a precision change cleanly disposes the previous trio.

function RunRail({
  precision,
  chains,
  parseError,
  polymerLen,
  tooShort,
  tooLong,
}: {
  precision: BoltzPrecision
  chains: ParsedChain[]
  parseError: string | null
  polymerLen: number
  tooShort: boolean
  tooLong: boolean
}) {
  const bundle = useMemo(() => boltzBundle(precision), [precision])
  const trunk = useModelSession(bundle.trunk)
  const diffusion = useModelSession(bundle.diffusion_step)
  const confidence = useModelSession(bundle.confidence)
  const { setStructure, setStreamingFrame, setStreaming, setError, error } = useBoltz()

  const allReady =
    trunk.status === 'ready' && diffusion.status === 'ready' && confidence.status === 'ready'
  const anyLoading =
    trunk.status === 'fetching' ||
    trunk.status === 'compiling' ||
    diffusion.status === 'fetching' ||
    diffusion.status === 'compiling' ||
    confidence.status === 'fetching' ||
    confidence.status === 'compiling'
  const anyError =
    trunk.status === 'error' || diffusion.status === 'error' || confidence.status === 'error'

  const bundleBytes = useMemo(() => bundleApproxBytes(precision), [precision])
  const loadedBytes =
    (trunk.progress?.bytesLoaded ?? 0) +
    (diffusion.progress?.bytesLoaded ?? 0) +
    (confidence.progress?.bytesLoaded ?? 0)
  const loadPct = bundleBytes > 0 ? Math.min(100, (loadedBytes / bundleBytes) * 100) : 0

  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [autoRun, setAutoRun] = useState(false)

  const inputValid = !parseError && !tooShort && !tooLong && polymerLen > 0

  const doPredict = useCallback(async () => {
    if (!trunk.handle || !diffusion.handle || !confidence.handle) return
    setRunning(true)
    setError(null)
    setProgress(null)
    try {
      const withBlobs = await Promise.all(
        chains.map(async (c) => {
          if (c.type !== 'ligand') return c
          const blob: LigandBlob = await loadLigandBlob(c.sequence)
          return { ...c, blob }
        }),
      )
      const feats = featurizeChains(withBlobs)

      setStreaming(true)
      const labelBase = withBlobs[0]?.name ? withBlobs[0].name.slice(0, 24) : 'unnamed'
      const label =
        withBlobs.length > 1 ? `${labelBase}+${withBlobs.length - 1}` : labelBase
      const placeholderPlddt = new Float32Array(feats.N).fill(50)
      const STREAM_EVERY = 3
      let inFlight = false

      const result = await predict({
        feats,
        trunk: trunk.handle,
        diffusion: diffusion.handle,
        confidence: confidence.handle,
        recyclingSteps: 1,
        samplingSteps: 50,
        seed: 42,
        onProgress: (e) => {
          setProgress(e)
        },
        onStep: (denoised, step, total) => {
          if (step % STREAM_EVERY !== 0 && step !== total) return
          if (inFlight) return
          inFlight = true
          try {
            const cifFrame = writeMmcif({
              feats,
              atomCoords: denoised,
              plddt: placeholderPlddt,
              chains: withBlobs,
              modelId: `step-${step}`,
            })
            setStreamingFrame({
              data: cifFrame,
              format: 'mmcif',
              id: `${label} (diffusion ${step}/${total})`,
            })
          } finally {
            inFlight = false
          }
        },
      })
      const cif = writeMmcif({
        feats,
        atomCoords: result.atomCoords,
        plddt: result.plddt,
        chains: withBlobs,
        modelId: 'predicted',
      })
      setStructure(
        { data: cif, format: 'mmcif', id: `${label} (predicted)` },
        `Boltz-2 · ${chains.length} chain${chains.length > 1 ? 's' : ''} · ${(result.elapsedMs / 1000).toFixed(1)} s`,
      )
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setRunning(false)
      setStreaming(false)
    }
  }, [
    chains,
    trunk.handle,
    diffusion.handle,
    confidence.handle,
    setError,
    setStreaming,
    setStreamingFrame,
    setStructure,
  ])

  // Auto-run: when engine finishes loading after the user clicked Run,
  // kick the predict immediately so the click is a single user action.
  useEffect(() => {
    if (autoRun && allReady && !running) {
      setAutoRun(false)
      void doPredict()
    }
  }, [autoRun, allReady, running, doPredict])

  const onClick = () => {
    if (running || anyLoading) return
    if (!allReady) {
      setAutoRun(true)
      void trunk.load()
      void diffusion.load()
      void confidence.load()
      return
    }
    void doPredict()
  }

  let label: string
  let disabled = false
  if (running) {
    label = phaseLabel(progress)
    disabled = true
  } else if (anyLoading) {
    label = `Loading ${formatBytes(loadedBytes)} / ${formatBytes(bundleBytes)}`
    disabled = true
  } else if (autoRun) {
    label = 'Preparing…'
    disabled = true
  } else if (allReady) {
    label = inputValid ? 'Run' : 'Run'
    disabled = !inputValid
  } else {
    label = inputValid ? `Run · download ${formatBytes(bundleBytes)} first` : 'Run'
    disabled = !inputValid
  }

  return (
    <div className="flex flex-col gap-1.5 px-5 py-3">
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className="w-full px-3 py-2.5 text-sm font-medium transition-colors"
        style={{
          background: disabled ? 'var(--muted)' : 'var(--oxblood)',
          color: disabled ? 'var(--ink-faded)' : 'var(--primary-foreground)',
          cursor: disabled ? 'not-allowed' : 'pointer',
          border: '1px solid',
          borderColor: disabled ? 'var(--rule)' : 'var(--oxblood)',
        }}
      >
        {label}
      </button>
      {anyLoading && (
        <div
          className="h-1 w-full overflow-hidden"
          style={{ background: 'var(--rule)' }}
        >
          <div
            className="h-full transition-[width]"
            style={{ width: `${loadPct}%`, background: 'var(--oxblood)' }}
          />
        </div>
      )}
      {(anyError || error) && (
        <p
          className="text-xs leading-snug"
          style={{ color: 'var(--destructive)' }}
        >
          {trunk.error ?? diffusion.error ?? confidence.error ?? error}
        </p>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Sysdata tray — bottom strip: device tier · memory pressure bar.

function tierLabel(tier: DeviceCapabilities['tier']): string {
  if (tier === 'desktop-discrete') return 'Desktop (discrete GPU)'
  if (tier === 'desktop-integrated') return 'Desktop (integrated GPU)'
  return 'Mobile'
}

function SysdataTray({ device }: { device: DeviceCapabilities | null }) {
  const memoryEstimate = useBoltz((s) => s.memoryEstimate)
  const palette = memoryEstimate
    ? ({
        idle: { fill: 'var(--ink-faded)', label: 'idle' },
        green: { fill: '#3a7d4b', label: 'ok' },
        yellow: { fill: '#b58a1e', label: 'tight' },
        red: { fill: 'var(--destructive)', label: 'risk' },
      } as const)[memoryEstimate.level]
    : null
  const pct = memoryEstimate ? Math.min(memoryEstimate.pressureRatio, 1.5) / 1.5 : 0

  return (
    <div
      className="flex flex-col gap-1 border-t px-5 py-2 text-[11px] leading-snug"
      style={{
        borderColor: 'var(--rule)',
        color: 'var(--ink-faded)',
        background: 'var(--card)',
      }}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate" title={device?.reason ?? 'Detecting device…'}>
          {device
            ? `${tierLabel(device.tier)} · ${device.webgpu ? 'WebGPU' : 'WASM'}`
            : 'Detecting device…'}
        </span>
        {memoryEstimate && memoryEstimate.level !== 'idle' && (
          <span className="shrink-0">
            Memory {palette?.label} · {formatBytes(memoryEstimate.totalBytes)} / {formatBytes(memoryEstimate.availableBytes)}
          </span>
        )}
      </div>
      <div
        className="relative h-1 w-full overflow-hidden"
        style={{ background: 'var(--rule)' }}
        title={memoryEstimate?.reason}
      >
        {memoryEstimate && memoryEstimate.level !== 'idle' && (
          <div
            className="h-full transition-[width]"
            style={{ width: `${pct * 100}%`, background: palette?.fill }}
          />
        )}
        <div
          className="absolute top-0 h-full w-px"
          style={{
            left: `${(0.9 / 1.5) * 100}%`,
            background: 'var(--ink-faded)',
            opacity: 0.5,
          }}
          aria-hidden
        />
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// BoltzInput — composes the left-pane sections + sticky run + sysdata tray.

export function BoltzInput() {
  const {
    fasta,
    setFasta,
    setStructure,
    setMemoryEstimate,
    setDevice: setSharedDevice,
    setPrecisionMirror,
  } = useBoltz()
  const [precision, setPrecisionLocal] = useState<BoltzPrecision>(DEFAULT_PRECISION)
  const setPrecision = (p: BoltzPrecision) => {
    setPrecisionLocal(p)
    setPrecisionMirror(p)
  }
  const [device, setDevice] = useState<DeviceCapabilities | null>(null)
  const userPickedPrecision = useRef(false)

  useEffect(() => {
    let cancelled = false
    detectDevice().then((d) => {
      if (cancelled) return
      setDevice(d)
      setSharedDevice(d)
      if (!userPickedPrecision.current) setPrecision(d.recommendedPrecision)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    setMemoryEstimate(estimateMemory(fasta, precision, device))
  }, [fasta, precision, device, setMemoryEstimate])

  // Parse FASTA on every render — cheap, and the chip list + length preview
  // need to track edits live. Errors are surfaced inline rather than thrown.
  let chains: ParsedChain[] = []
  let parseError: string | null = null
  try {
    chains = parseFasta(fasta)
  } catch (e) {
    parseError = (e as Error).message
  }
  const cleaned = chains.map((c) => ({
    ...c,
    sequence: c.sequence.replace(/[^A-Za-z]/g, '').toUpperCase(),
  }))
  const polymerChains = cleaned.filter((c) => c.type !== 'ligand')
  const ligandChains = cleaned.filter((c) => c.type === 'ligand')
  const polymerLen = polymerChains.reduce((acc, c) => acc + c.sequence.length, 0)
  const tooShort = polymerLen > 0 && polymerLen < 8
  const tooLong = polymerLen > 1024

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Scrollable content region — sections stack vertically. */}
      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-5 py-4">
        <SequencesSection
          fasta={fasta}
          setFasta={setFasta}
          parseError={parseError}
          polymerChainCount={polymerChains.length}
          ligandChainCount={ligandChains.length}
          polymerLen={polymerLen}
          tooShort={tooShort}
          tooLong={tooLong}
          onLoadFile={(payload, source) => setStructure(payload, source)}
        />
        <LigandsSection
          fasta={fasta}
          setFasta={setFasta}
          ligandChains={ligandChains}
        />
        <ModelSection
          precision={precision}
          setPrecision={(p) => {
            userPickedPrecision.current = true
            setPrecision(p)
          }}
          device={device}
        />
      </div>

      {/* Sticky bottom: run rail (key forces remount per precision) + sysdata tray. */}
      <div className="shrink-0">
        <RunRail
          key={precision}
          precision={precision}
          chains={cleaned}
          parseError={parseError}
          polymerLen={polymerLen}
          tooShort={tooShort}
          tooLong={tooLong}
        />
        <SysdataTray device={device} />
      </div>
    </div>
  )
}

// Dev-only featurizer regression targets. Kept around for the (unmounted)
// FeaturizerSelfCheck panel so re-enabling it is one mount call.
const GOLDEN_TARGETS: { id: string; seq: string; url: string; aa: number }[] = [
  {
    id: '1L2Y',
    seq: 'NLYIQWLKDGGPSSGRPPPS',
    url: '/feats/1L2Y.golden.boltz-feats',
    aa: 20,
  },
  {
    id: '1CRN',
    seq: 'TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN',
    url: '/feats/1CRN.golden.boltz-feats',
    aa: 46,
  },
]

// Exported so TS doesn't kill it as unused. Not mounted in any pane; dev users
// who want it can drop `<FeaturizerSelfCheck />` into MemoryProbe or similar.
export function FeaturizerSelfCheck() {
  const [running, setRunning] = useState<string | null>(null)
  const [report, setReport] = useState<ValidationReport | null>(null)
  const runOne = async (t: (typeof GOLDEN_TARGETS)[number]) => {
    setRunning(t.id)
    setReport(null)
    try {
      const r = await validateAgainstGolden(t.url, t.seq, t.id)
      setReport(r)
    } catch (e) {
      setReport({
        target: t.id,
        sequence: t.seq,
        pass: false,
        totalTensors: 0,
        passCount: 0,
        diffs: [{ name: '(runtime error)', pass: false, reason: (e as Error).message }],
        elapsedMs: 0,
      })
    } finally {
      setRunning(null)
    }
  }
  return (
    <div className="flex flex-col gap-2">
      <label
        className="font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        Featurizer self-check
      </label>
      <div className="flex gap-1">
        {GOLDEN_TARGETS.map((t) => (
          <Button
            key={t.id}
            variant="outline"
            className="flex-1"
            disabled={running !== null}
            onClick={() => runOne(t)}
          >
            {running === t.id
              ? 'Validating…'
              : `${t.id} · ${t.aa} aa`}
          </Button>
        ))}
      </div>
      {report && (
        <div
          className="flex flex-col gap-1 border p-2 font-mono text-[10px]"
          style={{
            borderColor: report.pass ? 'var(--oxblood)' : 'var(--destructive)',
            background: 'var(--paper-mottle)',
            color: 'var(--ink)',
          }}
        >
          <div className="flex items-center justify-between uppercase tracking-widest">
            <span style={{ color: report.pass ? 'var(--oxblood)' : 'var(--destructive)' }}>
              {report.target} · {report.passCount} / {report.totalTensors} pass
            </span>
            <span style={{ color: 'var(--ink-faded)' }}>
              {report.elapsedMs.toFixed(0)} ms
            </span>
          </div>
          {!report.pass && (
            <ul className="flex max-h-48 flex-col gap-0.5 overflow-y-auto leading-snug">
              {report.diffs
                .filter((d) => !d.pass)
                .slice(0, 20)
                .map((d) => (
                  <li key={d.name} style={{ color: 'var(--destructive)' }}>
                    <span style={{ color: 'var(--ink)' }}>{d.name}</span>{' '}
                    — {d.reason}
                  </li>
                ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}


// ─────────────────────────────────────────────────────────────────────────────
// Canvas pane — Mol\* viewer + empty state.

export function BoltzCanvas() {
  const { structure, error } = useBoltz()

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <p
          className="max-w-md text-center text-sm"
          style={{ color: 'var(--destructive)' }}
        >
          {error}
        </p>
      </div>
    )
  }
  if (!structure) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <p
          className="max-w-md text-center text-sm leading-relaxed"
          style={{ color: 'var(--ink-faded)' }}
        >
          Paste a sequence or open a structure file to begin.
        </p>
      </div>
    )
  }
  return (
    <div className="flex h-full min-h-0 flex-col">
      <MolViewer structure={structure} />
    </div>
  )
}
