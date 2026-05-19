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
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  MolViewer,
  DEFAULT_GLASS_PARAMS,
  DEFAULT_CRYSTAL_PARAMS,
  DEFAULT_VACUUM_PARAMS,
  GLASS_COLOR_THEMES,
  type StructurePayload,
  type ViewMode,
  type GlassParams,
  type VacuumParams,
  type GlassColorTheme,
} from './MolViewer'
import { useModelSession } from '@/hooks/useModelSession'
import { formatBytes } from '@/engine/fetcher'
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

interface BoltzActState {
  structure: StructurePayload | null
  source: string
  error: string | null
  fasta: string
  setStructure: (payload: StructurePayload | null, source: string) => void
  setError: (e: string | null) => void
  setFasta: (text: string) => void
}

const useBoltz = create<BoltzActState>((set) => ({
  structure: null,
  source: '',
  error: null,
  fasta: '',
  setStructure: (payload, source) =>
    set({ structure: payload, source, error: null }),
  setError: (e) => set({ error: e }),
  setFasta: (text) => set({ fasta: text }),
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

const CRAMBIN_PDB_URL = 'https://files.rcsb.org/download/1CRN.pdb'

async function fetchExample(): Promise<StructurePayload> {
  const res = await fetch(CRAMBIN_PDB_URL)
  if (!res.ok) throw new Error(`Failed to fetch 1CRN: HTTP ${res.status}`)
  const data = await res.text()
  return { data, format: 'pdb', id: '1CRN' }
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine loader sub-panel
//
// Three useModelSession hooks (one per graph) are created at the component
// level and keyed by precision via a parent `key` prop so a precision change
// cleanly disposes the previous trio and remounts with fresh manifests.

function GraphProgressBar({
  label,
  status,
  progress,
  totalBytes,
}: {
  label: string
  status: ReturnType<typeof useModelSession>['status']
  progress: ReturnType<typeof useModelSession>['progress']
  totalBytes: number
}) {
  const loaded = progress?.bytesLoaded ?? 0
  const total = progress?.bytesTotal ?? totalBytes
  const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0
  const ready = status === 'ready'
  const rate = progress?.bytesPerSecond
    ? ` · ${formatBytes(progress.bytesPerSecond)}/s`
    : ''
  return (
    <div className="flex flex-col gap-1">
      <div
        className="flex items-center justify-between font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        <span>{label}</span>
        <span>
          {ready
            ? 'ready'
            : status === 'compiling'
              ? 'compiling…'
              : status === 'fetching'
                ? `${formatBytes(loaded)} / ${formatBytes(total)}${rate}`
                : status === 'error'
                  ? 'error'
                  : 'idle'}
        </span>
      </div>
      <div className="h-1 w-full" style={{ background: 'var(--paper-mottle)' }}>
        <div
          className="h-1 transition-[width] duration-150"
          style={{
            width: `${ready ? 100 : pct}%`,
            background: ready
              ? 'var(--oxblood)'
              : status === 'error'
                ? 'var(--destructive)'
                : 'var(--brass)',
          }}
        />
      </div>
    </div>
  )
}

function EngineLoaderFor({ precision }: { precision: BoltzPrecision }) {
  const bundle = boltzBundle(precision)
  const trunk = useModelSession(bundle.trunk)
  const diffusion = useModelSession(bundle.diffusion_step)
  const confidence = useModelSession(bundle.confidence)

  const allIdle =
    trunk.status === 'idle' &&
    diffusion.status === 'idle' &&
    confidence.status === 'idle'
  const anyInFlight =
    trunk.status === 'fetching' ||
    trunk.status === 'compiling' ||
    diffusion.status === 'fetching' ||
    diffusion.status === 'compiling' ||
    confidence.status === 'fetching' ||
    confidence.status === 'compiling'
  const allReady =
    trunk.status === 'ready' &&
    diffusion.status === 'ready' &&
    confidence.status === 'ready'
  const anyError =
    trunk.status === 'error' ||
    diffusion.status === 'error' ||
    confidence.status === 'error'

  const total = bundleApproxBytes(precision)

  return (
    <div className="flex flex-col gap-3">
      {allIdle && (
        <Button
          variant="outline"
          onClick={() => {
            void trunk.load()
            void diffusion.load()
            void confidence.load()
          }}
        >
          Load engine ({formatBytes(total)})
        </Button>
      )}

      {(anyInFlight || allReady || anyError) && (
        <div
          className="flex flex-col gap-2 border p-3"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--paper-mottle)',
          }}
        >
          <GraphProgressBar
            label="Trunk"
            status={trunk.status}
            progress={trunk.progress}
            totalBytes={bundle.trunk.approxBytes + (bundle.trunk.externalDataApproxBytes ?? 0)}
          />
          <GraphProgressBar
            label="Diffusion step"
            status={diffusion.status}
            progress={diffusion.progress}
            totalBytes={
              bundle.diffusion_step.approxBytes +
              (bundle.diffusion_step.externalDataApproxBytes ?? 0)
            }
          />
          <GraphProgressBar
            label="Confidence"
            status={confidence.status}
            progress={confidence.progress}
            totalBytes={
              bundle.confidence.approxBytes +
              (bundle.confidence.externalDataApproxBytes ?? 0)
            }
          />
        </div>
      )}

      {allReady && precision !== 'fp16' && (
        <PredictPanel trunk={trunk.handle!} diffusion={diffusion.handle!} confidence={confidence.handle!} />
      )}

      {allReady && precision === 'fp16' && (
        <div
          className="flex flex-col gap-2 border p-3"
          style={{ borderColor: 'var(--rule)' }}
        >
          <p
            className="font-mono text-[10px] uppercase tracking-widest leading-relaxed"
            style={{ color: 'var(--oxblood)' }}
          >
            Engine warm · {trunk.handle?.executionProvider}
          </p>
          <p className="text-xs" style={{ color: 'var(--ink-faded)' }}>
            v0.1 TS orchestrator routes fp32 graph boundaries; int8 inherits the
            same boundaries (quantize_dynamic keeps I/O fp32, only weights
            quantize). fp16 boundaries need an explicit pack/unpack path that
            lands in a follow-up. Reload as <strong>int8</strong> (mobile-tier
            default) or <strong>fp32</strong> to run the live demo.
          </p>
        </div>
      )}

      {anyError && (
        <p
          className="font-mono text-[10px] uppercase tracking-widest leading-relaxed"
          style={{ color: 'var(--destructive)' }}
        >
          {trunk.error ?? diffusion.error ?? confidence.error}
        </p>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Prediction sub-panel — fires the orchestration loop using the live
// featurizer on whatever sequence is in the FASTA textarea. Visible only
// when the three sessions are ready at fp32 precision.

function PredictPanel({
  trunk,
  diffusion,
  confidence,
}: {
  trunk: NonNullable<ReturnType<typeof useModelSession>['handle']>
  diffusion: NonNullable<ReturnType<typeof useModelSession>['handle']>
  confidence: NonNullable<ReturnType<typeof useModelSession>['handle']>
}) {
  const { fasta, setStructure, setError } = useBoltz()
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<ProgressEvent | null>(null)
  const [stats, setStats] = useState<{
    plddtMean: number
    plddtMin: number
    plddtMax: number
    elapsedMs: number
    residueCount: number
  } | null>(null)

  // Parse the FASTA on render so we can preflight length and disable the button.
  // Wrapping in try/catch keeps the panel responsive while the user is mid-edit
  // (the parser throws on empty body / empty header which is fine at submit time
  // but noisy during typing).
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
  const totalLen = cleaned.reduce((acc, c) => acc + c.sequence.length, 0)
  const chainSummary = cleaned.length === 0
    ? '(empty)'
    : cleaned
        .map((c) => `${c.name || '(no header)'} · ${c.sequence.length}`)
        .join('  +  ')
  const tooShort = totalLen < 8
  const tooLong = totalLen > 1024

  const phaseLabel = (e: ProgressEvent | null) => {
    if (!e) return 'Initialising…'
    if (e.phase === 'recycling') return `Recycling ${e.step + 1}/${e.total}`
    if (e.phase === 'sampling') return `Sampling ${e.step}/${e.total}  σ ${e.sigma.toFixed(2)}`
    if (e.phase === 'confidence') return 'Confidence head…'
    return 'Done'
  }

  return (
    <div
      className="flex flex-col gap-2 border p-3"
      style={{ borderColor: 'var(--oxblood)' }}
    >
      <p
        className="font-mono text-[10px] uppercase tracking-widest leading-relaxed"
        style={{ color: 'var(--oxblood)' }}
      >
        Engine warm · {trunk.executionProvider}.
      </p>
      <div
        className="flex items-center justify-between gap-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        <span title={chainSummary} className="truncate">
          {parseError
            ? <span style={{ color: 'var(--destructive)' }}>{parseError}</span>
            : chainSummary}
        </span>
        <span className="whitespace-nowrap">
          {cleaned.length > 1 && <span>{cleaned.length} chains · </span>}
          {totalLen} res
          {tooShort && totalLen > 0 && (
            <span style={{ color: 'var(--destructive)' }}> · &lt;8</span>
          )}
          {tooLong && (
            <span style={{ color: 'var(--destructive)' }}> · &gt;1024</span>
          )}
        </span>
      </div>
      <Button
        variant="outline"
        disabled={running || tooShort || tooLong || parseError !== null}
        onClick={async () => {
          setRunning(true)
          setError(null)
          setStats(null)
          setProgress(null)
          try {
            const feats = featurizeChains(cleaned)
            const result = await predict({
              feats,
              trunk,
              diffusion,
              confidence,
              recyclingSteps: 1,
              samplingSteps: 50,
              seed: 42,
              onProgress: (e) => setProgress(e),
            })
            const cif = writeMmcif({
              feats,
              atomCoords: result.atomCoords,
              plddt: result.plddt,
              chains: cleaned,
              modelId: 'predicted',
            })
            // Diagnostic: log the first 30 lines + tail of the mmCIF so we
            // can sanity-check the writer output independently of Mol*'s parser.
            // Also log array types / lengths to catch typed-array surprises.
            {
              const head = cif.split('\n').slice(0, 30).join('\n')
              const tail = cif.split('\n').slice(-5).join('\n')
              console.log(
                '[BoltzAct] mmCIF length=%d  atomCoords=%s[%d]  plddt=%s[%d] sample=%o..%o',
                cif.length,
                result.atomCoords.constructor.name,
                result.atomCoords.length,
                result.plddt.constructor.name,
                result.plddt.length,
                Array.from(result.atomCoords.slice(0, 6)),
                Array.from(result.plddt.slice(0, 4)),
              )
              console.log('[BoltzAct] mmCIF head:\n' + head + '\n…\n' + tail)
            }
            const labelBase = cleaned[0]?.name
              ? cleaned[0].name.slice(0, 24)
              : 'unnamed'
            const label = cleaned.length > 1
              ? `${labelBase}+${cleaned.length - 1}`
              : labelBase
            setStructure(
              { data: cif, format: 'mmcif', id: `${label} (predicted)` },
              `Boltz-2 prediction · ${cleaned.length} chain${cleaned.length > 1 ? 's' : ''} · ${(result.elapsedMs / 1000).toFixed(1)} s`,
            )
            const mean =
              Array.from(result.plddt).reduce((a, b) => a + b, 0) /
              result.plddt.length
            let pMin = result.plddt[0]
            let pMax = result.plddt[0]
            for (const v of result.plddt) {
              if (v < pMin) pMin = v
              if (v > pMax) pMax = v
            }
            setStats({
              plddtMean: mean,
              plddtMin: pMin,
              plddtMax: pMax,
              elapsedMs: result.elapsedMs,
              residueCount: totalLen,
            })
          } catch (err) {
            setError((err as Error).message)
          } finally {
            setRunning(false)
          }
        }}
      >
        {running ? phaseLabel(progress) : 'Predict structure'}
      </Button>
      {stats && (
        <div
          className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          <span>Residues</span>
          <span style={{ color: 'var(--ink)' }}>{stats.residueCount}</span>
          <span>pLDDT mean</span>
          <span style={{ color: 'var(--ink)' }}>{stats.plddtMean.toFixed(1)}</span>
          <span>pLDDT range</span>
          <span style={{ color: 'var(--ink)' }}>
            {stats.plddtMin.toFixed(1)} – {stats.plddtMax.toFixed(1)}
          </span>
          <span>Elapsed</span>
          <span style={{ color: 'var(--ink)' }}>
            {(stats.elapsedMs / 1000).toFixed(1)} s
          </span>
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Slots

const EXAMPLE_1L2Y = `>1L2Y Trp-cage miniprotein (20 aa)
NLYIQWLKDGGPSSGRPPPS`
const EXAMPLE_1CRN = `>1CRN Crambin (46 aa)
TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN`
const EXAMPLE_1UBQ = `>1UBQ Ubiquitin (76 aa)
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`
const EXAMPLE_DIMER = `>chain_a 1L2Y copy A
NLYIQWLKDGGPSSGRPPPS
>chain_b 1L2Y copy B
NLYIQWLKDGGPSSGRPPPS`
// Classic UUCG tetraloop — a 10-nt RNA hairpin that folds reliably.
const EXAMPLE_RNA = `>uucg_loop rna
CGCUUCGGCG`
// Drew–Dickerson dodecamer: self-complementary B-form DNA duplex.
const EXAMPLE_DNA = `>strand_a dna
CGCGAATTCGCG
>strand_b dna
CGCGAATTCGCG`

export function BoltzInput() {
  const { fasta, setFasta, setStructure, setError } = useBoltz()
  const [loadingExample, setLoadingExample] = useState(false)
  const [precision, setPrecision] = useState<BoltzPrecision>(DEFAULT_PRECISION)

  return (
    <div className="flex flex-col gap-4">
      {/* Predict section */}
      <div className="flex flex-col gap-2">
        <label
          className="font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          Predict from sequence
        </label>
        <textarea
          value={fasta}
          onChange={(e) => setFasta(e.target.value)}
          rows={5}
          spellCheck={false}
          placeholder=">my_protein&#10;MKLLI…"
          className="w-full resize-y border p-2 font-mono text-xs leading-relaxed"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--paper-mottle)',
            color: 'var(--ink)',
          }}
        />
        <div className="flex flex-wrap gap-1">
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_1L2Y)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Trp-cage miniprotein — smallest validated target"
          >
            1L2Y · 20 aa
          </button>
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_1CRN)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Crambin — small plant protein, three disulfides"
          >
            1CRN · 46 aa
          </button>
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_1UBQ)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Ubiquitin — classic fold benchmark"
          >
            1UBQ · 76 aa
          </button>
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_DIMER)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="1L2Y homodimer — smallest multi-chain target"
          >
            Dimer · 2×20
          </button>
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_RNA)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="UUCG tetraloop — small RNA hairpin (10 nt)"
          >
            RNA · 10 nt
          </button>
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_DNA)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Drew–Dickerson dodecamer — B-form DNA duplex (2 × 12 nt)"
          >
            DNA · 2×12
          </button>
        </div>

        <label
          className="mt-1 font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          Precision
        </label>
        <div className="flex flex-col gap-1">
          {PRECISIONS.map((p) => (
            <label
              key={p}
              className="flex cursor-pointer items-center gap-2 border p-2 text-xs"
              style={{
                borderColor: precision === p ? 'var(--oxblood)' : 'var(--rule)',
                background: precision === p ? 'var(--paper-mottle)' : 'transparent',
              }}
            >
              <input
                type="radio"
                name="boltz-precision"
                checked={precision === p}
                onChange={() => setPrecision(p)}
              />
              <span style={{ color: 'var(--ink)' }}>{PRECISION_LABEL[p]}</span>
            </label>
          ))}
        </div>

        {/* Precision change remounts the loader, disposing previous sessions. */}
        <EngineLoaderFor key={precision} precision={precision} />
      </div>

      <div className="h-px" style={{ background: 'var(--rule)' }} aria-hidden />

      {/* Viewer section — live today */}
      <div className="flex flex-col gap-1.5">
        <label
          className="font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          Load a structure
        </label>
        <label
          className="cursor-pointer border px-3 py-2 text-center text-sm transition-colors"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--paper-mottle)',
            color: 'var(--ink)',
          }}
        >
          Choose .pdb or .mmCIF…
          <input
            type="file"
            accept=".pdb,.ent,.cif,.mmcif"
            className="hidden"
            onChange={async (e) => {
              const file = e.target.files?.[0]
              if (!file) return
              try {
                const data = await file.text()
                const format = detectFormat(file.name, data)
                setStructure({ data, format, id: file.name }, `local: ${file.name}`)
              } catch (err) {
                setError((err as Error).message)
              }
              e.target.value = ''
            }}
          />
        </label>
        <Button
          variant="outline"
          disabled={loadingExample}
          onClick={async () => {
            setLoadingExample(true)
            try {
              const payload = await fetchExample()
              setStructure(payload, 'RCSB 1CRN · Crambin (46 aa)')
            } catch (err) {
              setError((err as Error).message)
            } finally {
              setLoadingExample(false)
            }
          }}
        >
          {loadingExample ? 'Fetching example…' : 'Load example (Crambin, 46 aa)'}
        </Button>
      </div>

      <div className="h-px" style={{ background: 'var(--rule)' }} aria-hidden />

      <FeaturizerSelfCheck />
    </div>
  )
}

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

function FeaturizerSelfCheck() {
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

function ViewModeToggle({
  value,
  onChange,
}: {
  value: ViewMode
  onChange: (v: ViewMode) => void
}) {
  const items: Array<{ key: ViewMode; label: string }> = [
    { key: 'cartoon', label: 'Cartoon' },
    { key: 'glass', label: 'Glass' },
    { key: 'crystal', label: 'Crystal' },
    { key: 'vacuum', label: 'Vacuum' },
  ]
  return (
    <div
      role="radiogroup"
      aria-label="View mode"
      className="flex items-center gap-0 font-mono text-[10px] uppercase tracking-widest"
      style={{ color: 'var(--ink-faded)' }}
    >
      <span style={{ marginRight: 8 }}>view</span>
      {items.map((it, i) => {
        const active = value === it.key
        return (
          <button
            key={it.key}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(it.key)}
            style={{
              padding: '2px 8px',
              border: '1px solid var(--rule)',
              borderLeftWidth: i === 0 ? 1 : 0,
              background: active ? 'var(--ink)' : 'transparent',
              color: active ? 'var(--paper)' : 'var(--ink-faded)',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              letterSpacing: '0.15em',
              textTransform: 'uppercase',
              cursor: 'pointer',
            }}
          >
            {it.label}
          </button>
        )
      })}
    </div>
  )
}

type GlassSliderKey =
  | 'sizeFactor'
  | 'baseSize'
  | 'bfactorFactor'
  | 'resolution'
  | 'smoothness'
  | 'radiusOffset'
  | 'alpha'
  | 'emissive'
  | 'roughness'
  | 'metalness'
  | 'bumpiness'
  | 'bloomStrength'
  | 'bloomRadius'
  | 'bloomThreshold'
  | 'exposure'

type SliderGroup = 'geometry' | 'surface' | 'material' | 'postprocess'

interface SliderDef {
  key: GlassSliderKey
  label: string
  min: number
  max: number
  step: number
  group: SliderGroup
}

const GLASS_SLIDERS: SliderDef[] = [
  // glass-only (putty tube geometry)
  { key: 'sizeFactor',     label: 'size factor',   min: 0.05, max: 3,    step: 0.05, group: 'geometry' },
  { key: 'baseSize',       label: 'base size',     min: 0,    max: 2,    step: 0.05, group: 'geometry' },
  { key: 'bfactorFactor',  label: 'B-fact factor', min: 0,    max: 0.05, step: 0.001, group: 'geometry' },
  // crystal-only (gaussian surface)
  { key: 'resolution',     label: 'resolution',    min: 0.2,  max: 3,    step: 0.05, group: 'surface' },
  { key: 'smoothness',     label: 'smoothness',    min: 1,    max: 3,    step: 0.05, group: 'surface' },
  { key: 'radiusOffset',   label: 'radius offset', min: 0,    max: 3,    step: 0.05, group: 'surface' },
  // shared
  { key: 'alpha',          label: 'alpha',         min: 0,    max: 1,    step: 0.01, group: 'material' },
  { key: 'emissive',       label: 'emissive',      min: 0,    max: 1,    step: 0.01, group: 'material' },
  { key: 'roughness',      label: 'roughness',     min: 0,    max: 1,    step: 0.01, group: 'material' },
  { key: 'metalness',      label: 'metalness',     min: 0,    max: 1,    step: 0.01, group: 'material' },
  { key: 'bumpiness',      label: 'bumpiness',     min: 0,    max: 1,    step: 0.01, group: 'material' },
  { key: 'bloomStrength',  label: 'bloom strength',min: 0,    max: 3,    step: 0.05, group: 'postprocess' },
  { key: 'bloomRadius',    label: 'bloom radius',  min: 0,    max: 2,    step: 0.05, group: 'postprocess' },
  { key: 'bloomThreshold', label: 'bloom thresh',  min: 0,    max: 1,    step: 0.01, group: 'postprocess' },
  { key: 'exposure',       label: 'exposure',      min: 0.3,  max: 3,    step: 0.05, group: 'postprocess' },
]

function GlassSlidersPanel({
  mode,
  params,
  onChange,
}: {
  mode: 'glass' | 'crystal'
  params: GlassParams
  onChange: (next: GlassParams) => void
}) {
  const firstGroup: [SliderGroup, string] =
    mode === 'crystal' ? ['surface', 'surface'] : ['geometry', 'geometry']
  const groups: Array<[SliderGroup, string]> = [
    firstGroup,
    ['material', 'material'],
    ['postprocess', 'postprocess'],
  ]
  const resetTarget = mode === 'crystal' ? DEFAULT_CRYSTAL_PARAMS : DEFAULT_GLASS_PARAMS
  const headerLabel = mode === 'crystal' ? 'crystal · debug' : 'glass · debug'
  return (
    <div
      className="font-mono"
      style={{
        border: '1px solid var(--rule)',
        padding: '10px 12px',
        fontSize: 10,
        color: 'var(--ink-faded)',
      }}
    >
      <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
        <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
          {headerLabel}
        </span>
        <button
          type="button"
          onClick={() => onChange(resetTarget)}
          style={{
            padding: '2px 8px',
            border: '1px solid var(--rule)',
            background: 'transparent',
            color: 'var(--ink-faded)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}
        >
          reset
        </button>
      </div>
      <div className="grid grid-cols-3 gap-x-6 gap-y-1">
        {groups.map(([g, gLabel]) => (
          <div key={g} className="flex flex-col gap-1">
            <div
              style={{
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--ink-faded)',
                opacity: 0.7,
                marginBottom: 2,
              }}
            >
              {gLabel}
            </div>
            {GLASS_SLIDERS.filter((s) => s.group === g).map((s) => {
              const v = params[s.key] as number
              return (
                <label key={s.key} className="flex items-center gap-2">
                  <span style={{ width: 88, color: 'var(--ink-faded)' }}>{s.label}</span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={v}
                    onChange={(e) =>
                      onChange({ ...params, [s.key]: Number(e.target.value) })
                    }
                    style={{ flex: 1, minWidth: 0 }}
                  />
                  <span
                    style={{
                      width: 44,
                      textAlign: 'right',
                      color: 'var(--ink)',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {s.step < 0.01 ? v.toFixed(3) : v.toFixed(2)}
                  </span>
                </label>
              )
            })}
          </div>
        ))}
      </div>
      <div
        className="flex items-center gap-3"
        style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid var(--rule)' }}
      >
        <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
          color
        </span>
        <select
          value={params.colorTheme}
          onChange={(e) =>
            onChange({ ...params, colorTheme: e.target.value as GlassColorTheme })
          }
          style={{
            background: 'transparent',
            border: '1px solid var(--rule)',
            color: 'var(--ink)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            padding: '2px 6px',
            textTransform: 'lowercase',
          }}
        >
          {GLASS_COLOR_THEMES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1" style={{ cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={params.colorReverse}
            onChange={(e) =>
              onChange({ ...params, colorReverse: e.target.checked })
            }
          />
          <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            reverse
          </span>
        </label>
        <span style={{ flex: 1 }} />
        <label className="flex items-center gap-2">
          <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            background
          </span>
          <input
            type="color"
            value={'#' + params.backgroundColor.toString(16).padStart(6, '0')}
            onChange={(e) =>
              onChange({
                ...params,
                backgroundColor: parseInt(e.target.value.slice(1), 16),
              })
            }
            style={{
              width: 26,
              height: 18,
              border: '1px solid var(--rule)',
              background: 'transparent',
              padding: 0,
              cursor: 'pointer',
            }}
          />
        </label>
      </div>
    </div>
  )
}

type VacuumSliderKey =
  | 'wireAlpha'
  | 'wireEmissive'
  | 'wireRoughness'
  | 'wireMetalness'
  | 'wireSizeFactor'
  | 'wireBaseSize'
  | 'wireBfactorFactor'
  | 'shellAlpha'
  | 'shellEmissive'
  | 'shellRoughness'
  | 'shellMetalness'
  | 'shellResolution'
  | 'shellSmoothness'
  | 'shellRadiusOffset'
  | 'bloomStrength'
  | 'bloomRadius'
  | 'bloomThreshold'
  | 'exposure'

type VacuumSliderGroup = 'wire' | 'shell' | 'postprocess'

interface VacuumSliderDef {
  key: VacuumSliderKey
  label: string
  min: number
  max: number
  step: number
  group: VacuumSliderGroup
}

const VACUUM_SLIDERS: VacuumSliderDef[] = [
  { key: 'wireEmissive',      label: 'emissive',      min: 0,    max: 2,    step: 0.01, group: 'wire' },
  { key: 'wireAlpha',         label: 'alpha',         min: 0,    max: 1,    step: 0.01, group: 'wire' },
  { key: 'wireRoughness',     label: 'roughness',     min: 0,    max: 1,    step: 0.01, group: 'wire' },
  { key: 'wireMetalness',     label: 'metalness',     min: 0,    max: 1,    step: 0.01, group: 'wire' },
  { key: 'wireSizeFactor',    label: 'size factor',   min: 0.05, max: 3,    step: 0.05, group: 'wire' },
  { key: 'wireBaseSize',      label: 'base size',     min: 0,    max: 2,    step: 0.05, group: 'wire' },
  { key: 'wireBfactorFactor', label: 'B-fact factor', min: 0,    max: 0.05, step: 0.001, group: 'wire' },
  { key: 'shellAlpha',        label: 'alpha',         min: 0,    max: 1,    step: 0.01, group: 'shell' },
  { key: 'shellEmissive',     label: 'emissive',      min: 0,    max: 1,    step: 0.01, group: 'shell' },
  { key: 'shellRoughness',    label: 'roughness',     min: 0,    max: 1,    step: 0.01, group: 'shell' },
  { key: 'shellMetalness',    label: 'metalness',     min: 0,    max: 1,    step: 0.01, group: 'shell' },
  { key: 'shellResolution',   label: 'resolution',    min: 0.2,  max: 3,    step: 0.05, group: 'shell' },
  { key: 'shellSmoothness',   label: 'smoothness',    min: 1,    max: 3,    step: 0.05, group: 'shell' },
  { key: 'shellRadiusOffset', label: 'radius offset', min: 0,    max: 5,    step: 0.05, group: 'shell' },
  { key: 'bloomStrength',     label: 'bloom strength',min: 0,    max: 3,    step: 0.05, group: 'postprocess' },
  { key: 'bloomRadius',       label: 'bloom radius',  min: 0,    max: 2,    step: 0.05, group: 'postprocess' },
  { key: 'bloomThreshold',    label: 'bloom thresh',  min: 0,    max: 1,    step: 0.01, group: 'postprocess' },
  { key: 'exposure',          label: 'exposure',      min: 0.3,  max: 3,    step: 0.05, group: 'postprocess' },
]

function VacuumSlidersPanel({
  params,
  onChange,
}: {
  params: VacuumParams
  onChange: (next: VacuumParams) => void
}) {
  const groups: Array<[VacuumSliderGroup, string]> = [
    ['wire', 'wire'],
    ['shell', 'shell'],
    ['postprocess', 'postprocess'],
  ]
  return (
    <div
      className="font-mono"
      style={{
        border: '1px solid var(--rule)',
        padding: '10px 12px',
        fontSize: 10,
        color: 'var(--ink-faded)',
      }}
    >
      <div className="flex items-center justify-between" style={{ marginBottom: 8 }}>
        <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
          vacuum · debug
        </span>
        <button
          type="button"
          onClick={() => onChange(DEFAULT_VACUUM_PARAMS)}
          style={{
            padding: '2px 8px',
            border: '1px solid var(--rule)',
            background: 'transparent',
            color: 'var(--ink-faded)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}
        >
          reset
        </button>
      </div>
      <div className="grid grid-cols-3 gap-x-6 gap-y-1">
        {groups.map(([g, gLabel]) => (
          <div key={g} className="flex flex-col gap-1">
            <div
              style={{
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: 'var(--ink-faded)',
                opacity: 0.7,
                marginBottom: 2,
              }}
            >
              {gLabel}
            </div>
            {VACUUM_SLIDERS.filter((s) => s.group === g).map((s) => {
              const v = params[s.key] as number
              return (
                <label key={s.key} className="flex items-center gap-2">
                  <span style={{ width: 88, color: 'var(--ink-faded)' }}>{s.label}</span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={v}
                    onChange={(e) =>
                      onChange({ ...params, [s.key]: Number(e.target.value) })
                    }
                    style={{ flex: 1, minWidth: 0 }}
                  />
                  <span
                    style={{
                      width: 44,
                      textAlign: 'right',
                      color: 'var(--ink)',
                      fontVariantNumeric: 'tabular-nums',
                    }}
                  >
                    {s.step < 0.01 ? v.toFixed(3) : v.toFixed(2)}
                  </span>
                </label>
              )
            })}
          </div>
        ))}
      </div>
      <div
        className="flex items-center gap-3"
        style={{ marginTop: 10, paddingTop: 8, borderTop: '1px solid var(--rule)' }}
      >
        <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
          color
        </span>
        <select
          value={params.colorTheme}
          onChange={(e) =>
            onChange({ ...params, colorTheme: e.target.value as GlassColorTheme })
          }
          style={{
            background: 'transparent',
            border: '1px solid var(--rule)',
            color: 'var(--ink)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            padding: '2px 6px',
            textTransform: 'lowercase',
          }}
        >
          {GLASS_COLOR_THEMES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1" style={{ cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={params.colorReverse}
            onChange={(e) =>
              onChange({ ...params, colorReverse: e.target.checked })
            }
          />
          <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            reverse
          </span>
        </label>
        <span style={{ flex: 1 }} />
        <label className="flex items-center gap-2">
          <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
            background
          </span>
          <input
            type="color"
            value={'#' + params.backgroundColor.toString(16).padStart(6, '0')}
            onChange={(e) =>
              onChange({
                ...params,
                backgroundColor: parseInt(e.target.value.slice(1), 16),
              })
            }
            style={{
              width: 26,
              height: 18,
              border: '1px solid var(--rule)',
              background: 'transparent',
              padding: 0,
              cursor: 'pointer',
            }}
          />
        </label>
      </div>
    </div>
  )
}

export function BoltzCanvas() {
  const { structure, error } = useBoltz()
  const [viewMode, setViewMode] = useState<ViewMode>('cartoon')
  // Per-mode params: switching between modes preserves each one's tuning
  // state, so you can A/B them without re-dialing sliders.
  const [glassParams, setGlassParams] = useState<GlassParams>(DEFAULT_GLASS_PARAMS)
  const [crystalParams, setCrystalParams] = useState<GlassParams>(DEFAULT_CRYSTAL_PARAMS)
  const [vacuumParams, setVacuumParams] = useState<VacuumParams>(DEFAULT_VACUUM_PARAMS)
  const activeGlassParams = viewMode === 'crystal' ? crystalParams : glassParams
  const setActiveGlassParams =
    viewMode === 'crystal' ? setCrystalParams : setGlassParams
  if (error) {
    return <p style={{ color: 'var(--destructive)' }}>{error}</p>
  }
  if (!structure) {
    return (
      <div className="flex h-full min-h-[280px] items-center justify-center">
        <p className="max-w-md text-center" style={{ color: 'var(--ink-faded)' }}>
          Load a structure on the left to inspect it. The viewer is Mol*,
          configured for a paper-tinted backdrop and the field-guide
          register; right-click for advanced controls.
        </p>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      <div className="flex justify-end">
        <ViewModeToggle value={viewMode} onChange={setViewMode} />
      </div>
      <MolViewer
        structure={structure}
        viewMode={viewMode}
        glassParams={activeGlassParams}
        vacuumParams={vacuumParams}
      />
      {(viewMode === 'glass' || viewMode === 'crystal') && (
        <GlassSlidersPanel
          mode={viewMode}
          params={activeGlassParams}
          onChange={setActiveGlassParams}
        />
      )}
      {viewMode === 'vacuum' && (
        <VacuumSlidersPanel params={vacuumParams} onChange={setVacuumParams} />
      )}
    </div>
  )
}

export function BoltzOutput() {
  const { structure, source } = useBoltz()
  if (!structure) {
    return (
      <p
        className="font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        No specimen mounted.
      </p>
    )
  }

  const lines = structure.data.split(/\r?\n/)
  const atomCount = lines.filter((l) => l.startsWith('ATOM ') || l.startsWith('ATOM  ')).length
  const hetCount = lines.filter((l) => l.startsWith('HETATM')).length
  const headerLine =
    structure.format === 'pdb'
      ? lines.find((l) => l.startsWith('HEADER'))?.slice(10, 70).trim()
      : lines.find((l) => l.startsWith('_struct.title'))?.split(/\s+/).slice(1).join(' ')

  const rows: [string, string][] = [
    ['Identifier', structure.id],
    ['Format', structure.format.toUpperCase()],
    ['Source', source],
    ['ATOM records', String(atomCount)],
    ['HETATM records', String(hetCount)],
    ['Header', headerLine || '—'],
  ]

  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="uppercase tracking-widest" style={{ color: 'var(--ink-faded)' }}>
            {k}
          </dt>
          <dd style={{ color: 'var(--ink)', wordBreak: 'break-word' }}>{v}</dd>
        </div>
      ))}
    </dl>
  )
}
