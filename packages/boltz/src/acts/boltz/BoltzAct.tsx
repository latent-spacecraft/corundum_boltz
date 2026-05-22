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
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  MolViewer,
  METALS,
  BUNDLED_PRESETS,
  type StructurePayload,
  type Metal,
  type JewelryPreset,
  type JewelryPresets,
} from './MolViewer'
import { MoleroViewer } from '@/molero/MoleroViewer'
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
import { loadLigandBlob, type LigandBlob } from './featurizer/ligand'
import { useLigandInsertSlot } from './LigandDrawer'

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
  setStructure: (payload: StructurePayload | null, source: string) => void
  setStreamingFrame: (payload: StructurePayload) => void
  setStreaming: (s: boolean) => void
  setError: (e: string | null) => void
  setFasta: (text: string) => void
}

const useBoltz = create<BoltzActState>((set) => ({
  structure: null,
  source: '',
  error: null,
  fasta: '',
  streaming: false,
  setStructure: (payload, source) =>
    set({ structure: payload, source, error: null, streaming: false }),
  // Streaming frames replace structure but keep the source line stable so
  // the output pane doesn't churn the provenance label every frame.
  setStreamingFrame: (payload) => set({ structure: payload, error: null }),
  setStreaming: (s) => set({ streaming: s }),
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
  const { fasta, setStructure, setStreamingFrame, setStreaming, setError } = useBoltz()
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
  // For totalLen / tooShort / tooLong: polymer residues count as 1 unit each;
  // a ligand chain only contributes its CCD label, but the trunk runs token-
  // per-atom so the *real* token cost of a ligand is its atom count, which we
  // only know after fetching the blob. For preflight purposes we conservatively
  // count one ligand token-equivalent per ligand chain; the limit check is
  // re-asserted on the real `feats.N` once the blob is loaded.
  const totalLen = cleaned.reduce((acc, c) => {
    if (c.type === 'ligand') return acc + 1
    return acc + c.sequence.length
  }, 0)
  const polymerLen = cleaned
    .filter((c) => c.type !== 'ligand')
    .reduce((acc, c) => acc + c.sequence.length, 0)
  const chainSummary = cleaned.length === 0
    ? '(empty)'
    : cleaned
        .map((c) =>
          c.type === 'ligand'
            ? `${c.name || c.sequence} (ligand)`
            : `${c.name || '(no header)'} · ${c.sequence.length}`,
        )
        .join('  +  ')
  // Length budget applies to polymer residues only (ligands are tiny relative
  // to the 1024-residue trunk budget — a typical cofactor adds ~30-50 atom-
  // tokens, well under the polymer cap).
  const tooShort = polymerLen < 8
  const tooLong = polymerLen > 1024
  void totalLen

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
          {polymerLen} res
          {tooShort && polymerLen > 0 && (
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
            // Resolve ligand blobs from /ccd/<CODE>.json before featurizing.
            // Polymer chains pass through unchanged.
            const withBlobs = await Promise.all(
              cleaned.map(async (c) => {
                if (c.type !== 'ligand') return c
                const blob: LigandBlob = await loadLigandBlob(c.sequence)
                return { ...c, blob }
              }),
            )
            const feats = featurizeChains(withBlobs)

            // Streaming setup: throttle per-step frames to ~1 in every 3
            // diffusion steps so the canvas rebuild (wire + side-chains +
            // ligand reps) overlaps with the next ONNX call. The shell
            // stays hidden during streaming; pLDDT isn't known yet so we
            // pass a flat 50.0 placeholder — wire thickness reads as
            // uniform metal until the confidence head runs.
            setStreaming(true)
            const labelBaseStream = withBlobs[0]?.name
              ? withBlobs[0].name.slice(0, 24)
              : 'unnamed'
            const labelStream = withBlobs.length > 1
              ? `${labelBaseStream}+${withBlobs.length - 1}`
              : labelBaseStream
            const placeholderPlddt = new Float32Array(feats.N).fill(50)
            const STREAM_EVERY = 3
            let inFlight = false  // drop frames if a prior write is still painting

            const result = await predict({
              feats,
              trunk,
              diffusion,
              confidence,
              recyclingSteps: 1,
              samplingSteps: 50,
              seed: 42,
              onProgress: (e) => setProgress(e),
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
                    // id ticks each frame so React/MolViewer notice the change
                    id: `${labelStream} (diffusion ${step}/${total})`,
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
            const labelBase = withBlobs[0]?.name
              ? withBlobs[0].name.slice(0, 24)
              : 'unnamed'
            const label = withBlobs.length > 1
              ? `${labelBase}+${withBlobs.length - 1}`
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
              residueCount: feats.N,
            })
          } catch (err) {
            setError((err as Error).message)
          } finally {
            setRunning(false)
            // Defensive: setStructure on the success path already clears
            // streaming, but a mid-run error would leave it stuck on.
            setStreaming(false)
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
// Crambin (1CRN) + a heme cofactor — exercises the ligand cofolding path.
// Crambin doesn't actually bind heme biologically; this is a featurization
// smoke test, not a biology claim.
const EXAMPLE_PROT_LIG = `>1CRN Crambin
TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
>heme ligand
HEM`

function validateUniProtAccession(accession: string): string {
  const acc = accession.trim().toUpperCase()
  if (!acc) throw new Error('Enter a UniProt accession (e.g. P02768)')
  if (!/^[A-Z][A-Z0-9]{5,9}$/.test(acc)) {
    throw new Error(`'${acc}' doesn't look like a UniProt accession`)
  }
  return acc
}

/**
 * Fetch the AlphaFold-DB precomputed structure for a UniProt accession.
 *
 * AlphaFoldDB hosts structures at a stable URL template; for proteins ≤ 2700
 * residues the `-F1` fragment is the only model and `model_v4` is the
 * current (2024+) database version. mmCIF carries per-residue pLDDT in the
 * B-factor column — Mol* already paints that, and Molero Phase 2's emission
 * channel will read it directly.
 *
 * No Boltz inference required — this is the fast "show me what AlphaFold
 * predicted" path. Useful both as a viewer testbed and as a baseline for
 * comparing Boltz predictions against AlphaFold's.
 */
async function fetchAlphaFold(accession: string): Promise<StructurePayload> {
  const acc = validateUniProtAccession(accession)
  const url = `https://alphafold.ebi.ac.uk/files/AF-${acc}-F1-model_v4.cif`
  const res = await fetch(url)
  if (!res.ok) {
    if (res.status === 404) {
      throw new Error(
        `AlphaFoldDB has no structure for ${acc} ` +
        `(not yet predicted, or > 2700 residues — multi-fragment models aren't supported yet)`,
      )
    }
    throw new Error(`AlphaFoldDB fetch failed: HTTP ${res.status}`)
  }
  const data = await res.text()
  return { data, format: 'mmcif' as const, id: `AF-${acc}` }
}

/**
 * Fetch a UniProt entry by accession and return a FASTA-ready chunk.
 *
 * UniProt's JSON endpoint carries the sequence + protein name + organism +
 * a rich `features` array (active sites, modifications, variants, …) and
 * cross-references to AlphaFold / PDB. We use sequence + name + organism
 * to populate the FASTA box today; the full payload is stashed in window
 * scope as a forward-looking hook so Molero's channel-mapping system can
 * pick up per-residue features (modifications, variants, active sites)
 * without re-fetching.
 */
async function fetchUniProt(accession: string): Promise<string> {
  const acc = validateUniProtAccession(accession)
  const res = await fetch(`https://rest.uniprot.org/uniprotkb/${acc}.json`)
  if (!res.ok) {
    if (res.status === 404) throw new Error(`UniProt accession '${acc}' not found`)
    throw new Error(`UniProt fetch failed: HTTP ${res.status}`)
  }
  const data = await res.json()
  const seq: string | undefined = data?.sequence?.value
  if (!seq) throw new Error(`UniProt response for ${acc} has no sequence`)
  const name: string =
    data?.proteinDescription?.recommendedName?.fullName?.value ??
    data?.proteinDescription?.submissionNames?.[0]?.fullName?.value ??
    'unknown'
  const organism: string = data?.organism?.scientificName ?? ''
  const header = `>${acc} ${name}${organism ? ' | ' + organism : ''}`
  // Stash the full payload — Molero Phase 2 (property channels) will read
  // features[] (ACTIVE_SITE, MOD_RES, VARIANT, DISULFID, …) here without
  // another network call. Keyed by accession so multiple loads coexist.
  ;(window as unknown as { __uniprotCache?: Record<string, unknown> }).__uniprotCache =
    {
      ...((window as unknown as { __uniprotCache?: Record<string, unknown> }).__uniprotCache ?? {}),
      [acc]: data,
    }
  return `${header}\n${seq}`
}

export function BoltzInput() {
  const { fasta, setFasta, setStructure, setError } = useBoltz()
  const [loadingExample, setLoadingExample] = useState(false)
  const [precision, setPrecision] = useState<BoltzPrecision>(DEFAULT_PRECISION)
  const [uniprotAcc, setUniprotAcc] = useState('')
  const [uniprotLoading, setUniprotLoading] = useState(false)
  const [uniprotError, setUniprotError] = useState<string | null>(null)
  const setLigandInsert = useLigandInsertSlot((s) => s.setInsert)

  // Wire the drawer's "click a ligand" action to this textarea. Re-registers
  // on every render so the slot always closes over the latest fasta string —
  // the drawer can be opened/clicked any time without staleness.
  useEffect(() => {
    setLigandInsert((ccd) => {
      const code = ccd.toUpperCase()
      // De-dup: skip if the same ligand chain is already in the input.
      const tag = `ligand`
      const re = new RegExp(`^>\\S+\\s+${tag}\\s*\\r?\\n${code}\\b`, 'mi')
      if (re.test(fasta)) return
      const chunk = `>lig_${code} ligand\n${code}`
      const next = fasta.trim() ? `${fasta.trim()}\n${chunk}\n` : `${chunk}\n`
      setFasta(next)
    })
    return () => setLigandInsert(null)
  }, [fasta, setFasta, setLigandInsert])

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
          <button
            type="button"
            onClick={() => setFasta(EXAMPLE_PROT_LIG)}
            className="flex-1 border px-2 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
            title="Crambin + heme cofactor — first ligand cofolding test"
          >
            Prot + HEM
          </button>
        </div>

        {/* UniProt-accession loader — one shared input drives two actions:
              · AlphaFold (primary)   — fetch precomputed mmCIF and render
                                        immediately; no Boltz inference needed.
              · Sequence (secondary) — populate the FASTA box for a refold.
            Full UniProt JSON is cached on window.__uniprotCache so Molero's
            Phase-2 channel mappings can read per-residue features later. */}
        <label
          className="mt-1 font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          UniProt accession
        </label>
        <div className="flex gap-1">
          <input
            type="text"
            value={uniprotAcc}
            onChange={(e) => setUniprotAcc(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !uniprotLoading) {
                e.preventDefault()
                e.currentTarget.blur()
                // Enter defaults to the primary action: render AlphaFold structure.
                ;(async () => {
                  setUniprotLoading(true)
                  setUniprotError(null)
                  try {
                    const payload = await fetchAlphaFold(uniprotAcc)
                    setStructure(payload, `AlphaFoldDB · ${payload.id}`)
                  } catch (err) {
                    setUniprotError((err as Error).message)
                  } finally {
                    setUniprotLoading(false)
                  }
                })()
              }
            }}
            placeholder="P02768"
            spellCheck={false}
            className="flex-1 border px-2 py-1 font-mono text-xs uppercase tracking-wide"
            style={{
              borderColor: 'var(--rule)',
              background: 'var(--paper-mottle)',
              color: 'var(--ink)',
            }}
          />
          <button
            type="button"
            disabled={uniprotLoading || !uniprotAcc.trim()}
            onClick={async () => {
              setUniprotLoading(true)
              setUniprotError(null)
              try {
                const payload = await fetchAlphaFold(uniprotAcc)
                setStructure(payload, `AlphaFoldDB · ${payload.id}`)
              } catch (err) {
                setUniprotError((err as Error).message)
              } finally {
                setUniprotLoading(false)
              }
            }}
            className="border px-3 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{
              borderColor: 'var(--oxblood)',
              color: 'var(--ink)',
              background: 'var(--paper-mottle)',
              opacity: uniprotLoading || !uniprotAcc.trim() ? 0.5 : 1,
            }}
            title="Fetch the precomputed AlphaFold structure (mmCIF) and render it directly — no Boltz inference."
          >
            {uniprotLoading ? 'Fetching…' : 'AlphaFold'}
          </button>
          <button
            type="button"
            disabled={uniprotLoading || !uniprotAcc.trim()}
            onClick={async () => {
              setUniprotLoading(true)
              setUniprotError(null)
              try {
                const chunk = await fetchUniProt(uniprotAcc)
                setFasta(chunk)
              } catch (err) {
                setUniprotError((err as Error).message)
              } finally {
                setUniprotLoading(false)
              }
            }}
            className="border px-3 py-1 font-mono text-[10px] uppercase tracking-widest"
            style={{
              borderColor: 'var(--rule)',
              color: 'var(--ink-faded)',
              opacity: uniprotLoading || !uniprotAcc.trim() ? 0.5 : 1,
            }}
            title="Fetch sequence + UniProt metadata into the FASTA box, ready to refold with Boltz."
          >
            Sequence
          </button>
        </div>
        {uniprotError && (
          <p
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: 'var(--destructive)' }}
          >
            {uniprotError}
          </p>
        )}

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

// ─────────────────────────────────────────────────────────────────────────────
// Canvas slot — jewelry mode only. The user picks a metal; the wire/shell
// material is locked to the jewelry register. No slider noise.

function MetalToggle({
  value,
  presets,
  onChange,
}: {
  value: Metal
  presets: JewelryPresets
  onChange: (v: Metal) => void
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Metal"
      className="flex items-center gap-0 font-mono text-[10px] uppercase tracking-widest"
      style={{ color: 'var(--ink-faded)' }}
    >
      <span style={{ marginRight: 8 }}>metal</span>
      {METALS.map((m, i) => {
        const active = value === m
        return (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(m)}
            title={m}
            style={{
              padding: '2px 10px',
              border: '1px solid var(--rule)',
              borderLeftWidth: i === 0 ? 1 : 0,
              background: active
                ? `#${presets[m].color.toString(16).padStart(6, '0')}`
                : 'transparent',
              color: active ? '#1a1410' : 'var(--ink-faded)',
              fontFamily: 'var(--font-mono)',
              fontSize: 10,
              letterSpacing: '0.15em',
              textTransform: 'uppercase',
              cursor: 'pointer',
            }}
          >
            {m}
          </button>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Material settings panel — every numeric param in the active metal's
// preset, grouped. The user adjusts, hits Save to persist to
// jewelry-presets.json via a dev-only Vite middleware.

type SliderGroupKey = 'metal' | 'scene' | 'lighting' | 'wire' | 'sidechain' | 'ligand' | 'shell'
interface NumSlider {
  key: keyof JewelryPreset
  label: string
  min: number
  max: number
  step: number
  group: SliderGroupKey
}
const NUM_SLIDERS: NumSlider[] = [
  // metal armature material
  { key: 'metalness',           label: 'metalness',         min: 0,    max: 1,    step: 0.01,  group: 'metal' },
  { key: 'roughness',           label: 'roughness',         min: 0,    max: 1,    step: 0.01,  group: 'metal' },
  { key: 'emissive',            label: 'emissive',          min: 0,    max: 1,    step: 0.01,  group: 'metal' },
  // scene
  { key: 'exposure',            label: 'exposure',          min: 0.3,  max: 3,    step: 0.05,  group: 'scene' },
  { key: 'bloomStrength',       label: 'bloom strength',    min: 0,    max: 3,    step: 0.05,  group: 'scene' },
  { key: 'bloomRadius',         label: 'bloom radius',      min: 0,    max: 2,    step: 0.05,  group: 'scene' },
  { key: 'bloomThreshold',      label: 'bloom thresh',      min: 0,    max: 1,    step: 0.01,  group: 'scene' },
  // studio rig — intensities only; positions/colors are baked into MolViewer
  { key: 'ambientIntensity',    label: 'ambient',           min: 0,    max: 1,    step: 0.01,  group: 'lighting' },
  { key: 'keyIntensity',        label: 'key',               min: 0,    max: 3,    step: 0.05,  group: 'lighting' },
  { key: 'fillIntensity',       label: 'fill',              min: 0,    max: 3,    step: 0.05,  group: 'lighting' },
  { key: 'rimIntensity',        label: 'rim',               min: 0,    max: 3,    step: 0.05,  group: 'lighting' },
  { key: 'topIntensity',        label: 'top',               min: 0,    max: 3,    step: 0.05,  group: 'lighting' },
  { key: 'bounceIntensity',     label: 'bounce',            min: 0,    max: 3,    step: 0.05,  group: 'lighting' },
  // wire (putty)
  { key: 'wireSizeFactor',      label: 'size factor',       min: 0.05, max: 3,    step: 0.05,  group: 'wire' },
  { key: 'wireBaseSize',        label: 'base size',         min: 0,    max: 2,    step: 0.05,  group: 'wire' },
  { key: 'wireBfactorFactor',   label: 'B-fact factor',     min: 0,    max: 0.05, step: 0.001, group: 'wire' },
  // side chains
  { key: 'sideChainSizeFactor', label: 'size factor',       min: 0.05, max: 1,    step: 0.01,  group: 'sidechain' },
  { key: 'sideChainAspectRatio',label: 'aspect ratio',      min: 0.1,  max: 2,    step: 0.05,  group: 'sidechain' },
  { key: 'sideChainBondScale',  label: 'bond scale',        min: 0.05, max: 1,    step: 0.01,  group: 'sidechain' },
  // ligand
  { key: 'ligandSizeFactor',    label: 'size factor',       min: 0.05, max: 1.5,  step: 0.01,  group: 'ligand' },
  { key: 'ligandAspectRatio',   label: 'aspect ratio',      min: 0.1,  max: 2,    step: 0.05,  group: 'ligand' },
  { key: 'ligandBondScale',     label: 'bond scale',        min: 0.05, max: 1,    step: 0.01,  group: 'ligand' },
  // shell (liquid-glass CSS backdrop-filter overlay)
  { key: 'shellBlur',             label: 'blur',         min: 0,   max: 30,  step: 0.5,  group: 'shell' },
  { key: 'shellBrightness',       label: 'brightness',   min: 0.5, max: 2,   step: 0.01, group: 'shell' },
  { key: 'shellSaturation',       label: 'saturation',   min: 0,   max: 3,   step: 0.05, group: 'shell' },
  { key: 'shellEnvelopePad',      label: 'envelope pad', min: 0,   max: 60,  step: 1,    group: 'shell' },
  { key: 'shellSmoothIterations', label: 'smoothing',    min: 0,   max: 4,   step: 1,    group: 'shell' },
  { key: 'shellTintAmount',       label: 'tint amount',  min: 0,   max: 1,   step: 0.01, group: 'shell' },
  { key: 'shellEdgeHighlight',    label: 'edge rim',     min: 0,   max: 1,   step: 0.01, group: 'shell' },
  { key: 'shellEdgeWidth',        label: 'edge width',   min: 0,   max: 60,  step: 1,    group: 'shell' },
]
const GROUP_ORDER: SliderGroupKey[] = ['metal', 'scene', 'lighting', 'wire', 'sidechain', 'ligand', 'shell']
const GROUP_LABEL: Record<SliderGroupKey, string> = {
  metal: 'metal',
  scene: 'scene',
  lighting: 'lighting',
  wire: 'wire',
  sidechain: 'side chains',
  ligand: 'ligand',
  shell: 'shell',
}

function hexFromInt(n: number): string {
  return '#' + (n & 0xffffff).toString(16).padStart(6, '0')
}
function intFromHex(s: string): number {
  return parseInt(s.replace('#', ''), 16) & 0xffffff
}

function JewelrySettingsPanel({
  metal,
  preset,
  onChange,
  onSave,
  onReset,
  saveState,
}: {
  metal: Metal
  preset: JewelryPreset
  onChange: (next: JewelryPreset) => void
  onSave: () => void
  onReset: () => void
  saveState: 'idle' | 'saving' | 'saved' | 'error'
}) {
  const [collapsed, setCollapsed] = useState(false)
  const set = <K extends keyof JewelryPreset>(k: K, v: JewelryPreset[K]) =>
    onChange({ ...preset, [k]: v })
  return (
    <div
      className="font-mono"
      style={{
        border: '1px solid var(--rule)',
        fontSize: 10,
        color: 'var(--ink-faded)',
      }}
    >
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full cursor-pointer select-none items-center justify-between"
        style={{
          padding: '6px 12px',
          background: 'transparent',
          border: 'none',
          fontFamily: 'var(--font-mono)',
          fontSize: 10,
          color: 'var(--ink-faded)',
          letterSpacing: '0.15em',
          textTransform: 'uppercase',
          cursor: 'pointer',
        }}
      >
        <span>{collapsed ? '▶' : '▼'}&nbsp;&nbsp;{metal} · material</span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {saveState === 'saved' && (
            <span style={{ color: 'var(--oxblood)' }}>saved</span>
          )}
          {saveState === 'error' && (
            <span style={{ color: 'var(--destructive)' }}>save error</span>
          )}
        </span>
      </button>
      {!collapsed && (
        <div style={{ padding: '4px 12px 12px' }}>
          {/* Color + background — special non-slider rows */}
          <div
            className="flex items-center gap-6"
            style={{ paddingBottom: 8, marginBottom: 8, borderBottom: '1px solid var(--rule)' }}
          >
            <label className="flex items-center gap-2">
              <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                color
              </span>
              <input
                type="color"
                value={hexFromInt(preset.color)}
                onChange={(e) => set('color', intFromHex(e.target.value))}
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
            <label className="flex items-center gap-2">
              <span style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}>
                background
              </span>
              <input
                type="color"
                value={hexFromInt(preset.background)}
                onChange={(e) => set('background', intFromHex(e.target.value))}
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
            <span style={{ flex: 1 }} />
            <button
              type="button"
              onClick={onReset}
              style={{
                padding: '2px 10px',
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
            <button
              type="button"
              onClick={onSave}
              disabled={saveState === 'saving'}
              style={{
                padding: '2px 10px',
                border: '1px solid var(--oxblood)',
                background: 'var(--oxblood)',
                color: 'var(--primary-foreground)',
                fontFamily: 'var(--font-mono)',
                fontSize: 10,
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                cursor: saveState === 'saving' ? 'wait' : 'pointer',
                opacity: saveState === 'saving' ? 0.6 : 1,
              }}
            >
              {saveState === 'saving' ? 'saving…' : 'save as default'}
            </button>
          </div>

          <div className="grid grid-cols-3 gap-x-6 gap-y-1">
            {GROUP_ORDER.map((g) => {
              const sliders = NUM_SLIDERS.filter((s) => s.group === g)
              if (sliders.length === 0) return null
              return (
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
                    {GROUP_LABEL[g]}
                  </div>
                  {sliders.map((s) => {
                    const v = preset[s.key] as number
                    return (
                      <label key={s.key} className="flex items-center gap-2">
                        <span style={{ width: 96, color: 'var(--ink-faded)' }}>
                          {s.label}
                        </span>
                        <input
                          type="range"
                          min={s.min}
                          max={s.max}
                          step={s.step}
                          value={v}
                          onChange={(e) => set(s.key, Number(e.target.value) as any)}
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
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

type Engine = 'molstar' | 'molero'

function EngineToggle({
  value,
  onChange,
}: {
  value: Engine
  onChange: (v: Engine) => void
}) {
  const items: { key: Engine; label: string; title: string }[] = [
    { key: 'molstar', label: 'Mol*',  title: 'Mol* / jewelry register (default, full featured)' },
    { key: 'molero', label: 'Molero', title: 'Molero / WebGPU (Phase 1 — spheres only)' },
  ]
  return (
    <div
      role="radiogroup"
      aria-label="Renderer engine"
      className="flex items-center gap-0 font-mono text-[10px] uppercase tracking-widest"
      style={{ color: 'var(--ink-faded)' }}
    >
      <span style={{ marginRight: 8 }}>engine</span>
      {items.map((it, i) => {
        const active = value === it.key
        return (
          <button
            key={it.key}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(it.key)}
            title={it.title}
            style={{
              padding: '2px 10px',
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

export function BoltzCanvas() {
  const { structure, error, streaming } = useBoltz()
  const [engine, setEngine] = useState<Engine>('molstar')
  const [metal, setMetal] = useState<Metal>('gold')
  const [presets, setPresets] = useState<JewelryPresets>(BUNDLED_PRESETS)
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle')

  const updateActivePreset = (next: JewelryPreset) =>
    setPresets({ ...presets, [metal]: next })

  const saveAllPresets = async () => {
    setSaveState('saving')
    try {
      const res = await fetch('/__save_jewelry_preset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(presets, null, 2),
      })
      if (!res.ok) throw new Error(await res.text())
      setSaveState('saved')
      setTimeout(() => setSaveState('idle'), 2500)
    } catch (e) {
      console.error('[BoltzCanvas] save preset failed:', e)
      setSaveState('error')
      setTimeout(() => setSaveState('idle'), 4000)
    }
  }

  const resetActivePreset = () =>
    setPresets({ ...presets, [metal]: BUNDLED_PRESETS[metal] })

  if (error) {
    return <p style={{ color: 'var(--destructive)' }}>{error}</p>
  }
  if (!structure) {
    return (
      <div className="flex h-full min-h-[280px] items-center justify-center">
        <p className="max-w-md text-center" style={{ color: 'var(--ink-faded)' }}>
          Load a structure on the left to inspect it. Chains render as a
          polished metal armature inside a translucent gem shell; confidence
          drives both wire thickness and gem tint.
        </p>
      </div>
    )
  }
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-end gap-4">
        <EngineToggle value={engine} onChange={setEngine} />
        {engine === 'molstar' && (
          <MetalToggle value={metal} presets={presets} onChange={setMetal} />
        )}
      </div>
      {engine === 'molstar' ? (
        <MolViewer structure={structure} metal={metal} presets={presets} streaming={streaming} />
      ) : (
        <MoleroViewer structure={structure} />
      )}
      {engine === 'molstar' && (
        <JewelrySettingsPanel
          metal={metal}
          preset={presets[metal]}
          onChange={updateActivePreset}
          onSave={saveAllPresets}
          onReset={resetActivePreset}
          saveState={saveState}
        />
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
