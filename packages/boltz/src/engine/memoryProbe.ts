/**
 * In-browser memory profiler — driven by the debug panel, fed by hand
 * from the predict pipeline.
 *
 * Two memory APIs we try, in priority order:
 *
 *   1. performance.measureUserAgentSpecificMemory()  ← Chrome 89+, Safari 16+
 *      Requires COOP/COEP (already on; see vite.config.ts). Reports the
 *      *process-wide* breakdown including the WASM heap that ORT-Web
 *      lives in — exactly the number Safari evicts on. Async, can take
 *      ~100ms because it waits for GC.
 *
 *   2. performance.memory.usedJSHeapSize                ← Chrome non-standard
 *      JS heap only — misses WASM, but better than nothing on Chromium
 *      that doesn't expose the standard API.
 *
 * Both fail silently to `{ bytes: 0, source: 'none' }`. The panel renders
 * a "memory API unavailable" hint when that happens.
 *
 * Recording model: a zustand-backed timeline of (phase, timestamp, bytes)
 * tuples. `recordPhase()` is callable from anywhere — orchestrator, model
 * session hook, BoltzAct's load button. The panel shows a deltas-from-start
 * table and a running peak.
 */
import { create } from 'zustand'

interface MeasureMemoryResult {
  bytes: number
  breakdown?: unknown
}

type MeasureMemoryFn = () => Promise<MeasureMemoryResult>

interface LegacyPerformanceMemory {
  usedJSHeapSize: number
  totalJSHeapSize: number
  jsHeapSizeLimit: number
}

export type MemorySource =
  | 'measureUserAgentSpecificMemory'
  | 'performance.memory'
  | 'none'

export interface MemorySnapshot {
  bytes: number
  source: MemorySource
  /** Platform-specific breakdown when `measureUserAgentSpecificMemory` is used. */
  rawBreakdown?: unknown
}

export interface PhaseSnapshot extends MemorySnapshot {
  /** Free-text phase label, e.g. "engine.trunk.ready". */
  phase: string
  /** performance.now() at recording time. */
  t: number
  /** Optional token count for the active task, used to label sweeps. */
  n?: number
}

export async function getMemorySnapshot(): Promise<MemorySnapshot> {
  const measure = (
    performance as unknown as { measureUserAgentSpecificMemory?: MeasureMemoryFn }
  ).measureUserAgentSpecificMemory
  if (typeof measure === 'function') {
    try {
      const r = await measure.call(performance)
      return {
        bytes: r.bytes,
        source: 'measureUserAgentSpecificMemory',
        rawBreakdown: r.breakdown,
      }
    } catch {
      // SecurityError when COI isn't established, or NotAllowedError when
      // the page is hidden. Fall through.
    }
  }
  const legacy = (
    performance as unknown as { memory?: LegacyPerformanceMemory }
  ).memory
  if (legacy) {
    return { bytes: legacy.usedJSHeapSize, source: 'performance.memory' }
  }
  return { bytes: 0, source: 'none' }
}

interface ProbeState {
  /** Toggles instrumentation. When false, recordPhase() is a no-op. */
  enabled: boolean
  /** Currently-active token count (set by the sweep buttons), surfaced
   *  alongside each phase snapshot for sweep correlation. */
  activeN: number | null
  /** Phase timeline. Grows append-only across one full prediction run;
   *  cleared by the panel's Reset button. */
  snapshots: PhaseSnapshot[]
  setEnabled: (e: boolean) => void
  setActiveN: (n: number | null) => void
  recordPhase: (phase: string) => Promise<void>
  clear: () => void
}

export const useMemoryProbe = create<ProbeState>((set, get) => ({
  enabled: false,
  activeN: null,
  snapshots: [],
  setEnabled: (enabled) => set({ enabled }),
  setActiveN: (n) => set({ activeN: n }),
  recordPhase: async (phase: string) => {
    if (!get().enabled) return
    const snap = await getMemorySnapshot()
    const t = performance.now()
    const n = get().activeN ?? undefined
    set((s) => ({
      snapshots: [...s.snapshots, { phase, t, n, ...snap }],
    }))
  },
  clear: () => set({ snapshots: [], activeN: null }),
}))

/** Convenience helper for non-React callers. */
export async function recordPhase(phase: string): Promise<void> {
  await useMemoryProbe.getState().recordPhase(phase)
}

/**
 * Synthetic FASTA generators for the sweep. Real proteins of these exact
 * lengths would force us to bundle PDB IDs and chase variation — for a
 * memory probe a repeating motif is fine. Boltz featurizes any 20-letter
 * amino-acid sequence; the trunk's attention is sequence-length-driven,
 * not sequence-content-driven.
 */
const PROBE_MOTIF = 'GASLVTMNRDEKPQHFYWICX'

export function syntheticFasta(n: number): string {
  const target = Math.max(1, Math.floor(n))
  let seq = ''
  while (seq.length < target) seq += PROBE_MOTIF
  return `>probe_n${target}\n${seq.slice(0, target)}`
}
