/**
 * Memory probe debug panel.
 *
 * Switchable instrumentation that records `(phase, bytes, t)` snapshots
 * across one prediction run. Mounted as a `<details>` drawer at the
 * bottom of the app, alongside WebGpuDebug.
 *
 * Workflow for probing the Safari OOM at confidence-head load:
 *
 *   1. Open the drawer, click "Enable instrumentation".
 *   2. Click a sweep button (e.g. "Inject N=300 FASTA") — this fills the
 *      Predict input box with a synthetic sequence of that length.
 *   3. Load engine + Predict normally. Every phase boundary records a
 *      snapshot.
 *   4. Crash or success, the table shows where memory went. Reset to
 *      probe a different size.
 *
 * The instrumentation is off by default — calling `recordPhase()` while
 * disabled is a no-op, so leaving the hooks live in production is free.
 *
 * The "Pre-warm baseline" button takes a single snapshot before anything
 * runs, so the deltas-from-start column has a reference point.
 */
import { useEffect, useState } from 'react'
import {
  useMemoryProbe,
  getMemorySnapshot,
  syntheticFasta,
} from '@/engine/memoryProbe'
import { useBoltz } from '@/acts/boltz/BoltzAct'
import { formatBytes } from '@/engine/fetcher'

const PROBE_SIZES = [50, 150, 300, 500] as const

export function MemoryProbe() {
  const { enabled, snapshots, setEnabled, setActiveN, clear } = useMemoryProbe()
  const setFasta = useBoltz((s) => s.setFasta)
  const [baselineBytes, setBaselineBytes] = useState<number | null>(null)
  const [baselineSource, setBaselineSource] = useState<string>('')

  // Whenever the user clicks a sweep size, push the FASTA into the input
  // box AND tag the active token count so subsequent snapshots are
  // labeled with N.
  const injectSweep = (n: number) => {
    setActiveN(n)
    setFasta(syntheticFasta(n))
  }

  const takeBaseline = async () => {
    const s = await getMemorySnapshot()
    setBaselineBytes(s.bytes)
    setBaselineSource(s.source)
  }

  // Auto-refresh the baseline figure every 2s while the panel is open
  // and instrumentation is OFF — gives the user a sense of background
  // memory before they kick off a run. When enabled flips on, we stop
  // (the recorded snapshots take over).
  useEffect(() => {
    if (enabled) return
    let cancelled = false
    const tick = async () => {
      if (cancelled) return
      const s = await getMemorySnapshot()
      if (cancelled) return
      setBaselineBytes(s.bytes)
      setBaselineSource(s.source)
    }
    void tick()
    const id = window.setInterval(tick, 2000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [enabled])

  const peakBytes = snapshots.reduce((m, s) => Math.max(m, s.bytes), 0)
  const startBytes = snapshots[0]?.bytes ?? baselineBytes ?? 0
  const startT = snapshots[0]?.t ?? 0

  return (
    <details
      className="border-t"
      style={{ borderColor: 'var(--rule)', background: 'var(--card)' }}
    >
      <summary
        className="cursor-pointer select-none px-6 py-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        Memory probe
        {enabled && (
          <span style={{ color: 'var(--oxblood)', marginInlineStart: 12 }}>
            ● recording — {snapshots.length} snapshots
          </span>
        )}
      </summary>
      <div className="flex flex-col gap-3 px-6 py-4">
        {/* Controls row */}
        <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] uppercase tracking-widest">
          <button
            type="button"
            onClick={() => setEnabled(!enabled)}
            className="border px-2 py-1"
            style={{
              borderColor: enabled ? 'var(--oxblood)' : 'var(--rule)',
              color: enabled ? 'var(--oxblood)' : 'var(--ink)',
              background: enabled ? 'var(--paper-mottle)' : 'transparent',
            }}
          >
            {enabled ? 'Recording ●' : 'Enable instrumentation'}
          </button>
          <button
            type="button"
            onClick={clear}
            className="border px-2 py-1"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink)' }}
          >
            Clear trace
          </button>
          <button
            type="button"
            onClick={takeBaseline}
            className="border px-2 py-1"
            style={{ borderColor: 'var(--rule)', color: 'var(--ink)' }}
          >
            Snapshot baseline
          </button>

          <span style={{ color: 'var(--rule)' }}>│</span>

          <span style={{ color: 'var(--ink-faded)' }}>Inject sweep:</span>
          {PROBE_SIZES.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => injectSweep(n)}
              className="border px-2 py-1"
              style={{ borderColor: 'var(--rule)', color: 'var(--ink)' }}
            >
              N={n}
            </button>
          ))}
        </div>

        {/* Baseline readout */}
        {baselineBytes !== null && (
          <p
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: 'var(--ink-faded)' }}
          >
            Baseline · {formatBytes(baselineBytes)} via {baselineSource || 'none'}
            {baselineSource === 'none' &&
              ' — neither performance.measureUserAgentSpecificMemory nor performance.memory is available'}
            {peakBytes > 0 && ` · peak this run ${formatBytes(peakBytes)}`}
          </p>
        )}

        {/* Trace */}
        {snapshots.length > 0 && (
          <div
            className="overflow-x-auto border"
            style={{ borderColor: 'var(--rule)' }}
          >
            <table
              className="w-full font-mono text-[11px]"
              style={{ color: 'var(--ink)' }}
            >
              <thead style={{ background: 'var(--paper-mottle)' }}>
                <tr>
                  <th className="px-2 py-1 text-left">Phase</th>
                  <th className="px-2 py-1 text-right">N</th>
                  <th className="px-2 py-1 text-right">t (ms)</th>
                  <th className="px-2 py-1 text-right">Bytes</th>
                  <th className="px-2 py-1 text-right">Δ from start</th>
                  <th className="px-2 py-1 text-right">Δ from prev</th>
                  <th className="px-2 py-1 text-left">Source</th>
                </tr>
              </thead>
              <tbody>
                {snapshots.map((s, i) => {
                  const prev = i > 0 ? snapshots[i - 1] : null
                  const dStart = s.bytes - startBytes
                  const dPrev = prev ? s.bytes - prev.bytes : 0
                  return (
                    <tr key={i} style={{ borderTop: '1px solid var(--rule)' }}>
                      <td className="px-2 py-1">{s.phase}</td>
                      <td className="px-2 py-1 text-right">{s.n ?? '—'}</td>
                      <td className="px-2 py-1 text-right">
                        {(s.t - startT).toFixed(0)}
                      </td>
                      <td className="px-2 py-1 text-right">
                        {formatBytes(s.bytes)}
                      </td>
                      <td
                        className="px-2 py-1 text-right"
                        style={{
                          color:
                            dStart > 0 ? 'var(--oxblood)' : 'var(--ink-faded)',
                        }}
                      >
                        {dStart >= 0 ? '+' : ''}
                        {formatBytes(dStart)}
                      </td>
                      <td
                        className="px-2 py-1 text-right"
                        style={{
                          color:
                            dPrev > 32 * 1024 * 1024
                              ? 'var(--destructive)'
                              : 'var(--ink-faded)',
                        }}
                      >
                        {dPrev >= 0 ? '+' : ''}
                        {formatBytes(dPrev)}
                      </td>
                      <td
                        className="px-2 py-1 text-[9px] uppercase tracking-widest"
                        style={{ color: 'var(--ink-faded)' }}
                      >
                        {s.source.replace('measureUserAgentSpecificMemory', 'ua-mem')}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        <p
          className="font-mono text-[9px] leading-snug"
          style={{ color: 'var(--ink-faded)' }}
        >
          Workflow: enable instrumentation, inject a sweep size, run Load
          engine + Predict normally. Each phase records a snapshot.
          measureUserAgentSpecificMemory() runs an internal GC before
          measuring, so calls take ~50-200 ms — phase-boundary placement
          (not per-step) keeps the overhead negligible during sampling.
        </p>
      </div>
    </details>
  )
}
