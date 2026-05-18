/**
 * WebGPU debug panel.
 *
 * Read-only adapter + features + limits dump, plus a per-graph EP toggle
 * hint. Mounted as a collapsed <details> drawer at the bottom of the app
 * so it's out of the way until you go looking. Critical limits are
 * flagged inline — `maxStorageBuffersPerShaderStage = 8` is the F12
 * trip-wire that bricked WebGPU early on (a 25-input Concat couldn't
 * compile; output silently zeroed; structure collapsed). Future export
 * work should decompose any Concat fan-in > 4 to stay portable.
 */
import { useEffect, useState } from 'react'
import { engine } from '@/engine/client'
import type { SessionHandle, IOMetadataEntry } from '@/engine/worker'

interface AdapterDump {
  vendor: string
  architecture: string
  device: string
  description: string
  features: string[]
  limits: Record<string, number>
}

const KEY_LIMITS = [
  'maxStorageBuffersPerShaderStage',
  'maxStorageBufferBindingSize',
  'maxComputeWorkgroupStorageSize',
  'maxBufferSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxBindGroups',
] as const

export function WebGpuDebug() {
  const [dump, setDump] = useState<AdapterDump | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const gpu = (navigator as Navigator & { gpu?: GPU }).gpu
        if (!gpu) {
          if (!cancelled) setError('navigator.gpu is undefined — WebGPU not exposed in this context.')
          return
        }
        const adapter = await gpu.requestAdapter()
        if (!adapter) {
          if (!cancelled) setError('No adapter returned by navigator.gpu.requestAdapter().')
          return
        }
        const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info ?? ({} as GPUAdapterInfo)
        const features: string[] = []
        adapter.features.forEach((f) => features.push(f))
        const limits: Record<string, number> = {}
        for (const k of Object.keys(adapter.limits as object)) {
          const v = (adapter.limits as unknown as Record<string, number>)[k]
          if (typeof v === 'number') limits[k] = v
        }
        if (!cancelled) {
          setDump({
            vendor: info.vendor ?? '(unknown)',
            architecture: info.architecture ?? '(unknown)',
            device: info.device ?? '(unknown)',
            description: info.description ?? '(unknown)',
            features: features.sort(),
            limits,
          })
        }
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <details
      className="border-t"
      style={{ borderColor: 'var(--rule)', background: 'var(--card)' }}
    >
      <summary
        className="cursor-pointer select-none px-6 py-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        WebGPU debug
      </summary>
      <div
        className="grid grid-cols-1 gap-6 px-6 py-4 md:grid-cols-3"
        style={{ color: 'var(--foreground)' }}
      >
        <Section title="Adapter">
          {error && (
            <p className="text-xs" style={{ color: 'var(--destructive)' }}>
              {error}
            </p>
          )}
          {dump && (
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
              {(
                [
                  ['Vendor', dump.vendor],
                  ['Architecture', dump.architecture],
                  ['Device', dump.device],
                  ['Description', dump.description],
                ] as const
              ).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt
                    className="uppercase tracking-widest text-[10px]"
                    style={{ color: 'var(--ink-faded)' }}
                  >
                    {k}
                  </dt>
                  <dd style={{ wordBreak: 'break-word' }}>{v}</dd>
                </div>
              ))}
            </dl>
          )}
        </Section>

        <Section title={`Key limits${dump ? ` · ${Object.keys(dump.limits).length} total` : ''}`}>
          {dump && (
            <ul className="space-y-1 font-mono text-xs">
              {KEY_LIMITS.map((k) => {
                const v = dump.limits[k]
                const flag =
                  k === 'maxStorageBuffersPerShaderStage' && typeof v === 'number' && v < 16
                return (
                  <li key={k} className="grid grid-cols-[1fr_auto] gap-3">
                    <span
                      className="truncate"
                      style={{ color: 'var(--ink-faded)' }}
                      title={k}
                    >
                      {k}
                    </span>
                    <span
                      style={{
                        color: flag ? 'var(--destructive)' : 'var(--foreground)',
                      }}
                    >
                      {v ?? '—'}
                      {flag ? ' ⚠' : ''}
                    </span>
                  </li>
                )
              })}
            </ul>
          )}
          {dump && (
            <p
              className="mt-3 text-[10px] leading-snug"
              style={{ color: 'var(--ink-faded)' }}
            >
              F12 trip-wire: a Concat with N inputs needs N storage buffers in one
              compute stage. WebGPU caps at <code>maxStorageBuffersPerShaderStage</code>
              (8 on most adapters); the F12 export decomposed wide Concats to fan-in 4
              for portability.
            </p>
          )}
        </Section>

        <Section title="Features">
          {dump && (
            <ul className="grid grid-cols-1 gap-0.5 font-mono text-xs">
              {dump.features.length === 0 && (
                <li style={{ color: 'var(--ink-faded)' }}>(none)</li>
              )}
              {dump.features.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          )}
        </Section>

        <div
          className="col-span-full pt-3 text-[10px] leading-snug"
          style={{ color: 'var(--ink-faded)', borderTop: '1px solid var(--rule)' }}
        >
          Execution-provider order is set in{' '}
          <code>src/acts/boltz/models.ts</code> as{' '}
          <code>executionProviders: ['webgpu', 'wasm']</code>. To force CPU/WASM
          for diagnostics, edit that array to <code>['wasm']</code> and reload —
          a hot toggle is parked for v0.2.
        </div>

        <div className="col-span-full">
          <PrecisionDiagnostics />
        </div>
      </div>
    </details>
  )
}

/**
 * Precision diagnostics — reads currently-loaded engine sessions and
 * shows their input/output dtypes + shapes. Drives the Phase B work of
 * threading non-fp32 boundaries through orchestrate.ts: this panel
 * tells you exactly which tensors need fp16 packing/unpacking, and
 * whether int8 graphs keep fp32 I/O (as quantize_dynamic typically does).
 */
function PrecisionDiagnostics() {
  const [sessions, setSessions] = useState<SessionHandle[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function refresh() {
    setBusy(true)
    setError(null)
    try {
      const list = await engine.listSessions()
      // Comlink returns a structured clone; for the readonly arrays we
      // copy them so React's identity diffing isn't fooled.
      setSessions([...list])
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void refresh()
  }, [])

  return (
    <section
      className="pt-3"
      style={{ borderTop: '1px solid var(--rule)' }}
    >
      <div className="mb-2 flex items-baseline justify-between">
        <h3
          className="font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          Precision · loaded sessions
        </h3>
        <button
          type="button"
          onClick={() => void refresh()}
          disabled={busy}
          style={{
            padding: '2px 8px',
            border: '1px solid var(--rule)',
            background: 'transparent',
            color: 'var(--ink-faded)',
            fontFamily: 'var(--font-mono)',
            fontSize: 10,
            letterSpacing: '0.15em',
            textTransform: 'uppercase',
            cursor: busy ? 'wait' : 'pointer',
            opacity: busy ? 0.5 : 1,
          }}
        >
          {busy ? '…' : 'refresh'}
        </button>
      </div>
      {error && (
        <p className="text-xs" style={{ color: 'var(--destructive)' }}>
          {error}
        </p>
      )}
      {!error && sessions.length === 0 && (
        <p
          className="font-mono text-[10px]"
          style={{ color: 'var(--ink-faded)' }}
        >
          No sessions loaded. Click <strong>Load engine</strong> in the Input
          pane to bring up the three graphs at the selected precision, then
          refresh.
        </p>
      )}
      {sessions.length > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {sessions.map((s) => (
            <SessionMetaCard key={s.id} session={s} />
          ))}
        </div>
      )}
    </section>
  )
}

function dtypeColor(dtype?: string): string {
  if (!dtype) return 'var(--ink-faded)'
  if (dtype === 'float32') return 'var(--foreground)'
  if (dtype === 'float16') return '#d97706' // amber — fp16 boundary, needs pack/unpack
  if (dtype === 'int8' || dtype === 'uint8') return '#7c3aed' // violet
  if (dtype === 'int64' || dtype === 'int32') return 'var(--ink-faded)'
  if (dtype === 'bool') return 'var(--ink-faded)'
  return 'var(--ink-faded)'
}

function formatShape(shape?: ReadonlyArray<number | string>): string {
  if (!shape || shape.length === 0) return '·'
  return '[' + shape.map((d) => (typeof d === 'number' ? String(d) : `${d}?`)).join(', ') + ']'
}

function SessionMetaCard({ session }: { session: SessionHandle }) {
  // Extract the (graph, precision) pair from manifest IDs of the form
  // `boltz2-{graph}-{precision}-v0.1`. Fall back to the raw modelId if
  // the pattern doesn't match (e.g. future model families).
  const m = session.modelId.match(/^boltz2-([^-]+)-([^-]+)-v/)
  const graph = m?.[1] ?? session.modelId
  const precision = m?.[2] ?? ''

  return (
    <div
      style={{
        border: '1px solid var(--rule)',
        padding: '8px 10px',
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
      }}
    >
      <div
        className="mb-2 flex items-baseline justify-between"
        style={{ letterSpacing: '0.15em', textTransform: 'uppercase' }}
      >
        <span style={{ color: 'var(--foreground)' }}>{graph}</span>
        <span style={{ color: 'var(--ink-faded)' }}>
          {precision} · {session.executionProvider}
        </span>
      </div>
      <MetaTable title="inputs" entries={session.inputMetadata} />
      <div style={{ height: 6 }} />
      <MetaTable title="outputs" entries={session.outputMetadata} />
    </div>
  )
}

function MetaTable({
  title,
  entries,
}: {
  title: string
  entries: readonly IOMetadataEntry[]
}) {
  return (
    <div>
      <div
        style={{
          letterSpacing: '0.15em',
          textTransform: 'uppercase',
          color: 'var(--ink-faded)',
          opacity: 0.7,
          marginBottom: 2,
        }}
      >
        {title} · {entries.length}
      </div>
      <ul className="grid grid-cols-[1fr_auto_auto] gap-x-2 gap-y-0.5">
        {entries.map((e) => (
          <li key={e.name} className="contents">
            <span
              style={{ color: 'var(--ink-faded)', wordBreak: 'break-all' }}
              title={e.name}
            >
              {e.name}
            </span>
            <span style={{ color: dtypeColor(e.dtype), textAlign: 'right' }}>
              {e.isTensor ? e.dtype : '(non-tensor)'}
            </span>
            <span
              style={{ color: 'var(--ink-faded)', textAlign: 'right' }}
              title={Array.isArray(e.shape) ? e.shape.join(' × ') : ''}
            >
              {formatShape(e.shape)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3
        className="mb-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        {title}
      </h3>
      {children}
    </section>
  )
}
