/**
 * App shell — three columns wrap the Boltz act, with the WebGPU debug
 * panel as a collapsible drawer at the bottom.
 *
 *   ┌───────────┬──────────────────────────────┬───────────┐
 *   │  Input    │           Canvas             │  Output   │
 *   │  (FASTA,  │      (Mol* viewer)           │  (stats,  │
 *   │  engine,  │                              │  header)  │
 *   │  predict) │                              │           │
 *   └───────────┴──────────────────────────────┴───────────┘
 *   ▶ WebGPU debug
 */
import { BoltzCanvas, BoltzInput, BoltzOutput } from './acts/boltz/BoltzAct'
import { WebGpuDebug } from './debug/WebGpuDebug'

export function App() {
  return (
    <div className="flex min-h-screen flex-col">
      <header
        className="border-b px-6 py-3"
        style={{ borderColor: 'var(--rule)' }}
      >
        <div className="flex items-baseline justify-between">
          <h1
            className="text-lg font-medium tracking-tight"
            style={{
              fontFamily: 'var(--font-display)',
              color: 'var(--foreground)',
            }}
          >
            Corundum · Boltz
          </h1>
          <span
            className="font-mono text-[10px] uppercase tracking-widest"
            style={{ color: 'var(--ink-faded)' }}
          >
            Browser-native Boltz-2 · v0.1
          </span>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 gap-px lg:grid-cols-[340px_1fr_320px]" style={{ background: 'var(--rule)' }}>
        <Pane title="Input" ordinal="I">
          <BoltzInput />
        </Pane>
        <Pane title="Canvas" ordinal="II">
          <BoltzCanvas />
        </Pane>
        <Pane title="Output" ordinal="III">
          <BoltzOutput />
        </Pane>
      </main>

      <WebGpuDebug />

      <footer
        className="border-t px-6 py-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
      >
        Boltz-2 weights © Wohlwend et al. (MIT) · served by HuggingFace · runs on your device
      </footer>
    </div>
  )
}

function Pane({
  title,
  ordinal,
  children,
}: {
  title: string
  ordinal: string
  children: React.ReactNode
}) {
  return (
    <section className="flex flex-col" style={{ background: 'var(--card)' }}>
      <header
        className="flex items-baseline justify-between border-b px-3 py-2"
        style={{ borderColor: 'var(--rule)' }}
      >
        <h2
          className="text-sm font-medium"
          style={{
            fontFamily: 'var(--font-display)',
            color: 'var(--foreground)',
          }}
        >
          {title}
        </h2>
        <span
          className="font-mono text-[10px] uppercase tracking-widest"
          style={{ color: 'var(--ink-faded)' }}
        >
          {ordinal}
        </span>
      </header>
      <div
        className="flex-1 overflow-auto p-4 text-sm"
        style={{ color: 'var(--foreground)' }}
      >
        {children}
      </div>
    </section>
  )
}
