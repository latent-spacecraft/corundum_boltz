/**
 * App shell — Braun register, golden-ratio split, 100vh.
 *
 *   ┌─────────────────────────────────────────────────────────┐
 *   │ Left pane (38.2%)    │  Toolbar  ─────────────────────  │
 *   │   Logo + wordmark    │  ╭─────────────────────────────╮ │
 *   │   Sequences          │  │                             │ │
 *   │   Ligands            │  │                             │ │
 *   │   Model              │  │         Canvas (Mol\*)      │ │
 *   │   Run                │  │                             │ │
 *   │   Sysdata            │  ╰─────────────────────────────╯ │
 *   ├─────────────────────────────────────────────────────────┤
 *   │             © Geoffrey Taghon 2026                      │
 *   └─────────────────────────────────────────────────────────┘
 *
 * No page scroll: the outer container is exactly 100vh and the left
 * pane scrolls internally if its content exceeds the viewport (rare
 * on standard sizes). Right pane is canvas-first; the toolbar floats
 * just above the viewport rectangle and the canvas claims everything
 * below it.
 */
import { BoltzCanvas, BoltzInput } from './acts/boltz/BoltzAct'
import { LigandDrawer } from './acts/boltz/LigandDrawer'
import { GemLogo } from './components/GemLogo'

export function App() {
  return (
    <div
      className="grid h-screen w-screen overflow-hidden"
      style={{
        gridTemplateRows: '1fr auto',
        background: 'var(--background)',
        color: 'var(--foreground)',
      }}
    >
      <div
        className="grid min-h-0 overflow-hidden"
        style={{ gridTemplateColumns: '38.2% 61.8%' }}
      >
        <aside
          className="flex h-full min-h-0 flex-col overflow-hidden border-r"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--card)',
          }}
        >
          <header
            className="flex shrink-0 items-center gap-3 px-5 py-4 border-b"
            style={{ borderColor: 'var(--rule)' }}
          >
            <GemLogo size={48} title="Corundum" />
            <h1
              className="text-base"
              style={{
                fontWeight: 800,
                letterSpacing: '-0.01em',
                lineHeight: 1,
              }}
            >
              Corundum
              <span style={{ color: 'var(--ink-faded)', margin: '0 0.4em', fontWeight: 300 }}>
                /
              </span>
              <span style={{ fontWeight: 400 }}>Boltz-2</span>
            </h1>
          </header>
          <div className="flex-1 min-h-0 overflow-y-auto">
            <BoltzInput />
          </div>
        </aside>

        <main className="flex h-full min-h-0 flex-col overflow-hidden">
          <BoltzCanvas />
        </main>
      </div>

      <footer
        className="flex items-center justify-center border-t px-4 py-1.5 text-[11px]"
        style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
      >
        ©{' '}
        <a
          href="https://www.linkedin.com/in/gtaghon"
          target="_blank"
          rel="noopener noreferrer"
          className="mx-1 underline-offset-2 hover:underline transition-colors"
          style={{ color: 'var(--ink)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--oxblood)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--ink)')}
        >
          Geoffrey Taghon
        </a>{' '}
        2026
      </footer>

      <LigandDrawer />
    </div>
  )
}
