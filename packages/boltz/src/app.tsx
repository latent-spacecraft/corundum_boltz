/**
 * App shell — Braun register, 100vh.
 *
 *   Desktop (≥ md, golden-ratio split)
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
 *   Mobile (< 768 px, stacked)
 *   ┌──────────────────┐
 *   │ Header           │
 *   │ Input (scrolls)  │   ~50% viewport
 *   ├──────────────────┤
 *   │                  │
 *   │ Canvas           │   ~50% viewport
 *   │                  │
 *   ├──────────────────┤
 *   │ Footer           │
 *   └──────────────────┘
 *
 * No page scroll: the outer container is exactly 100vh; the input pane
 * scrolls internally if its sections overflow. The breakpoint is
 * Tailwind's `md` (768 px) — phones in both portrait and landscape
 * (most under 900 px wide) stack; tablets and desktops split.
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
      <div className="flex min-h-0 flex-col overflow-hidden md:grid md:grid-cols-[38.2%_61.8%]">
        <aside
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden border-b md:border-b-0 md:border-r"
          style={{
            borderColor: 'var(--rule)',
            background: 'var(--card)',
          }}
        >
          <header
            className="flex shrink-0 items-center gap-3 border-b px-5 py-3 md:py-4"
            style={{ borderColor: 'var(--rule)' }}
          >
            <GemLogo size={40} title="Corundum" />
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
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <BoltzInput />
          </div>
        </aside>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
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
