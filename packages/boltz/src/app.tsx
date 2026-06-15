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
        className="flex items-center justify-center gap-3 border-t px-4 py-1.5 text-[11px]"
        style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
      >
        <span>
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
        </span>
        <span style={{ color: 'var(--rule)' }}>│</span>
        <a
          href="https://github.com/latent-spacecraft/corundum_boltz"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 underline-offset-2 hover:underline transition-colors"
          style={{ color: 'var(--ink)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--oxblood)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--ink)')}
          title="Source on GitHub"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            <path d="M12 .5a11.5 11.5 0 0 0-3.635 22.412c.575.106.785-.25.785-.555 0-.273-.01-.997-.016-1.957-3.198.695-3.874-1.541-3.874-1.541-.523-1.33-1.278-1.685-1.278-1.685-1.044-.713.08-.699.08-.699 1.154.082 1.762 1.186 1.762 1.186 1.026 1.757 2.692 1.25 3.348.956.104-.744.402-1.25.73-1.538-2.553-.291-5.238-1.277-5.238-5.683 0-1.256.448-2.282 1.183-3.087-.119-.291-.513-1.463.112-3.05 0 0 .965-.31 3.16 1.18a10.96 10.96 0 0 1 5.756 0c2.194-1.49 3.158-1.18 3.158-1.18.626 1.587.232 2.759.114 3.05.737.805 1.182 1.831 1.182 3.087 0 4.417-2.69 5.389-5.253 5.674.413.355.78 1.057.78 2.131 0 1.538-.014 2.778-.014 3.155 0 .307.207.667.79.553A11.5 11.5 0 0 0 12 .5Z" />
          </svg>
          GitHub
        </a>
      </footer>

      <LigandDrawer />
    </div>
  )
}
