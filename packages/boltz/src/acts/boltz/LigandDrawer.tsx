/**
 * Ligand drawer — searchable library of CCD cofactors and small molecules.
 *
 * Mounted as a collapsed `<details>` slot at the bottom of the app (next to
 * the WebGPU debug drawer). On open, fetches `/ccd/index.json` (~4 KB, 34
 * entries today). Search runs case-insensitive substring against every
 * entry's CCD code + name + synonyms.
 *
 * Thumbnails are rendered inline as SVG from each ligand's blob — no Mol*
 * involved, fast and lazy-loaded per row when the row enters the viewport.
 * Click a row → appends `>lig_<CCD> ligand\n<CCD>` to the FASTA textarea so
 * the user can immediately fire a co-prediction.
 */
import { useEffect, useMemo, useRef, useState } from 'react'
import { create } from 'zustand'
import { loadLigandBlob } from './featurizer/ligand'
import { renderLigandSvg } from './ligandSvg'

// ─────────────────────────────────────────────────────────────────────────────
// Index types

interface LigandIndexEntry {
  ccd: string
  name: string
  synonyms: string[]
  formula: string
  n_atoms: number
}

interface LigandIndex {
  version: string
  entries: LigandIndexEntry[]
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared zustand handle for "append to FASTA textarea". The drawer doesn't
// own the FASTA state — BoltzAct does — so we expose a tiny setter slot
// the BoltzAct sub-tree registers on mount.

interface FastaInsertSlot {
  insert: ((ccd: string) => void) | null
  setInsert: (fn: ((ccd: string) => void) | null) => void
}

export const useLigandInsertSlot = create<FastaInsertSlot>((set) => ({
  insert: null,
  setInsert: (fn) => set({ insert: fn }),
}))

// Open state for the cofactor picker — exposed as a zustand store so the
// "Browse cofactor library…" link in the input pane can open it from
// elsewhere in the tree.
interface DrawerOpenState {
  open: boolean
  setOpen: (open: boolean) => void
}
export const useLigandDrawer = create<DrawerOpenState>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}))

// ─────────────────────────────────────────────────────────────────────────────
// Per-row thumbnail with lazy load.

function LigandThumbnail({ ccd, size = 64 }: { ccd: string; size?: number }) {
  const [svg, setSvg] = useState<string | null>(null)
  const [visible, setVisible] = useState(false)
  const ref = useRef<HTMLDivElement | null>(null)

  // Defer blob fetch until the row enters the viewport — keeps initial drawer
  // open snappy even if the user has a hundred ligands.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (visible) return
    const obs = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          obs.disconnect()
        }
      },
      { rootMargin: '120px' },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [visible])

  useEffect(() => {
    if (!visible) return
    let cancelled = false
    loadLigandBlob(ccd)
      .then((blob) => {
        if (!cancelled) setSvg(renderLigandSvg(blob, size))
      })
      .catch(() => {
        if (!cancelled) setSvg('')
      })
    return () => {
      cancelled = true
    }
  }, [ccd, size, visible])

  return (
    <div
      ref={ref}
      style={{
        width: size,
        height: size,
        background: 'var(--paper-mottle)',
        flexShrink: 0,
      }}
      dangerouslySetInnerHTML={svg ? { __html: svg } : undefined}
    />
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// Drawer

export function LigandDrawer() {
  const [index, setIndex] = useState<LigandIndex | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const open = useLigandDrawer((s) => s.open)
  const setOpen = useLigandDrawer((s) => s.setOpen)
  const insert = useLigandInsertSlot((s) => s.insert)

  // ESC closes; click on backdrop closes.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  // Load the index once, when the drawer is first opened. Cheap (~4 KB) but
  // no point fetching it before the user looks.
  useEffect(() => {
    if (!open || index) return
    fetch('/ccd/index.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((data: LigandIndex) => setIndex(data))
      .catch((e) => setError((e as Error).message))
  }, [open, index])

  const filtered = useMemo(() => {
    if (!index) return []
    const q = query.trim().toLowerCase()
    if (!q) return index.entries
    return index.entries.filter((e) => {
      if (e.ccd.toLowerCase().includes(q)) return true
      if (e.name.toLowerCase().includes(q)) return true
      for (const s of e.synonyms) {
        if (s.toLowerCase().includes(q)) return true
      }
      if (e.formula.toLowerCase().includes(q)) return true
      return false
    })
  }, [index, query])

  const handleClick = (ccd: string) => {
    if (insert) insert(ccd)
    setOpen(false)
  }

  if (!open) return null

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Cofactor library"
      onClick={(e) => {
        if (e.target === e.currentTarget) setOpen(false)
      }}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(20, 20, 20, 0.35)',
        zIndex: 50,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4rem',
      }}
    >
      <div
        className="flex max-h-full max-w-3xl flex-col overflow-hidden border"
        style={{ borderColor: 'var(--rule)', background: 'var(--card)' }}
      >
        <div
          className="flex shrink-0 items-center justify-between border-b px-5 py-3"
          style={{ borderColor: 'var(--rule)', color: 'var(--ink-faded)' }}
        >
          <span className="text-sm font-medium" style={{ color: 'var(--ink)' }}>
            Cofactor library
          </span>
          <div className="flex items-center gap-4 text-xs">
            {index && (
              <span>
                {filtered.length}
                {filtered.length !== index.entries.length && (
                  <span style={{ color: 'var(--ink-faded)' }}> / {index.entries.length}</span>
                )}
                {' shown'}
              </span>
            )}
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close cofactor library"
              className="text-lg leading-none"
              style={{ color: 'var(--ink)' }}
            >
              ×
            </button>
          </div>
        </div>
        <div className="overflow-auto px-5 py-4">
        {error && (
          <p className="font-mono text-xs" style={{ color: 'var(--destructive)' }}>
            Failed to load /ccd/index.json: {error}
          </p>
        )}
        {!error && (
          <input
            type="search"
            placeholder="Search by name, CCD code, or formula — heme, vitamin b12, ATP, FE…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="mb-3 w-full border px-2 py-1.5 font-mono text-xs"
            style={{
              borderColor: 'var(--rule)',
              background: 'var(--paper-mottle)',
              color: 'var(--ink)',
            }}
          />
        )}
        {!error && index && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-2">
            {filtered.map((e) => (
              <button
                key={e.ccd}
                type="button"
                onClick={() => handleClick(e.ccd)}
                title={`Append >lig_${e.ccd} ligand\\n${e.ccd} to the FASTA input`}
                disabled={!insert}
                className="flex cursor-pointer items-center gap-2.5 border p-2 text-left transition-colors"
                style={{
                  borderColor: 'var(--rule)',
                  background: 'transparent',
                  color: 'var(--ink)',
                  opacity: insert ? 1 : 0.5,
                }}
              >
                <LigandThumbnail ccd={e.ccd} />
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span
                    className="truncate text-xs"
                    style={{ color: 'var(--foreground)' }}
                  >
                    {e.name}
                  </span>
                  <span
                    className="text-[10px]"
                    style={{ color: 'var(--ink-faded)' }}
                  >
                    <span className="font-mono uppercase tracking-wide">{e.ccd}</span>
                    {' · '}
                    {e.formula}
                    {' · '}
                    {e.n_atoms} atoms
                  </span>
                </div>
              </button>
            ))}
            {filtered.length === 0 && (
              <p
                className="col-span-full font-mono text-xs"
                style={{ color: 'var(--ink-faded)' }}
              >
                No matches.
              </p>
            )}
          </div>
        )}
        </div>
      </div>
    </div>
  )
}
