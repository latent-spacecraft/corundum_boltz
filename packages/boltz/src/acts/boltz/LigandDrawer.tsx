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
import { loadLigandBlob, type LigandBlob } from './featurizer/ligand'

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

// ─────────────────────────────────────────────────────────────────────────────
// CPK-ish coloring + radii for inline thumbnails. Only the elements we
// actually ship in the starter pack are enumerated; everything else falls
// back to a neutral gray.

const ELEMENT_COLOR: Record<string, string> = {
  H:  '#e0e0e0',
  C:  '#3a3a3a',
  N:  '#3050f8',
  O:  '#d33',
  S:  '#e7c32a',
  P:  '#ff8000',
  F:  '#90e050',
  CL: '#1ff01f',
  BR: '#a62929',
  I:  '#940094',
  FE: '#e06633',
  ZN: '#7d80b0',
  MG: '#88c050',
  CU: '#c88033',
  MN: '#9c7ac7',
  CA: '#3dff00',
  CO: '#f090a0',
}
const ELEMENT_RADIUS: Record<string, number> = {
  H: 0.4, C: 0.7, N: 0.65, O: 0.6, S: 0.85, P: 0.9, FE: 1.05,
}

const elementColor = (sym: string) => ELEMENT_COLOR[sym.toUpperCase()] ?? '#888'
const elementRadius = (sym: string) => ELEMENT_RADIUS[sym.toUpperCase()] ?? 0.75

// Cheap 2D projection: drop the axis of smallest variance. Good enough for
// porphyrins, flavins, nucleotides — anything roughly planar comes out
// face-on. Highly-3D ligands (carbohydrates, B12) get a usable but compressed
// silhouette.
function projectAxes(positions: [number, number, number][]): [number, number][] {
  const N = positions.length
  if (N === 0) return []
  if (N === 1) return [[0, 0]]
  let cx = 0, cy = 0, cz = 0
  for (const p of positions) { cx += p[0]; cy += p[1]; cz += p[2] }
  cx /= N; cy /= N; cz /= N
  let vx = 0, vy = 0, vz = 0
  for (const p of positions) {
    vx += (p[0] - cx) ** 2
    vy += (p[1] - cy) ** 2
    vz += (p[2] - cz) ** 2
  }
  const ranked = [
    [vx, 0] as const,
    [vy, 1] as const,
    [vz, 2] as const,
  ].sort((a, b) => b[0] - a[0])
  const ax = ranked[0][1]
  const ay = ranked[1][1]
  return positions.map((p) => [p[ax] - [cx, cy, cz][ax], p[ay] - [cx, cy, cz][ay]] as [number, number])
}

function renderThumbnailSvg(blob: LigandBlob, size = 64): string {
  const positions = blob.atoms.map((a) => a.ref_pos)
  const pts = projectAxes(positions)
  if (pts.length === 0) return ''
  if (pts.length === 1) {
    // Single-atom ion: render a generously sized circle at the centre.
    const a = blob.atoms[0]
    const color = elementColor(a.element_sym)
    return (
      `<svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">` +
      `<circle cx="${size / 2}" cy="${size / 2}" r="${size * 0.32}" fill="${color}" stroke="rgba(0,0,0,0.35)" stroke-width="1"/>` +
      `<text x="${size / 2}" y="${size / 2 + 4}" text-anchor="middle" font-family="monospace" font-size="${size * 0.35}" fill="white" font-weight="bold">${a.element_sym}</text>` +
      `</svg>`
    )
  }
  const xs = pts.map((p) => p[0])
  const ys = pts.map((p) => p[1])
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const span = Math.max(xMax - xMin, yMax - yMin, 0.001)
  const margin = 4
  const scale = (size - 2 * margin) / span
  const cxBox = (xMin + xMax) / 2, cyBox = (yMin + yMax) / 2
  const tx = (x: number) => size / 2 + (x - cxBox) * scale
  const ty = (y: number) => size / 2 + (y - cyBox) * scale

  const lines: string[] = []
  lines.push(`<svg viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg">`)
  // Bonds first, so atoms paint over the line endpoints.
  for (const b of blob.bonds) {
    const [x1, y1] = pts[b.i]
    const [x2, y2] = pts[b.j]
    const stroke = b.aromatic ? '#777' : '#555'
    const width = b.order >= 2 ? 1.3 : 0.9
    lines.push(`<line x1="${tx(x1).toFixed(2)}" y1="${ty(y1).toFixed(2)}" x2="${tx(x2).toFixed(2)}" y2="${ty(y2).toFixed(2)}" stroke="${stroke}" stroke-width="${width}" stroke-linecap="round"/>`)
  }
  for (let i = 0; i < blob.atoms.length; i++) {
    const a = blob.atoms[i]
    if (a.element === 1) continue // skip explicit hydrogens (CCD usually omits them anyway)
    const [x, y] = pts[i]
    const color = elementColor(a.element_sym)
    const r = elementRadius(a.element_sym) * 2.2
    lines.push(`<circle cx="${tx(x).toFixed(2)}" cy="${ty(y).toFixed(2)}" r="${r.toFixed(2)}" fill="${color}" stroke="rgba(0,0,0,0.35)" stroke-width="0.4"/>`)
  }
  lines.push('</svg>')
  return lines.join('')
}

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
        if (!cancelled) setSvg(renderThumbnailSvg(blob, size))
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
  const [open, setOpen] = useState(false)
  const insert = useLigandInsertSlot((s) => s.insert)

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
  }

  return (
    <details
      className="border-t"
      style={{ borderColor: 'var(--rule)', background: 'var(--card)' }}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary
        className="flex cursor-pointer select-none items-center justify-between px-6 py-2 font-mono text-[10px] uppercase tracking-widest"
        style={{ color: 'var(--ink-faded)' }}
      >
        <span>Ligand library</span>
        {index && (
          <span>
            {filtered.length}
            {filtered.length !== index.entries.length && (
              <span style={{ color: 'var(--ink-faded)' }}> / {index.entries.length}</span>
            )}
            {' shown'}
          </span>
        )}
      </summary>
      <div className="px-6 py-4">
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
                    className="truncate font-mono text-xs"
                    style={{ color: 'var(--foreground)' }}
                  >
                    {e.name}
                  </span>
                  <span
                    className="font-mono text-[10px] uppercase tracking-widest"
                    style={{ color: 'var(--ink-faded)' }}
                  >
                    {e.ccd} · {e.formula} · {e.n_atoms} atoms
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
    </details>
  )
}
