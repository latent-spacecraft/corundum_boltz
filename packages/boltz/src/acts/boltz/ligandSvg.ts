/**
 * Inline 2D ligand renderer — atoms + bonds projected to a flat SVG straight
 * from a LigandBlob's ref_pos/bonds. No Mol*, no layout engine; cheap enough
 * to render dozens of thumbnails or a live SMILES-inspector preview.
 *
 * Shared by the cofactor drawer (per-row thumbnails) and the ligand input's
 * SMILES inspector (confirm the entered SMILES is the intended molecule before
 * committing to a prediction).
 */
import type { LigandBlob } from './featurizer/ligand'

// CPK-ish coloring + radii. Only the elements we actually ship/expect are
// enumerated; everything else falls back to neutral gray.
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

export const elementColor = (sym: string) => ELEMENT_COLOR[sym.toUpperCase()] ?? '#888'
export const elementRadius = (sym: string) => ELEMENT_RADIUS[sym.toUpperCase()] ?? 0.75

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

export function renderLigandSvg(blob: LigandBlob, size = 64): string {
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
