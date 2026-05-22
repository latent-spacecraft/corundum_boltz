/**
 * Minimal atom-position extractor for PDB and mmCIF text.
 *
 * The Three.js refractive shell needs atom XYZs to build its Gaussian
 * density field. Rather than couple to Mol*'s internal Unit conformation
 * arrays, we parse the structure text directly — both formats expose
 * coords in straightforward ASCII.
 *
 * PDB: fixed-width columns 31-38 / 39-46 / 47-54 on ATOM and HETATM
 * lines (1-indexed per spec; 0-indexed slice 30:38 / 38:46 / 46:54).
 *
 * mmCIF: `_atom_site` loop. Header lines `_atom_site.<field>` define
 * column order; data rows are whitespace-separated. We locate the
 * `Cartn_x/y/z` column indices and read each ATOM/HETATM row.
 */
import type { StructureFormat } from './MolViewer'

export function extractAtomPositions(
  data: string,
  format: StructureFormat,
): Float32Array {
  if (format === 'pdb') return parsePdb(data)
  return parseMmcif(data)
}

function parsePdb(text: string): Float32Array {
  const out: number[] = []
  // Avoid splitting huge text twice; iterate via newline indices.
  let lineStart = 0
  for (let i = 0; i <= text.length; i++) {
    if (i !== text.length && text.charCodeAt(i) !== 0x0a /* \n */) continue
    const line = text.slice(lineStart, i)
    lineStart = i + 1
    // ATOM / HETATM record types only. The cheap prefix check beats slice+startsWith.
    if (
      (line.charCodeAt(0) === 0x41 /* A */ && line.startsWith('ATOM')) ||
      (line.charCodeAt(0) === 0x48 /* H */ && line.startsWith('HETATM'))
    ) {
      const x = parseFloat(line.slice(30, 38))
      const y = parseFloat(line.slice(38, 46))
      const z = parseFloat(line.slice(46, 54))
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        out.push(x, y, z)
      }
    }
  }
  return new Float32Array(out)
}

function parseMmcif(text: string): Float32Array {
  // Walk lines; locate the _atom_site loop header block; record the
  // column index of Cartn_x/y/z and group_PDB; then parse rows until
  // the next loop_/data_ marker or non-data line.
  const lines = text.split(/\r?\n/)
  const out: number[] = []

  let i = 0
  while (i < lines.length) {
    const l = lines[i]
    if (l === 'loop_' || l.trimStart() === 'loop_') {
      // Collect header lines that follow
      const headers: string[] = []
      let j = i + 1
      while (j < lines.length) {
        const h = lines[j].trim()
        if (!h.startsWith('_')) break
        headers.push(h)
        j++
      }
      // Only act if this is an _atom_site loop
      if (headers.length > 0 && headers[0].startsWith('_atom_site.')) {
        const colGroup = headers.indexOf('_atom_site.group_PDB')
        const colX = headers.indexOf('_atom_site.Cartn_x')
        const colY = headers.indexOf('_atom_site.Cartn_y')
        const colZ = headers.indexOf('_atom_site.Cartn_z')
        if (colX < 0 || colY < 0 || colZ < 0) {
          i = j
          continue
        }
        // Parse data rows until the loop ends
        while (j < lines.length) {
          const row = lines[j]
          const trimmed = row.trim()
          if (
            trimmed === '' ||
            trimmed.startsWith('loop_') ||
            trimmed.startsWith('_') ||
            trimmed.startsWith('data_') ||
            trimmed.startsWith('#')
          ) {
            break
          }
          // Whitespace-tokenize. mmCIF allows quoted strings with spaces
          // but coordinate values never contain whitespace, so a simple
          // split is safe for the columns we care about.
          const tokens = trimmed.split(/\s+/)
          // If we know which column flags ATOM/HETATM, gate on it; else accept all rows.
          if (colGroup >= 0) {
            const g = tokens[colGroup]
            if (g !== 'ATOM' && g !== 'HETATM') {
              j++
              continue
            }
          }
          const x = parseFloat(tokens[colX])
          const y = parseFloat(tokens[colY])
          const z = parseFloat(tokens[colZ])
          if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
            out.push(x, y, z)
          }
          j++
        }
        i = j
        continue
      }
      i = j
      continue
    }
    i++
  }
  return new Float32Array(out)
}
