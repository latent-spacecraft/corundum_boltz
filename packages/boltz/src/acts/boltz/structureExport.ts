/**
 * Structure export — mmCIF and PDB downloads from the loaded viewer state.
 *
 * Boltz writes mmCIF directly (see `writeMmcif`); that's the canonical output.
 * For PDB we parse the `_atom_site` loop of that mmCIF and re-emit each atom
 * as a fixed-column PDB ATOM/HETATM record. Mol*'s `to_mmCIF` produces a
 * loop with the same standard columns, so user-loaded PDB structures can be
 * round-tripped through Mol* → mmCIF → PDB without us writing a separate
 * PDB → PDB pass.
 *
 * The parser is intentionally narrow: single data block, a single _atom_site
 * loop, whitespace-separated tokens, no semicolon-delimited strings. That
 * covers everything our pipeline emits or Mol* re-emits. Exotic PDBx/mmCIF
 * dialects (multi-model, quoted text fields, hetero/molecular categories)
 * aren't in scope for alpha; if a user loads one and converts to PDB, the
 * structure path through Mol*-as-normaliser still gives us a clean loop.
 */

const PDB_HEADER_PREFIX = 'HEADER    PREDICTED STRUCTURE'

function padRight(s: string, width: number): string {
  return s.length >= width ? s.slice(0, width) : s + ' '.repeat(width - s.length)
}

function padLeft(s: string, width: number): string {
  return s.length >= width ? s.slice(0, width) : ' '.repeat(width - s.length) + s
}

function fixed(n: number, width: number, decimals: number): string {
  const s = (Number.isFinite(n) ? n : 0).toFixed(decimals)
  return padLeft(s, width)
}

/**
 * PDB atom-name column rule: a 4-char name fills cols 13-16; ≤3-char names
 * normally start at col 14 (col 13 left blank) unless the element symbol is
 * two characters (Ca, Fe, …), in which case the name starts at col 13. This
 * is how `CA` (carbon-alpha) and `Ca` (calcium) stay distinguishable.
 */
function formatAtomName(name: string, element: string): string {
  const n = name.replace(/"/g, '').slice(0, 4)
  if (n.length === 4) return n
  if (element.length === 2) return padRight(n, 4)
  return ' ' + padRight(n, 3)
}

interface AtomRow {
  group: 'ATOM' | 'HETATM'
  serial: number
  atomName: string
  altLoc: string
  resName: string
  chainId: string
  resSeq: number
  iCode: string
  x: number
  y: number
  z: number
  occupancy: number
  bfactor: number
  element: string
}

function pdbAtomLine(r: AtomRow): string {
  const recordName = padRight(r.group, 6)
  const serial = padLeft(String(r.serial % 100000), 5)
  const atomName = formatAtomName(r.atomName, r.element)
  const altLoc =
    r.altLoc === '.' || r.altLoc === '?' || !r.altLoc ? ' ' : r.altLoc.charAt(0)
  const resName = padLeft(r.resName.slice(0, 3), 3)
  const chainId = (r.chainId || ' ').charAt(0)
  // PDB resSeq is 4-digit; we wrap rather than corrupting columns.
  const resSeq = padLeft(String(((r.resSeq - 1) % 9999) + 1), 4)
  const iCode =
    r.iCode === '.' || r.iCode === '?' || !r.iCode ? ' ' : r.iCode.charAt(0)
  const x = fixed(r.x, 8, 3)
  const y = fixed(r.y, 8, 3)
  const z = fixed(r.z, 8, 3)
  const occ = fixed(r.occupancy, 6, 2)
  const b = fixed(r.bfactor, 6, 2)
  const element = padLeft(r.element.slice(0, 2), 2)
  // Columns 67-76 (10 spaces) sit between B-factor and element symbol.
  return `${recordName}${serial} ${atomName}${altLoc}${resName} ${chainId}${resSeq}${iCode}   ${x}${y}${z}${occ}${b}          ${element}`
}

/**
 * Walk the mmCIF text, locate the `_atom_site` loop, and emit one PDB
 * ATOM/HETATM record per row. The label (used in HEADER) is purely
 * cosmetic — viewers care about the ATOM block, not the header.
 */
export function mmcifToPdb(mmcif: string, label = 'STRUCTURE'): string {
  const lines = mmcif.split(/\r?\n/)
  let i = 0
  let cols: string[] | null = null

  // Scan for a `loop_` block whose first column name starts with `_atom_site.`
  while (i < lines.length) {
    const line = lines[i].trim()
    if (line === 'loop_') {
      const headerStart = i + 1
      let j = headerStart
      const block: string[] = []
      while (j < lines.length && lines[j].trim().startsWith('_')) {
        block.push(lines[j].trim())
        j++
      }
      if (block.length > 0 && block[0].startsWith('_atom_site.')) {
        cols = block
        i = j
        break
      }
      i = j
    } else {
      i++
    }
  }
  if (!cols) throw new Error('No _atom_site loop found in mmCIF input')

  const colIndex = (name: string) => cols!.indexOf('_atom_site.' + name)
  const c = {
    group: colIndex('group_PDB'),
    serial: colIndex('id'),
    element: colIndex('type_symbol'),
    atomName: colIndex('label_atom_id'),
    altLoc: colIndex('label_alt_id'),
    compId: colIndex('label_comp_id'),
    asymId: colIndex('label_asym_id'),
    seqId: colIndex('label_seq_id'),
    insCode: colIndex('pdbx_PDB_ins_code'),
    x: colIndex('Cartn_x'),
    y: colIndex('Cartn_y'),
    z: colIndex('Cartn_z'),
    occupancy: colIndex('occupancy'),
    bFactor: colIndex('B_iso_or_equiv'),
    authSeqId: colIndex('auth_seq_id'),
    authAsymId: colIndex('auth_asym_id'),
  }
  for (const required of ['group', 'serial', 'element', 'atomName', 'compId', 'asymId', 'seqId', 'x', 'y', 'z'] as const) {
    if (c[required] < 0) throw new Error(`mmCIF _atom_site is missing column for ${required}`)
  }

  const out: string[] = []
  out.push(padRight(`${PDB_HEADER_PREFIX} ${label.slice(0, 30).toUpperCase()}`, 80))

  // Track chain changes so we can emit TER between polymer chains — most
  // viewers don't strictly need it, but PyMOL/ChimeraX render bond inference
  // more cleanly when it's present.
  let prevChain: string | null = null
  let prevWasAtom = false
  let prevSerial = 0
  let prevResSeq = 0
  let prevResName = ''

  for (; i < lines.length; i++) {
    const raw = lines[i].trim()
    if (raw === '' || raw === '#') continue
    if (raw.startsWith('_') || raw === 'loop_' || raw.startsWith('data_')) break
    const tok = raw.split(/\s+/)
    if (tok.length < cols.length) continue

    const groupRaw = tok[c.group]
    const group: 'ATOM' | 'HETATM' = groupRaw === 'HETATM' ? 'HETATM' : 'ATOM'
    const serial = parseInt(tok[c.serial], 10) || 0
    const atomName = tok[c.atomName]
    const altLoc = c.altLoc >= 0 ? tok[c.altLoc] : '.'
    const resName = tok[c.compId]
    const chainSrc = c.authAsymId >= 0 ? tok[c.authAsymId] : tok[c.asymId]
    const chainId = chainSrc
    const seqRaw = c.authSeqId >= 0 ? tok[c.authSeqId] : tok[c.seqId]
    const resSeq = parseInt(seqRaw, 10) || 1
    const iCode = c.insCode >= 0 ? tok[c.insCode] : '?'
    const x = parseFloat(tok[c.x])
    const y = parseFloat(tok[c.y])
    const z = parseFloat(tok[c.z])
    const occupancy = c.occupancy >= 0 ? parseFloat(tok[c.occupancy]) || 1.0 : 1.0
    const bfactor = c.bFactor >= 0 ? parseFloat(tok[c.bFactor]) || 0.0 : 0.0
    const element = tok[c.element]

    if (prevWasAtom && group === 'ATOM' && prevChain !== null && prevChain !== chainId) {
      // TER record signals end of the previous polymer chain.
      const terSerial = padLeft(String((prevSerial + 1) % 100000), 5)
      const terResName = padLeft(prevResName.slice(0, 3), 3)
      const terChain = (prevChain || ' ').charAt(0)
      const terResSeq = padLeft(String(((prevResSeq - 1) % 9999) + 1), 4)
      out.push(`TER   ${terSerial}      ${terResName} ${terChain}${terResSeq}`)
    }

    out.push(
      pdbAtomLine({
        group,
        serial,
        atomName,
        altLoc,
        resName,
        chainId,
        resSeq,
        iCode,
        x,
        y,
        z,
        occupancy,
        bfactor,
        element,
      }),
    )

    prevChain = chainId
    prevWasAtom = group === 'ATOM'
    prevSerial = serial
    prevResSeq = resSeq
    prevResName = resName
  }

  out.push('END')
  return out.join('\n') + '\n'
}

/**
 * Trigger a browser download for a UTF-8 text string. The Blob URL is
 * revoked on the next microtask so we don't leak handles.
 */
export function downloadText(content: string, filename: string, mime = 'text/plain'): void {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  queueMicrotask(() => URL.revokeObjectURL(url))
}
