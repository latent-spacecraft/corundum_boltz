/**
 * mmCIF parser → Molero entity graph.
 *
 * Reads the `_atom_site` loop. Header lines `_atom_site.<field>` define
 * column order; data rows are whitespace-separated. We track:
 *   - group_PDB         → record type (ATOM / HETATM)
 *   - type_symbol       → element symbol
 *   - label_atom_id     → atom name (CA, N, O, etc.)
 *   - label_comp_id     → residue 3-letter code
 *   - label_asym_id     → chain label
 *   - label_seq_id      → residue sequence id within chain
 *   - Cartn_x/y/z       → coordinates (Å)
 *   - occupancy
 *   - B_iso_or_equiv    → temperature factor (= pLDDT × 100 for predictions)
 *
 * We don't (yet) handle: `_struct_conn` cross-links, `_chem_comp_bond`
 * residue templates, biological-assembly operators. Those land in slice
 * 1.2 alongside bond perception.
 *
 * Quoted strings: mmCIF allows single/double quotes for tokens with
 * whitespace, and `;`-delimited multi-line text. None of our target
 * columns ever contain whitespace, so a simple whitespace-split is
 * safe for the atom_site loop.
 */
import { atomicNumberFromSymbol } from '../chemistry/elements'
import type { RawAtomData, RawChainData, RawResidueData } from '../scene/scene'

export interface ParsedEntityGraph {
  atoms: RawAtomData
  residues: RawResidueData
  chains: RawChainData
}

/** Parse mmCIF text into an entity graph. Throws on malformed input. */
export function parseMmcif(text: string): ParsedEntityGraph {
  const lines = text.split(/\r?\n/)
  const out: ParsedEntityGraph = {
    atoms: emptyAtoms(),
    residues: emptyResidues(),
    chains: emptyChains(),
  }

  // Track residue/chain bookkeeping so we don't duplicate.
  // Key: `${asym_id}` → chainIndex
  const chainIndexByAsym = new Map<string, number>()
  // Key: `${chainIndex}|${comp_id}|${seq_id}|${ins_code}` → residueIndex
  const residueIndexByKey = new Map<string, number>()

  let i = 0
  while (i < lines.length) {
    const l = lines[i]
    if (l !== 'loop_' && l.trimStart() !== 'loop_') {
      i++
      continue
    }
    // Collect header lines that follow.
    const headers: string[] = []
    let j = i + 1
    while (j < lines.length) {
      const h = lines[j].trim()
      if (!h.startsWith('_')) break
      headers.push(h)
      j++
    }
    if (headers.length === 0 || !headers[0].startsWith('_atom_site.')) {
      i = j
      continue
    }
    const cols = atomSiteColumns(headers)
    if (cols.x < 0 || cols.y < 0 || cols.z < 0) {
      i = j
      continue
    }
    // Parse data rows.
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
      const tokens = trimmed.split(/\s+/)
      ingestAtomRow(tokens, cols, out, chainIndexByAsym, residueIndexByKey)
      j++
    }
    i = j
  }

  // Flush dynamic arrays into typed arrays for GPU upload.
  return finalizeBuffers(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// _atom_site column layout

interface AtomSiteCols {
  group: number
  typeSymbol: number
  atomId: number
  altLoc: number
  compId: number
  asymId: number
  entityId: number
  seqId: number
  insCode: number
  x: number
  y: number
  z: number
  occupancy: number
  bIso: number
  authSeqId: number
  authAsymId: number
  modelNum: number
}

function atomSiteColumns(headers: string[]): AtomSiteCols {
  const find = (suffix: string) => headers.indexOf(`_atom_site.${suffix}`)
  return {
    group:       find('group_PDB'),
    typeSymbol:  find('type_symbol'),
    atomId:      find('label_atom_id'),
    altLoc:      find('label_alt_id'),
    compId:      find('label_comp_id'),
    asymId:      find('label_asym_id'),
    entityId:    find('label_entity_id'),
    seqId:       find('label_seq_id'),
    insCode:     find('pdbx_PDB_ins_code'),
    x:           find('Cartn_x'),
    y:           find('Cartn_y'),
    z:           find('Cartn_z'),
    occupancy:   find('occupancy'),
    bIso:        find('B_iso_or_equiv'),
    authSeqId:   find('auth_seq_id'),
    authAsymId:  find('auth_asym_id'),
    modelNum:    find('pdbx_PDB_model_num'),
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row ingestion — appends to the dynamic arrays in `out`.

interface MutableGraph {
  atoms: {
    x: number[]
    y: number[]
    z: number[]
    atomicNumber: number[]
    atomNameId: number[]
    residueIndex: number[]
    chainIndex: number[]
    bfactor: number[]
    occupancy: number[]
    isHet: number[]
  }
  residues: {
    chainIndex: number[]
    compId: string[]
    seqId: number[]
    insCode: string[]
    atomStart: number[]
    atomEnd: number[]
  }
  chains: {
    asymId: string[]
    entityType: ('protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown')[]
    residueStart: number[]
    residueEnd: number[]
  }
}

function emptyAtoms() {
  return { count: 0 } as RawAtomData
}
function emptyResidues() {
  return { count: 0 } as RawResidueData
}
function emptyChains() {
  return { count: 0 } as RawChainData
}

function ingestAtomRow(
  tokens: string[],
  cols: AtomSiteCols,
  out: ParsedEntityGraph,
  chainIndexByAsym: Map<string, number>,
  residueIndexByKey: Map<string, number>,
) {
  // Lazily-built dynamic arrays — we attach to `out` once and grow.
  const mut = ensureMutable(out)

  // Multi-model files: keep only the first model for now. Slice 1.1
  // ships single-frame rendering; trajectory support lands in Phase 6.
  if (cols.modelNum >= 0) {
    const m = tokens[cols.modelNum]
    if (m && m !== '1' && m !== '.') {
      // Skip everything that isn't model 1.
      return
    }
  }

  const x = parseFloat(tokens[cols.x])
  const y = parseFloat(tokens[cols.y])
  const z = parseFloat(tokens[cols.z])
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return

  const symbol = cols.typeSymbol >= 0 ? tokens[cols.typeSymbol] : ''
  const atomicNumber = atomicNumberFromSymbol(symbol || guessElementFromName(
    cols.atomId >= 0 ? tokens[cols.atomId] : '',
  ))
  const atomName = cols.atomId >= 0 ? tokens[cols.atomId] : ''
  const compId = cols.compId >= 0 ? tokens[cols.compId] : 'UNK'
  const asymId = cols.asymId >= 0 ? tokens[cols.asymId] : 'A'
  const seqId = cols.seqId >= 0 ? parseInt(tokens[cols.seqId], 10) || 0 : 0
  const insCode = cols.insCode >= 0 ? tokens[cols.insCode] : '?'
  const occupancy = cols.occupancy >= 0 ? parseFloat(tokens[cols.occupancy]) : 1.0
  const bIso = cols.bIso >= 0 ? parseFloat(tokens[cols.bIso]) : 0.0
  const isHet = cols.group >= 0 && tokens[cols.group] === 'HETATM' ? 1 : 0

  // Chain bookkeeping.
  let chainIndex = chainIndexByAsym.get(asymId)
  if (chainIndex === undefined) {
    chainIndex = mut.chains.asymId.length
    chainIndexByAsym.set(asymId, chainIndex)
    mut.chains.asymId.push(asymId)
    mut.chains.entityType.push('unknown') // refined after all atoms parsed
    mut.chains.residueStart.push(mut.residues.compId.length)
    mut.chains.residueEnd.push(mut.residues.compId.length)
  }

  // Residue bookkeeping.
  const resKey = `${chainIndex}|${compId}|${seqId}|${insCode}`
  let residueIndex = residueIndexByKey.get(resKey)
  if (residueIndex === undefined) {
    residueIndex = mut.residues.compId.length
    residueIndexByKey.set(resKey, residueIndex)
    mut.residues.chainIndex.push(chainIndex)
    mut.residues.compId.push(compId)
    mut.residues.seqId.push(seqId)
    mut.residues.insCode.push(insCode === '.' || insCode === '?' ? '' : insCode)
    mut.residues.atomStart.push(mut.atoms.x.length)
    mut.residues.atomEnd.push(mut.atoms.x.length) // bumped after push
    mut.chains.residueEnd[chainIndex] = residueIndex + 1
  }

  // Atom row.
  mut.atoms.x.push(x)
  mut.atoms.y.push(y)
  mut.atoms.z.push(z)
  mut.atoms.atomicNumber.push(atomicNumber)
  mut.atoms.atomNameId.push(internAtomName(atomName))
  mut.atoms.residueIndex.push(residueIndex)
  mut.atoms.chainIndex.push(chainIndex)
  mut.atoms.bfactor.push(Number.isFinite(bIso) ? bIso : 0)
  mut.atoms.occupancy.push(Number.isFinite(occupancy) ? occupancy : 1)
  mut.atoms.isHet.push(isHet)
  mut.residues.atomEnd[residueIndex] = mut.atoms.x.length
}

// ─────────────────────────────────────────────────────────────────────────────
// Atom-name string interning. The valence-model + feature-flag lookups
// keyed by atom name need fast equality compares; ints beat strings.

const ATOM_NAME_TO_ID = new Map<string, number>()
const ATOM_NAME_TABLE: string[] = []

export function internAtomName(name: string): number {
  let id = ATOM_NAME_TO_ID.get(name)
  if (id === undefined) {
    id = ATOM_NAME_TABLE.length
    ATOM_NAME_TABLE.push(name)
    ATOM_NAME_TO_ID.set(name, id)
  }
  return id
}
export function atomNameFromId(id: number): string {
  return ATOM_NAME_TABLE[id] ?? ''
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback element guessing from PDB-style atom names. mmCIF should
// always have `type_symbol`, but defensive in case it doesn't.

function guessElementFromName(name: string): string {
  if (!name) return ''
  // PDB convention: 2-letter symbols start at column 13 (1-indexed); for
  // single-letter elements the symbol is at column 14. We don't have the
  // column position here so we use a heuristic on the trimmed name.
  const upper = name.toUpperCase()
  // Common 2-letter elements that appear in protein/ligand atom names.
  if (upper.startsWith('FE') || upper.startsWith('ZN') || upper.startsWith('MG') ||
      upper.startsWith('MN') || upper.startsWith('CA') && upper === 'CA' ||
      upper.startsWith('CL') || upper.startsWith('BR') || upper.startsWith('CU')) {
    // Special case: 'CA' is alpha carbon in residues, calcium in ions.
    // The caller passing residue 'UNK' / 'HOH' / ion residue would
    // disambiguate; here we err toward calcium only for bare 'CA'.
    if (upper === 'CA') return 'CA'
    return upper.slice(0, 2)
  }
  // Single-letter fallback: first alphabetic character.
  for (let i = 0; i < upper.length; i++) {
    const c = upper.charCodeAt(i)
    if (c >= 65 && c <= 90) return upper[i]
  }
  return ''
}

// ─────────────────────────────────────────────────────────────────────────────
// Finalization — convert dynamic arrays to typed arrays + classify chains.

const PROTEIN_RESIDUES = new Set([
  'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
  'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
  'SEC','PYL','MSE','UNK',
])
const RNA_RESIDUES = new Set(['A','G','C','U','N','I'])
const DNA_RESIDUES = new Set(['DA','DG','DC','DT','DN','DI'])
const WATER_RESIDUES = new Set(['HOH','WAT','H2O','DOD','D2O'])

function classifyChain(
  residues: MutableGraph['residues'],
  startResidue: number,
  endResidue: number,
): 'protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown' {
  let protein = 0, rna = 0, dna = 0, water = 0, other = 0
  for (let r = startResidue; r < endResidue; r++) {
    const id = residues.compId[r]
    if (PROTEIN_RESIDUES.has(id)) protein++
    else if (DNA_RESIDUES.has(id)) dna++
    else if (RNA_RESIDUES.has(id)) rna++
    else if (WATER_RESIDUES.has(id)) water++
    else other++
  }
  // Pick the dominant category. Ligands are typically single non-polymer residues.
  if (protein > 0 && protein >= Math.max(rna, dna, water, other)) return 'protein'
  if (rna > 0 && rna >= Math.max(protein, dna, water, other)) return 'rna'
  if (dna > 0 && dna >= Math.max(protein, rna, water, other)) return 'dna'
  if (water > 0 && water === (endResidue - startResidue)) return 'water'
  if (other > 0) return 'ligand'
  return 'unknown'
}

function ensureMutable(out: ParsedEntityGraph): MutableGraph {
  // Attached lazily — the typed-array `count: 0` forms in emptyAtoms/etc.
  // are stand-ins. We replace `out.atoms`, etc., with the typed-array
  // version in `finalizeBuffers`. During parsing we keep the mutable
  // dynamic arrays on a property of `out` keyed by `__mutable`.
  const M = (out as any).__mutable as MutableGraph | undefined
  if (M) return M
  const m: MutableGraph = {
    atoms: {
      x: [], y: [], z: [],
      atomicNumber: [],
      atomNameId: [],
      residueIndex: [],
      chainIndex: [],
      bfactor: [],
      occupancy: [],
      isHet: [],
    },
    residues: {
      chainIndex: [], compId: [], seqId: [], insCode: [],
      atomStart: [], atomEnd: [],
    },
    chains: { asymId: [], entityType: [], residueStart: [], residueEnd: [] },
  }
  ;(out as any).__mutable = m
  return m
}

function finalizeBuffers(out: ParsedEntityGraph): ParsedEntityGraph {
  const M = (out as any).__mutable as MutableGraph | undefined
  if (!M) return out
  // Classify chains now that all residues are known.
  for (let c = 0; c < M.chains.asymId.length; c++) {
    M.chains.entityType[c] = classifyChain(
      M.residues,
      M.chains.residueStart[c],
      M.chains.residueEnd[c],
    )
  }
  const atomCount = M.atoms.x.length
  const residueCount = M.residues.compId.length
  const chainCount = M.chains.asymId.length
  const atoms: RawAtomData = {
    count: atomCount,
    x: Float32Array.from(M.atoms.x),
    y: Float32Array.from(M.atoms.y),
    z: Float32Array.from(M.atoms.z),
    atomicNumber: Uint8Array.from(M.atoms.atomicNumber),
    atomNameId: Uint32Array.from(M.atoms.atomNameId),
    residueIndex: Uint32Array.from(M.atoms.residueIndex),
    chainIndex: Uint16Array.from(M.atoms.chainIndex),
    bfactor: Float32Array.from(M.atoms.bfactor),
    occupancy: Float32Array.from(M.atoms.occupancy),
    isHet: Uint8Array.from(M.atoms.isHet),
  }
  const residues: RawResidueData = {
    count: residueCount,
    chainIndex: Uint16Array.from(M.residues.chainIndex),
    compId: M.residues.compId,
    seqId: Int32Array.from(M.residues.seqId),
    insCode: M.residues.insCode,
    atomStart: Uint32Array.from(M.residues.atomStart),
    atomEnd: Uint32Array.from(M.residues.atomEnd),
  }
  const chains: RawChainData = {
    count: chainCount,
    asymId: M.chains.asymId,
    entityType: M.chains.entityType.slice(),
    residueStart: Uint32Array.from(M.chains.residueStart),
    residueEnd: Uint32Array.from(M.chains.residueEnd),
  }
  delete (out as any).__mutable
  return { atoms, residues, chains }
}
