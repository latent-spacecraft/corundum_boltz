/**
 * PDB parser → Molero entity graph.
 *
 * Fixed-width column layout per the PDB spec (1-indexed columns
 * 31-38 / 39-46 / 47-54 for x/y/z, 77-78 for element symbol). Reads
 * ATOM and HETATM records; skips alternate models (only model 1 is kept).
 *
 * Most missing pieces (CONECT records for explicit bonds, HEADER /
 * COMPND / SEQRES, MODRES, multi-model trajectories) are deferred to
 * slice 1.2+. For Phase 1.1 we need atoms + residues + chains + B-factor,
 * which the ATOM/HETATM rows carry directly.
 */
import { atomicNumberFromSymbol } from '../chemistry/elements'
import type { RawAtomData, RawChainData, RawResidueData } from '../scene/scene'
import { internAtomName, type ParsedEntityGraph } from './mmcif'

export function parsePdb(text: string): ParsedEntityGraph {
  const chainIndexByAsym = new Map<string, number>()
  const residueIndexByKey = new Map<string, number>()

  const xs: number[] = []
  const ys: number[] = []
  const zs: number[] = []
  const atomicNumbers: number[] = []
  const atomNameIds: number[] = []
  const residueIndices: number[] = []
  const chainIndices: number[] = []
  const bfactors: number[] = []
  const occupancies: number[] = []
  const isHets: number[] = []

  const resChainIndex: number[] = []
  const resCompId: string[] = []
  const resSeqId: number[] = []
  const resInsCode: string[] = []
  const resAtomStart: number[] = []
  const resAtomEnd: number[] = []

  const chAsymId: string[] = []
  const chEntityType: ('protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown')[] = []
  const chResidueStart: number[] = []
  const chResidueEnd: number[] = []

  // Walk lines via newline-index scan (avoids splitting huge strings twice).
  let inAlternateModel = false
  let lineStart = 0
  for (let i = 0; i <= text.length; i++) {
    if (i !== text.length && text.charCodeAt(i) !== 0x0a) continue
    const line = text.slice(lineStart, i)
    lineStart = i + 1

    if (line.length === 0) continue
    const head = line.charCodeAt(0)

    // MODEL record — only keep model 1.
    if (head === 0x4d /* M */ && line.startsWith('MODEL')) {
      const modelNum = parseInt(line.slice(10, 14).trim(), 10) || 1
      inAlternateModel = modelNum !== 1
      continue
    }
    if (head === 0x45 /* E */ && line.startsWith('ENDMDL')) {
      inAlternateModel = false
      continue
    }
    if (inAlternateModel) continue

    // ATOM / HETATM only.
    const isAtom = head === 0x41 && line.startsWith('ATOM')
    const isHet = head === 0x48 && line.startsWith('HETATM')
    if (!isAtom && !isHet) continue
    if (line.length < 54) continue // can't contain coords

    const atomName = line.slice(12, 16).trim()
    const compId = line.slice(17, 20).trim() || 'UNK'
    const asymId = line.slice(21, 22).trim() || 'A'
    const seqId = parseInt(line.slice(22, 26).trim(), 10) || 0
    const insCode = line.slice(26, 27).trim()
    const x = parseFloat(line.slice(30, 38))
    const y = parseFloat(line.slice(38, 46))
    const z = parseFloat(line.slice(46, 54))
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue
    const occupancy = line.length >= 60 ? parseFloat(line.slice(54, 60)) : 1.0
    const bIso = line.length >= 66 ? parseFloat(line.slice(60, 66)) : 0.0
    const elementSymbol = line.length >= 78
      ? line.slice(76, 78).trim()
      : guessElementFromName(atomName)
    const atomicNumber = atomicNumberFromSymbol(elementSymbol)

    let chainIndex = chainIndexByAsym.get(asymId)
    if (chainIndex === undefined) {
      chainIndex = chAsymId.length
      chainIndexByAsym.set(asymId, chainIndex)
      chAsymId.push(asymId)
      chEntityType.push('unknown')
      chResidueStart.push(resCompId.length)
      chResidueEnd.push(resCompId.length)
    }

    const resKey = `${chainIndex}|${compId}|${seqId}|${insCode}`
    let residueIndex = residueIndexByKey.get(resKey)
    if (residueIndex === undefined) {
      residueIndex = resCompId.length
      residueIndexByKey.set(resKey, residueIndex)
      resChainIndex.push(chainIndex)
      resCompId.push(compId)
      resSeqId.push(seqId)
      resInsCode.push(insCode)
      resAtomStart.push(xs.length)
      resAtomEnd.push(xs.length)
      chResidueEnd[chainIndex] = residueIndex + 1
    }

    xs.push(x)
    ys.push(y)
    zs.push(z)
    atomicNumbers.push(atomicNumber)
    atomNameIds.push(internAtomName(atomName))
    residueIndices.push(residueIndex)
    chainIndices.push(chainIndex)
    bfactors.push(Number.isFinite(bIso) ? bIso : 0)
    occupancies.push(Number.isFinite(occupancy) ? occupancy : 1)
    isHets.push(isHet ? 1 : 0)
    resAtomEnd[residueIndex] = xs.length
  }

  // Classify chains by dominant residue type.
  for (let c = 0; c < chAsymId.length; c++) {
    chEntityType[c] = classifyChain(resCompId, chResidueStart[c], chResidueEnd[c])
  }

  const atoms: RawAtomData = {
    count: xs.length,
    x: Float32Array.from(xs),
    y: Float32Array.from(ys),
    z: Float32Array.from(zs),
    atomicNumber: Uint8Array.from(atomicNumbers),
    atomNameId: Uint32Array.from(atomNameIds),
    residueIndex: Uint32Array.from(residueIndices),
    chainIndex: Uint16Array.from(chainIndices),
    bfactor: Float32Array.from(bfactors),
    occupancy: Float32Array.from(occupancies),
    isHet: Uint8Array.from(isHets),
  }
  const residues: RawResidueData = {
    count: resCompId.length,
    chainIndex: Uint16Array.from(resChainIndex),
    compId: resCompId,
    seqId: Int32Array.from(resSeqId),
    insCode: resInsCode,
    atomStart: Uint32Array.from(resAtomStart),
    atomEnd: Uint32Array.from(resAtomEnd),
  }
  const chains: RawChainData = {
    count: chAsymId.length,
    asymId: chAsymId,
    entityType: chEntityType,
    residueStart: Uint32Array.from(chResidueStart),
    residueEnd: Uint32Array.from(chResidueEnd),
  }
  return { atoms, residues, chains }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers duplicated locally (mmcif.ts has private versions). Tiny enough
// that the duplication is cheaper than another import-cycle headache.

const PROTEIN_RESIDUES = new Set([
  'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
  'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
  'SEC','PYL','MSE','UNK',
])
const RNA_RESIDUES = new Set(['A','G','C','U','N','I'])
const DNA_RESIDUES = new Set(['DA','DG','DC','DT','DN','DI'])
const WATER_RESIDUES = new Set(['HOH','WAT','H2O','DOD','D2O'])

function classifyChain(
  compIds: string[],
  startResidue: number,
  endResidue: number,
): 'protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown' {
  let protein = 0, rna = 0, dna = 0, water = 0, other = 0
  for (let r = startResidue; r < endResidue; r++) {
    const id = compIds[r]
    if (PROTEIN_RESIDUES.has(id)) protein++
    else if (DNA_RESIDUES.has(id)) dna++
    else if (RNA_RESIDUES.has(id)) rna++
    else if (WATER_RESIDUES.has(id)) water++
    else other++
  }
  if (protein > 0 && protein >= Math.max(rna, dna, water, other)) return 'protein'
  if (rna > 0 && rna >= Math.max(protein, dna, water, other)) return 'rna'
  if (dna > 0 && dna >= Math.max(protein, rna, water, other)) return 'dna'
  if (water > 0 && water === (endResidue - startResidue)) return 'water'
  if (other > 0) return 'ligand'
  return 'unknown'
}

function guessElementFromName(name: string): string {
  if (!name) return ''
  const upper = name.toUpperCase()
  // Standard amino-acid backbone atoms — single-element names.
  if (upper === 'CA' || upper === 'C' || upper === 'N' || upper === 'O') return upper[0]
  // First two chars if alphabetic — covers FE, ZN, MG, etc.
  const first = upper.charCodeAt(0)
  const second = upper.charCodeAt(1)
  if (first >= 65 && first <= 90 && second >= 65 && second <= 90) return upper.slice(0, 2)
  if (first >= 65 && first <= 90) return upper[0]
  return ''
}
