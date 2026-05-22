/**
 * Static lookup tables for the Boltz-2 featurizer.
 *
 * Source: `docs/featurizer_port/tables/*.json` (mirrored to `./tables/` for
 * static import). The data was dumped from `boltz/data/const.py` and the
 * RDKit mol pickles at `~/.boltz/mols/*.pkl`. Treat these as immutable —
 * regenerate them in boltz-dev if you find a bug, don't edit in place.
 *
 * Why static-import instead of fetch: the entire bundle is ~125 KB, much
 * smaller than the prediction noise floor, and inlining means no async
 * work at featurization time and no race against navigation.
 */

import tokensJson from './tables/tokens.json'
import topologyJson from './tables/residue_topology_protein.json'
import topologyRnaJson from './tables/residue_topology_rna.json'
import topologyDnaJson from './tables/residue_topology_dna.json'
import nucleicConstantsJson from './tables/nucleic_constants.json'
import atomMapsJson from './tables/atom_maps.json'
import chainTypesJson from './tables/chain_types.json'
import chiralityJson from './tables/chirality.json'
import methodTypesJson from './tables/method_types.json'
import bondTypesJson from './tables/bond_types.json'
import geometryJson from './tables/geometry_constants.json'

// ─────────────────────────────────────────────────────────────────────────────
// Chain type

export type ChainType = 'protein' | 'rna' | 'dna' | 'ligand'

// ─────────────────────────────────────────────────────────────────────────────
// tokens.json

interface TokensTable {
  data: {
    tokens: string[]
    token_ids: Record<string, number>
    prot_letter_to_token: Record<string, string>
  }
}

const tokensData = (tokensJson as unknown as TokensTable).data

interface NucleicConstantsTable {
  data: {
    rna_letter_to_token: Record<string, string>
    rna_token_to_letter: Record<string, string>
    dna_letter_to_token: Record<string, string>
    dna_token_to_letter: Record<string, string>
    nucleic_backbone_atom_names: string[]
    nucleic_backbone_atom_index: Record<string, number>
  }
}

const nucleicData = (nucleicConstantsJson as unknown as NucleicConstantsTable).data

/** Number of token classes (33 in current Boltz-2). */
export const NUM_TOKENS: number = tokensData.tokens.length

/**
 * Boltz residue-type token id from a single-letter code, dispatched by chain
 * type. Protein 'A' → ALA, RNA 'A' → A, DNA 'A' → DA. Unknown letters fall
 * back to the chain's "unknown" token. Ligand atoms always tokenise as UNK
 * (Boltz convention — every atom gets the same res_type).
 */
export function letterToTokenId(letter: string, chainType: ChainType = 'protein'): number {
  if (chainType === 'ligand') return UNK_TOKEN_ID
  const tok = letterToResName(letter, chainType)
  const id = tokensData.token_ids[tok]
  if (id !== undefined) return id
  const fallback = chainType === 'protein' ? 'UNK' : chainType === 'rna' ? 'N' : 'DN'
  return tokensData.token_ids[fallback]
}

/** Boltz residue name (3-letter for protein/DNA, 1-letter for RNA) from input letter. */
export function letterToResName(letter: string, chainType: ChainType = 'protein'): string {
  const upper = letter.toUpperCase()
  switch (chainType) {
    case 'protein': return tokensData.prot_letter_to_token[upper] ?? 'UNK'
    case 'rna':     return nucleicData.rna_letter_to_token[upper] ?? 'N'
    case 'dna':     return nucleicData.dna_letter_to_token[upper] ?? 'DN'
    case 'ligand':  return 'UNK' // unused — ligand featurization reads atoms from the blob.
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// residue_topology_protein.json

export interface ResidueAtom {
  name: string
  alt_name: string
  element: number      // atomic number
  element_sym: string
  charge: number
  chirality: string
  chirality_id: number
  leaving: boolean
  ref_pos: [number, number, number]
}

export interface ResidueBond {
  i: number
  j: number
  order: number
  type: string
  type_id: number
  aromatic: boolean
}

export interface ResidueTopology {
  atoms: ResidueAtom[]
  bonds: ResidueBond[]
  num_atoms: number
  center_atom_name: string
  center_atom_idx: number
  disto_atom_name: string
  disto_atom_idx: number
  /** Indices into `atoms[]` for the four protein backbone atoms in order [N, CA, C, O]. */
  backbone_atom_idx: number[]
}

interface TopologyTable {
  data: Record<string, ResidueTopology>
}

const topologyData = (topologyJson as unknown as TopologyTable).data
const topologyDataRna = (topologyRnaJson as unknown as TopologyTable).data
const topologyDataDna = (topologyDnaJson as unknown as TopologyTable).data

export function residueTopology(
  resName: string,
  chainType: ChainType = 'protein',
): ResidueTopology {
  if (chainType === 'ligand') {
    throw new Error('residueTopology() not applicable to ligand chains — load the per-CCD blob instead')
  }
  let table: Record<string, ResidueTopology>
  let fallback: string
  switch (chainType) {
    case 'protein': table = topologyData;    fallback = 'UNK'; break
    case 'rna':     table = topologyDataRna; fallback = 'N';   break
    case 'dna':     table = topologyDataDna; fallback = 'DN';  break
  }
  const t = table[resName] ?? table[fallback]
  if (!t) throw new Error(`Topology missing for ${resName} (chain type ${chainType})`)
  return t
}

// ─────────────────────────────────────────────────────────────────────────────
// chain_types.json

interface ChainTypesTable {
  data: { chain_types: string[]; chain_type_ids: Record<string, number> }
}

const chainTypes = (chainTypesJson as unknown as ChainTypesTable).data
export const PROTEIN_CHAIN_TYPE_ID: number    = chainTypes.chain_type_ids['PROTEIN']
export const RNA_CHAIN_TYPE_ID: number        = chainTypes.chain_type_ids['RNA']
export const DNA_CHAIN_TYPE_ID: number        = chainTypes.chain_type_ids['DNA']
export const NONPOLYMER_CHAIN_TYPE_ID: number = chainTypes.chain_type_ids['NONPOLYMER']

/** Ligand atoms are tokenised with the protein UNK token (Boltz convention). */
export const UNK_TOKEN_ID: number = tokensData.token_ids['UNK']

export function chainTypeId(type: ChainType): number {
  switch (type) {
    case 'protein': return PROTEIN_CHAIN_TYPE_ID
    case 'rna':     return RNA_CHAIN_TYPE_ID
    case 'dna':     return DNA_CHAIN_TYPE_ID
    case 'ligand':  return NONPOLYMER_CHAIN_TYPE_ID
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// atom_maps.json (informational; topology already carries center/disto idx)

interface AtomMapsTable {
  data: {
    res_to_center_atom: Record<string, string>
    res_to_disto_atom: Record<string, string>
  }
}

const atomMaps = (atomMapsJson as unknown as AtomMapsTable).data
export function centerAtomName(resName: string): string {
  return atomMaps.res_to_center_atom[resName] ?? 'CA'
}
export function distoAtomName(resName: string): string {
  return atomMaps.res_to_disto_atom[resName] ?? 'CA'
}

// ─────────────────────────────────────────────────────────────────────────────
// chirality.json

interface ChiralityTable {
  data: {
    chirality_types: string[]
    chirality_type_ids: Record<string, number>
  }
}

const chiralityData = (chiralityJson as unknown as ChiralityTable).data
export const CHIRALITY_TYPE_IDS = chiralityData.chirality_type_ids

// ─────────────────────────────────────────────────────────────────────────────
// method_types.json

interface MethodTypesTable {
  data: {
    method_types: string[]
    method_types_ids: Record<string, number>
  }
}

const methodTypesData = (methodTypesJson as unknown as MethodTypesTable).data
export const METHOD_TYPE_OTHER_ID: number =
  methodTypesData.method_types_ids['other'] ??
  methodTypesData.method_types_ids['OTHER'] ??
  0

// ─────────────────────────────────────────────────────────────────────────────
// bond_types.json

interface BondTypesTable {
  data: {
    bond_types: string[]
    bond_type_ids: Record<string, number>
  }
}

const bondTypesData = (bondTypesJson as unknown as BondTypesTable).data
export const BOND_TYPE_SINGLE_ID: number = bondTypesData.bond_type_ids['SINGLE']

// ─────────────────────────────────────────────────────────────────────────────
// geometry_constants.json

interface GeometryTable {
  data: { num_elements: number; [k: string]: unknown }
}

const geometryData = (geometryJson as unknown as GeometryTable).data
export const NUM_ELEMENTS: number = geometryData.num_elements

// ─────────────────────────────────────────────────────────────────────────────
// Constants derived from spec

/** Atoms-per-window (`W`) for the diffusion graph's atom encoder. */
export const ATOM_WINDOW_W = 32

/**
 * atom_backbone_feat one-hot layout:
 *   channel 0 = sidechain / off-list (any atom that isn't a tracked backbone)
 *   channels 1..4   = protein backbone N, CA, C, O
 *   channels 5..16  = nucleic backbone P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1'
 * Total = 1 + 4 + 12 = 17.
 *
 * Index maps below are 0-based offsets *within their group*; the +1 (and
 * +4 for nucleic) one-hot offset lives in `atomBackboneChannel()` to keep
 * the source-of-truth alignment with featurizerv2.py easy to read.
 */
export const PROTEIN_BACKBONE_INDEX: Record<string, number> = {
  N: 0,
  CA: 1,
  C: 2,
  O: 3,
}
export const NUCLEIC_BACKBONE_INDEX: Record<string, number> = Object.fromEntries(
  nucleicData.nucleic_backbone_atom_names.map((name, i) => [name, i]),
)
export const ATOM_BACKBONE_FEAT_DIM = 17

/**
 * Return the atom_backbone_feat one-hot channel for `atomName` on a chain of
 * `chainType`. 0 if the atom is not on the chain's backbone list.
 */
export function atomBackboneChannel(atomName: string, chainType: ChainType): number {
  if (chainType === 'protein') {
    const idx = PROTEIN_BACKBONE_INDEX[atomName]
    return idx !== undefined ? idx + 1 : 0
  }
  if (chainType === 'rna' || chainType === 'dna') {
    // Both share the 12-atom nucleic backbone vocabulary.
    const idx = NUCLEIC_BACKBONE_INDEX[atomName]
    return idx !== undefined ? idx + 5 : 0
  }
  // Ligand atoms have no backbone vocabulary — every atom is "sidechain".
  return 0
}
