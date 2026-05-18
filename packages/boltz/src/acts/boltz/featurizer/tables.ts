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
import atomMapsJson from './tables/atom_maps.json'
import chainTypesJson from './tables/chain_types.json'
import chiralityJson from './tables/chirality.json'
import methodTypesJson from './tables/method_types.json'
import bondTypesJson from './tables/bond_types.json'
import geometryJson from './tables/geometry_constants.json'

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

/** Number of token classes (33 in current Boltz-2). */
export const NUM_TOKENS: number = tokensData.tokens.length

/** Boltz residue-type token id from a single-letter AA code (e.g. 'A' → ALA id). */
export function letterToTokenId(letter: string): number {
  const tok = tokensData.prot_letter_to_token[letter.toUpperCase()]
  if (tok === undefined) {
    // Per spec gotchas: any unknown letter maps to UNK.
    return tokensData.token_ids['UNK']
  }
  const id = tokensData.token_ids[tok]
  if (id === undefined) throw new Error(`tokens.json missing id for ${tok}`)
  return id
}

/** Boltz 3-letter residue name from a single-letter AA code (UNK fallback). */
export function letterToResName(letter: string): string {
  return tokensData.prot_letter_to_token[letter.toUpperCase()] ?? 'UNK'
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

export function residueTopology(resName: string): ResidueTopology {
  const t = topologyData[resName]
  if (!t) {
    const fallback = topologyData['UNK']
    if (!fallback) throw new Error(`Topology missing for ${resName} and no UNK fallback`)
    return fallback
  }
  return t
}

// ─────────────────────────────────────────────────────────────────────────────
// chain_types.json

interface ChainTypesTable {
  data: { chain_types: string[]; chain_type_ids: Record<string, number> }
}

const chainTypes = (chainTypesJson as unknown as ChainTypesTable).data
export const PROTEIN_CHAIN_TYPE_ID: number = chainTypes.chain_type_ids['PROTEIN']

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
 * Per protein_backbone_atom_names = ["N", "CA", "C", "O"] (length 4) +
 * nucleic_backbone_atom_names (length 12) + 1 "off the list" channel
 * (the +1 in SPEC.md = 4 + 12 + 1 = 17).
 *
 * For protein inputs only the first four indices are ever set.
 */
export const PROTEIN_BACKBONE_INDEX: Record<string, number> = {
  N: 0,
  CA: 1,
  C: 2,
  O: 3,
}
export const ATOM_BACKBONE_FEAT_DIM = 17
