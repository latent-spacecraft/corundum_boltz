/**
 * Ligand (NONPOLYMER) blob types and loader.
 *
 * Each CCD ligand ships as `public/ccd/<CODE>.json`, generated Python-side by
 * scripts/python/extract_ligand_blob.py. The blob carries the atom list, bond
 * list, and the five geometry-constraint groups Boltz-2 derives from an RDKit
 * Mol (rdkit-bounds / chiral atoms / stereo bonds / planar bonds / planar
 * rings 5+6). Indices are 0-based within the ligand; the featurizer adds the
 * global atom offset when concatenating into the prediction.
 *
 * Loaded async from `/ccd/<CODE>.json` at the same time we resolve the FASTA
 * input (before featurizeChains is called).
 */
import type { ResidueAtom, ResidueBond } from './tables'

export interface LigandRdkitBound {
  i: number
  j: number
  is_bond: boolean
  is_angle: boolean
  upper: number
  lower: number
}

export interface LigandChiralAtom {
  /** 4 atom indices: 3 ranked neighbours + the chiral centre. */
  atoms: number[]
  is_reference: boolean
  is_r: boolean
}

export interface LigandStereoBond {
  /** 4 atom indices: start-neighbour, start, end, end-neighbour. */
  atoms: number[]
  is_check: boolean
  is_e: boolean
}

export interface LigandPlanarBond {
  /** 6 atom indices spanning the planar sp² centres. */
  atoms: number[]
}
export interface LigandPlanarRing5 { atoms: number[] }
export interface LigandPlanarRing6 { atoms: number[] }

export interface LigandBlob {
  ccd: string
  num_atoms: number
  atoms: ResidueAtom[]
  bonds: ResidueBond[]
  rdkit_bounds: LigandRdkitBound[]
  chiral_atoms: LigandChiralAtom[]
  stereo_bonds: LigandStereoBond[]
  planar_bonds: LigandPlanarBond[]
  planar_rings_5: LigandPlanarRing5[]
  planar_rings_6: LigandPlanarRing6[]
}

interface WrappedBlob {
  _doc?: string
  data: LigandBlob
}

const blobCache = new Map<string, Promise<LigandBlob>>()

/**
 * Fetch a ligand blob by CCD code. Cached by code so repeated chains
 * referencing the same ligand share a single network request.
 *
 * Returns the unwrapped `LigandBlob`. Throws if the CCD isn't shipped under
 * `public/ccd/` (caller should catch and surface as a UI parse error).
 */
export async function loadLigandBlob(ccd: string): Promise<LigandBlob> {
  const code = ccd.toUpperCase()
  const cached = blobCache.get(code)
  if (cached) return cached
  const p = fetchAndUnwrap(code)
  blobCache.set(code, p)
  try {
    return await p
  } catch (e) {
    // Don't keep failed promises in the cache — retry on next click.
    blobCache.delete(code)
    throw e
  }
}

async function fetchAndUnwrap(code: string): Promise<LigandBlob> {
  const res = await fetch(`/ccd/${code}.json`)
  if (!res.ok) {
    throw new Error(`Unknown ligand '${code}' (HTTP ${res.status} from /ccd/${code}.json)`)
  }
  const wrapped = (await res.json()) as WrappedBlob
  return wrapped.data
}
