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

/**
 * Seed the blob cache under `code` with an already-built blob. Lets the SMILES
 * path register its server-computed blob so the normal `loadLigandBlob(code)`
 * featurization path resolves it without a /ccd fetch.
 */
export function cacheLigandBlob(code: string, blob: LigandBlob): void {
  blobCache.set(code.toUpperCase(), Promise.resolve(blob))
}

export interface SmilesBlobResult {
  blob: LigandBlob
  /** Stable short code (= blob.ccd); also the exported residue name. */
  code: string
  /** RDKit canonical SMILES — the molecule's identity. */
  canonicalSmiles: string
}

// SMILES results are cached by the *input* SMILES string so repeated renders
// (chip thumbnail + inspector + predict) reuse one round-trip. Distinct input
// spellings of the same molecule each pay one request but collapse to the same
// code (the server canonicalises), so entity-dedup downstream still works.
const smilesCache = new Map<string, Promise<SmilesBlobResult>>()

const SMILES_ENDPOINT = '/__smiles_to_blob'

/**
 * Preprocess a user SMILES into a Boltz ligand blob via the server-side
 * endpoint (RDKit ETKDG + distance-geometry bounds — can't run in-browser).
 * The returned blob is also seeded into the by-code cache so a subsequent
 * `loadLigandBlob(result.code)` resolves locally.
 *
 * Throws with a UI-surfaceable message on parse failure or a missing endpoint
 * (e.g. a static deploy with no preprocessing backend wired up).
 */
export async function loadLigandBlobFromSmiles(smiles: string): Promise<SmilesBlobResult> {
  const key = smiles.trim()
  if (key.length === 0) throw new Error('Empty SMILES')
  const cached = smilesCache.get(key)
  if (cached) return cached
  const p = postSmiles(key)
  smilesCache.set(key, p)
  try {
    return await p
  } catch (e) {
    smilesCache.delete(key) // don't pin a failed request — let the user retry
    throw e
  }
}

async function postSmiles(smiles: string): Promise<SmilesBlobResult> {
  let res: Response
  try {
    res = await fetch(SMILES_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles }),
    })
  } catch {
    throw new Error('SMILES preprocessing endpoint unreachable — is the dev server running?')
  }
  let body: { ok?: boolean; error?: string; code?: string; canonical_smiles?: string; data?: LigandBlob }
  try {
    body = (await res.json()) as typeof body
  } catch {
    throw new Error(`SMILES preprocessing failed (HTTP ${res.status}, non-JSON response)`)
  }
  if (!res.ok || !body.ok || !body.data || !body.code) {
    throw new Error(body.error || `SMILES preprocessing failed (HTTP ${res.status})`)
  }
  const blob = body.data
  cacheLigandBlob(body.code, blob)
  return { blob, code: body.code, canonicalSmiles: body.canonical_smiles ?? smiles }
}
