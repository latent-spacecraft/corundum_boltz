/**
 * Browser-only "bring your own SMILES" ligand path.
 *
 * Kept separate from ligand.ts so that ligand.ts stays isomorphic (the node
 * smoke harness compiles the featurizer, including ligand.ts, under a DOM-less
 * tsconfig). This module pulls in the RDKit WASM loader (which touches
 * window/document), so it must only ever be imported from browser code.
 */
import { cacheLigandBlob, type LigandBlob, type SmilesBlobResult } from './ligand'
import { getRDKit } from './rdkit'

// SMILES results are cached by the *input* SMILES string so repeated renders
// (chip thumbnail + inspector + predict) reuse one build. Distinct input
// spellings of the same molecule each pay one build but collapse to the same
// code (RDKit canonicalises), so entity-dedup downstream still works.
const smilesCache = new Map<string, Promise<SmilesBlobResult>>()

/**
 * Preprocess a user SMILES into a Boltz ligand blob, fully in-browser via the
 * RDKit WASM op `get_boltz_blob` (ETKDGv3 embedding + distance-geometry bounds
 * + the six constraint groups). No server round-trip. The returned blob is also
 * seeded into the by-code cache so a subsequent `loadLigandBlob(result.code)`
 * resolves locally.
 *
 * Throws with a UI-surfaceable message on parse/embed failure or if the RDKit
 * module can't be loaded.
 */
export async function loadLigandBlobFromSmiles(smiles: string): Promise<SmilesBlobResult> {
  const key = smiles.trim()
  if (key.length === 0) throw new Error('Empty SMILES')
  const cached = smilesCache.get(key)
  if (cached) return cached
  const p = buildSmilesBlob(key)
  smilesCache.set(key, p)
  try {
    return await p
  } catch (e) {
    smilesCache.delete(key) // don't pin a failed build — let the user retry
    throw e
  }
}

/**
 * Stable short ligand code from canonical SMILES. Mirrors the Python
 * `derive_code`: "L" + first 5 hex chars of SHA1(canonical), upper-cased.
 */
async function deriveCode(canonical: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-1', new TextEncoder().encode(canonical))
  const hex = Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
  return 'L' + hex.slice(0, 5).toUpperCase()
}

async function buildSmilesBlob(smiles: string): Promise<SmilesBlobResult> {
  const RDKit = await getRDKit()
  let parsed: { ok?: boolean; error?: string; canonical_smiles?: string; data?: LigandBlob }
  try {
    parsed = JSON.parse(RDKit.get_boltz_blob(smiles)) as typeof parsed
  } catch (e) {
    throw new Error(`SMILES preprocessing failed (RDKit error: ${(e as Error).message})`)
  }
  if (!parsed.ok || !parsed.data) {
    throw new Error(parsed.error || 'SMILES preprocessing failed')
  }
  const canonicalSmiles = parsed.canonical_smiles ?? smiles
  // get_boltz_blob leaves data.ccd empty; the code is a SHA1 of the canonical
  // SMILES (kept JS-side so the WASM op needs no crypto).
  const code = await deriveCode(canonicalSmiles)
  const blob: LigandBlob = { ...parsed.data, ccd: code }
  cacheLigandBlob(code, blob)
  return { blob, code, canonicalSmiles }
}
