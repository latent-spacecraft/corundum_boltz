/**
 * Smoke test for the multi-chain featurizer (Phase 1: protein).
 *
 * Two gates:
 *   1. Regression: a single-chain protein input must produce identical
 *      chain-id tensors to the pre-multi-chain refactor — all zeros for
 *      asym/entity/sym, residue_index = 0..N-1. Confirms the
 *      `featurize(sequence)` wrapper preserves the single-chain golden path.
 *   2. Multi-chain invariants:
 *        - homodimer [A, A]: asym_id splits at the boundary, entity_id
 *          collapses, sym_id increments per copy, residue_index restarts.
 *        - heterodimer [A, B]: entity_id distinct per chain, sym_id all 0.
 *        - NxN/AxN tensors scale with total N.
 *
 * Run: `npm run smoke:featurizer` from `packages/boltz`.
 */
import { featurize, featurizeChains } from '../src/acts/boltz/featurizer'

const SEQ_A = 'NLYIQWLKDGGPSSGRPPPS'
const SEQ_B = 'MKWVTFISLLFLFSSAYS'

function int64ToArray(d: BigInt64Array): number[] {
  return Array.from(d, (x) => Number(x))
}

function expect(name: string, ok: boolean, detail?: string) {
  const flag = ok ? '✓' : '✗'
  console.log(`  ${flag} ${name}${detail ? ' — ' + detail : ''}`)
  if (!ok) process.exitCode = 1
}

function arrEq(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function rangeArr(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i)
}

console.log('— Regression: single-chain wrapper —')
{
  const f = featurize(SEQ_A)
  const N = SEQ_A.length
  expect('N matches sequence length', f.N === N, `got ${f.N}`)
  const asym = int64ToArray(f.tensors['asym_id']!.data as BigInt64Array)
  const entity = int64ToArray(f.tensors['entity_id']!.data as BigInt64Array)
  const sym = int64ToArray(f.tensors['sym_id']!.data as BigInt64Array)
  const resIdx = int64ToArray(f.tensors['residue_index']!.data as BigInt64Array)
  const tokIdx = int64ToArray(f.tensors['token_index']!.data as BigInt64Array)
  expect('asym_id all 0', asym.every((v) => v === 0))
  expect('entity_id all 0', entity.every((v) => v === 0))
  expect('sym_id all 0', sym.every((v) => v === 0))
  expect('residue_index = 0..N-1', arrEq(resIdx, rangeArr(N)))
  expect('token_index = 0..N-1', arrEq(tokIdx, rangeArr(N)))
}

console.log('— Multi-chain: homodimer [A, A] —')
{
  const f = featurizeChains([
    { sequence: SEQ_A, type: 'protein' },
    { sequence: SEQ_A, type: 'protein' },
  ])
  const NA = SEQ_A.length
  const expectedAsym = [...Array(NA).fill(0), ...Array(NA).fill(1)]
  const expectedEntity = Array(2 * NA).fill(0)
  const expectedSym = [...Array(NA).fill(0), ...Array(NA).fill(1)]
  const expectedResIdx = [...rangeArr(NA), ...rangeArr(NA)]
  const expectedTokIdx = rangeArr(2 * NA)

  const asym = int64ToArray(f.tensors['asym_id']!.data as BigInt64Array)
  const entity = int64ToArray(f.tensors['entity_id']!.data as BigInt64Array)
  const sym = int64ToArray(f.tensors['sym_id']!.data as BigInt64Array)
  const resIdx = int64ToArray(f.tensors['residue_index']!.data as BigInt64Array)
  const tokIdx = int64ToArray(f.tensors['token_index']!.data as BigInt64Array)

  expect('N = 2 × len(A)', f.N === 2 * NA, `got ${f.N}`)
  expect('asym_id splits at chain boundary', arrEq(asym, expectedAsym))
  expect('entity_id collapses (homomer)', arrEq(entity, expectedEntity))
  expect('sym_id increments per copy', arrEq(sym, expectedSym))
  expect('residue_index restarts per chain', arrEq(resIdx, expectedResIdx))
  expect('token_index stays global', arrEq(tokIdx, expectedTokIdx))
}

console.log('— Multi-chain: heterodimer [A, B] —')
{
  const f = featurizeChains([
    { sequence: SEQ_A, type: 'protein' },
    { sequence: SEQ_B, type: 'protein' },
  ])
  const NA = SEQ_A.length
  const NB = SEQ_B.length
  const expectedAsym = [...Array(NA).fill(0), ...Array(NB).fill(1)]
  const expectedEntity = [...Array(NA).fill(0), ...Array(NB).fill(1)]
  const expectedSym = Array(NA + NB).fill(0)
  const expectedResIdx = [...rangeArr(NA), ...rangeArr(NB)]

  const asym = int64ToArray(f.tensors['asym_id']!.data as BigInt64Array)
  const entity = int64ToArray(f.tensors['entity_id']!.data as BigInt64Array)
  const sym = int64ToArray(f.tensors['sym_id']!.data as BigInt64Array)
  const resIdx = int64ToArray(f.tensors['residue_index']!.data as BigInt64Array)

  expect('N = len(A) + len(B)', f.N === NA + NB, `got ${f.N}`)
  expect('asym_id splits', arrEq(asym, expectedAsym))
  expect('entity_id distinct per chain', arrEq(entity, expectedEntity))
  expect('sym_id all 0 (no homomer)', arrEq(sym, expectedSym))
  expect('residue_index restarts per chain', arrEq(resIdx, expectedResIdx))
}

console.log('— RNA single chain —')
{
  const RNA_SEQ = 'GGGAAACCC'
  const f = featurizeChains([{ sequence: RNA_SEQ, type: 'rna' }])
  const N = RNA_SEQ.length
  expect('N matches', f.N === N)
  const molType = int64ToArray(f.tensors['mol_type']!.data as BigInt64Array)
  // From chain_types.json: PROTEIN=0, DNA=1, RNA=2
  expect('mol_type = 2 (RNA) across all tokens', molType.every((v) => v === 2))
  const refSpaceUid = int64ToArray(f.tensors['ref_space_uid']!.data as BigInt64Array)
  expect('ref_space_uid populated', refSpaceUid.some((v) => v > 0))
  // RNA backbone atoms (P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2', C1')
  // should write channels 5..16. Verify by sampling: a residue must light up
  // at least one channel in 5..16.
  const abf = int64ToArray(f.tensors['atom_backbone_feat']!.data as BigInt64Array)
  let nucleicHits = 0
  for (let a = 0; a < f.A; a++) {
    for (let ch = 5; ch <= 16; ch++) {
      if (abf[a * 17 + ch]) { nucleicHits++; break }
    }
  }
  expect('some atoms hit nucleic channels 5..16', nucleicHits > 0, `${nucleicHits} atoms`)
  let proteinHits = 0
  for (let a = 0; a < f.A; a++) {
    for (let ch = 1; ch <= 4; ch++) {
      if (abf[a * 17 + ch]) { proteinHits++; break }
    }
  }
  expect('no atoms hit protein channels 1..4', proteinHits === 0, `${proteinHits} false-positive`)
}

console.log('— DNA single chain —')
{
  const DNA_SEQ = 'ACGTACGT'
  const f = featurizeChains([{ sequence: DNA_SEQ, type: 'dna' }])
  const molType = int64ToArray(f.tensors['mol_type']!.data as BigInt64Array)
  expect('mol_type = 1 (DNA) across all tokens', molType.every((v) => v === 1))
  // DNA has no O2' — backbone atoms slot into channels 5..16 minus channel
  // for O2' (its slot in NUCLEIC_BACKBONE_INDEX is index 10 → channel 15).
  // We only assert at least one nucleic channel is lit.
  const abf = int64ToArray(f.tensors['atom_backbone_feat']!.data as BigInt64Array)
  let nucleicHits = 0
  for (let a = 0; a < f.A; a++) {
    for (let ch = 5; ch <= 16; ch++) {
      if (abf[a * 17 + ch]) { nucleicHits++; break }
    }
  }
  expect('some atoms hit nucleic channels', nucleicHits > 0)
}

console.log('— Mixed: protein + RNA + DNA —')
{
  const f = featurizeChains([
    { sequence: SEQ_A, type: 'protein' },
    { sequence: 'GGGAAACCC', type: 'rna' },
    { sequence: 'ACGTACGT', type: 'dna' },
  ])
  const N_total = SEQ_A.length + 9 + 8
  expect('total N', f.N === N_total, `got ${f.N}`)
  const molType = int64ToArray(f.tensors['mol_type']!.data as BigInt64Array)
  const protTokens = molType.slice(0, SEQ_A.length).every((v) => v === 0)
  const rnaTokens = molType.slice(SEQ_A.length, SEQ_A.length + 9).every((v) => v === 2)
  const dnaTokens = molType.slice(SEQ_A.length + 9).every((v) => v === 1)
  expect('protein section mol_type = 0', protTokens)
  expect('RNA section mol_type = 2', rnaTokens)
  expect('DNA section mol_type = 1', dnaTokens)
  const asym = int64ToArray(f.tensors['asym_id']!.data as BigInt64Array)
  expect('asym_id 3 distinct values', new Set(asym).size === 3)
  const entity = int64ToArray(f.tensors['entity_id']!.data as BigInt64Array)
  expect('entity_id 3 distinct values', new Set(entity).size === 3)
}

console.log('— Sanity: NxN tensors scale with total N —')
{
  const f = featurizeChains([
    { sequence: SEQ_A, type: 'protein' },
    { sequence: SEQ_B, type: 'protein' },
  ])
  const N = SEQ_A.length + SEQ_B.length
  const dt = f.tensors['disto_target']!
  expect(
    `disto_target shape = [1, ${N}, ${N}, 1, 64]`,
    dt.shape[0] === 1 && dt.shape[1] === N && dt.shape[2] === N && dt.shape[3] === 1 && dt.shape[4] === 64,
    `got ${JSON.stringify(dt.shape)}`,
  )
  const a2t = f.tensors['atom_to_token']!
  expect(
    `atom_to_token shape = [1, A, ${N}] with N=${N}`,
    a2t.shape[0] === 1 && a2t.shape[1] === f.A && a2t.shape[2] === N,
    `got ${JSON.stringify(a2t.shape)}`,
  )
}

console.log(process.exitCode ? '\nFAIL' : '\nOK')
