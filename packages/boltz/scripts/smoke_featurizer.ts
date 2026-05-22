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
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { featurize, featurizeChains } from '../src/acts/boltz/featurizer'
import type { LigandBlob } from '../src/acts/boltz/featurizer/ligand'

const HERE = dirname(fileURLToPath(import.meta.url))

function loadBlob(ccd: string): LigandBlob {
  const path = resolve(HERE, `../public/ccd/${ccd}.json`)
  return JSON.parse(readFileSync(path, 'utf8')).data as LigandBlob
}

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

console.log('— Ligand single chain: HEM —')
{
  const hem = loadBlob('HEM')
  const f = featurizeChains([{ sequence: 'HEM', type: 'ligand', blob: hem }])
  expect('N = HEM atom count', f.N === hem.num_atoms, `got N=${f.N}, expected ${hem.num_atoms}`)
  expect('A >= N (1 atom per token)', f.A >= f.N)

  const molType = int64ToArray(f.tensors['mol_type']!.data as BigInt64Array)
  // NONPOLYMER = 3
  expect('mol_type = 3 (NONPOLYMER) across all tokens', molType.every((v) => v === 3))

  // Ligand atoms tokenise as UNK (token id 22). Each token's res_type one-hot
  // should have a 1 in column 22.
  const resType = (f.tensors['res_type']!.data as BigInt64Array)
  const NUM_TOKENS = f.tensors['res_type']!.shape[2]
  let unkHits = 0
  for (let n = 0; n < f.N; n++) {
    if (resType[n * NUM_TOKENS + 22] === 1n) unkHits++
  }
  expect('all ligand tokens res_type=UNK(22)', unkHits === f.N, `${unkHits}/${f.N}`)

  // atom_backbone_feat: every ligand atom should hit channel 0 (sidechain/off-list).
  const abf = int64ToArray(f.tensors['atom_backbone_feat']!.data as BigInt64Array)
  let ch0Hits = 0
  for (let a = 0; a < hem.num_atoms; a++) {
    if (abf[a * 17 + 0]) ch0Hits++
  }
  expect('all ligand atoms on channel 0', ch0Hits === hem.num_atoms, `${ch0Hits}/${hem.num_atoms}`)

  // atom_to_token: 1:1 mapping — atom a points at token a (within the ligand
  // block). Verify by checking the diagonal of the [A, N] one-hot.
  const a2t = int64ToArray(f.tensors['atom_to_token']!.data as BigInt64Array)
  let diagHits = 0
  for (let a = 0; a < hem.num_atoms; a++) {
    if (a2t[a * f.N + a] === 1) diagHits++
  }
  expect('atom_to_token is identity in ligand region', diagHits === hem.num_atoms)

  // All atoms share one augmentation group → ref_space_uid is constant.
  const refUid = int64ToArray(f.tensors['ref_space_uid']!.data as BigInt64Array)
  const uidsInLigand = new Set(refUid.slice(0, hem.num_atoms))
  expect('ref_space_uid constant within ligand', uidsInLigand.size === 1)

  // residue_index = 0 for every ligand atom-token (one residue per ligand).
  const resIdx = int64ToArray(f.tensors['residue_index']!.data as BigInt64Array)
  expect('residue_index = 0 for all ligand tokens', resIdx.every((v) => v === 0))

  // Constraint tensors: HEM has 0 chiral, 4 stereo, 2 planar bonds, 2 ring5,
  // 0 ring6, 903 rdkit bounds. Verify each tensor's trailing K dim matches.
  expect('chiral_atom_index K = 1 (padded, HEM has no chir)',
    f.tensors['chiral_atom_index']!.shape[2] === 1)
  expect('stereo_bond_index K = 4',
    f.tensors['stereo_bond_index']!.shape[2] === 4, JSON.stringify(f.tensors['stereo_bond_index']!.shape))
  expect('planar_bond_index K = 2',
    f.tensors['planar_bond_index']!.shape[2] === 2, JSON.stringify(f.tensors['planar_bond_index']!.shape))
  expect('planar_ring_5_index K = 2',
    f.tensors['planar_ring_5_index']!.shape[2] === 2)
  expect('planar_ring_6_index K = 1 (HEM has none)',
    f.tensors['planar_ring_6_index']!.shape[2] === 1)
  expect('rdkit_bounds_index K = 903',
    f.tensors['rdkit_bounds_index']!.shape[2] === 903)

  // token_bonds: HEM has 50 bonds; each becomes 2 entries (symmetric) in the
  // NxN matrix, so we expect 100 cells to be 1.0 inside the ligand block.
  const tokenBonds = f.tensors['token_bonds']!.data as Float32Array
  let bondCells = 0
  for (let i = 0; i < f.N; i++) {
    for (let j = 0; j < f.N; j++) {
      if (tokenBonds[i * f.N + j] === 1) bondCells++
    }
  }
  expect('token_bonds has 2 × 50 = 100 entries for HEM', bondCells === 100,
    `got ${bondCells}`)
}

console.log('— Mixed: protein + HEM ligand (offset check) —')
{
  const hem = loadBlob('HEM')
  const f = featurizeChains([
    { sequence: SEQ_A, type: 'protein' },
    { sequence: 'HEM', type: 'ligand', blob: hem },
  ])
  // Tokens: SEQ_A (20) + HEM atoms (43) = 63
  expect('N = 20 + 43', f.N === SEQ_A.length + hem.num_atoms, `got ${f.N}`)

  // asym_id: protein region = 0, ligand region = 1
  const asym = int64ToArray(f.tensors['asym_id']!.data as BigInt64Array)
  const protAsym = asym.slice(0, SEQ_A.length).every((v) => v === 0)
  const ligAsym = asym.slice(SEQ_A.length).every((v) => v === 1)
  expect('protein asym_id = 0', protAsym)
  expect('ligand asym_id = 1', ligAsym)

  // Constraint indices for HEM should be offset by the protein's atom count.
  // Find the first chiral row (HEM has 0 → empty) — use stereo instead.
  const stereo = int64ToArray(f.tensors['stereo_bond_index']!.data as BigInt64Array)
  const K_sb = f.tensors['stereo_bond_index']!.shape[2]
  expect('stereo K = 4', K_sb === 4)
  // Every index in stereo should be >= protein_atom_count (offset applied).
  // Find protein atom count by scanning atom_pad_mask of the smaller protein-only call:
  const protOnly = featurizeChains([{ sequence: SEQ_A, type: 'protein' }])
  const protAtoms = (protOnly.tensors['atom_pad_mask']!.data as Float32Array)
    .reduce((acc, v) => acc + (v ? 1 : 0), 0)
  const minStereoIdx = Math.min(...stereo.slice(0, 4 * K_sb).filter((v) => v >= 0))
  expect(`stereo indices >= protein atom count (${protAtoms})`,
    minStereoIdx >= protAtoms, `min stereo idx = ${minStereoIdx}, expected >= ${protAtoms}`)
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
