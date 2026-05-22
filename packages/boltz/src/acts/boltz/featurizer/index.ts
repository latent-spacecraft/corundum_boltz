/**
 * Boltz-2 featurizer (TS port).
 *
 * Mirrors `boltz/data/feature/featurizerv2.py::Boltz2Featurizer.process()`
 * for the single-sequence single-chain protein scope. Produces 78 tensors
 * the trunk ONNX graph consumes, as a `FeatsBundle` that the existing
 * orchestrator + worker accept by name.
 *
 * The source of truth is `docs/featurizer_port/SPEC.md`. Section-by-section
 * recipe; goldens under `docs/featurizer_port/golden/*` are byte-exact
 * oracles. If a tensor diverges from a golden, fix this file — never the
 * golden.
 */

import type { FeatsBundle, FeatsTensor, FeatsDtype } from '../featsLoader'
export { parseFasta, detectType, type ParsedChain } from './parseFasta'
export type { ChainType } from './tables'
import { makeRng, randomRotation, randomTranslation } from '../math'
import {
  ATOM_BACKBONE_FEAT_DIM,
  ATOM_WINDOW_W,
  BOND_TYPE_SINGLE_ID,
  type ChainType,
  atomBackboneChannel,
  chainTypeId,
  letterToResName,
  letterToTokenId,
  METHOD_TYPE_OTHER_ID,
  NUM_ELEMENTS,
  NUM_TOKENS,
  residueTopology,
} from './tables'
import type { LigandBlob } from './ligand'

// ─────────────────────────────────────────────────────────────────────────────
// Chain input model
//
// The featurizer is parameterised over a list of entities. A "chain" here is
// one polymer (Phase 1: protein only). Single-sequence callers go through the
// `featurize(sequence)` wrapper at the bottom of this file, which packages
// the input as a single-element chain list. Multi-chain callers use
// `featurizeChains(chains)` directly.

export interface ChainInput {
  /** For protein/RNA/DNA: residue letter string. For ligand: CCD 3-letter code. */
  sequence: string
  /** Entity type. Protein, RNA, DNA, or ligand (NONPOLYMER). */
  type: ChainType
  /** Required for `type: 'ligand'`. Loaded via loadLigandBlob() before
   *  featurizeChains is called. Unused for polymer chains. */
  blob?: LigandBlob
}

interface ChainMeta {
  /** asym_id — unique 0-based index per chain in order of appearance. */
  asymId: number
  /** entity_id — chains with identical (type, sequence) share this id. */
  entityId: number
  /** sym_id — 0-based index within an entity group (NCS copy number). */
  symId: number
  /** mol_type token id, derived from chain type. */
  molTypeId: number
  /** Original chain type — needed downstream for topology / backbone lookups. */
  type: ChainType
}

function assignChainMeta(chains: ChainInput[]): ChainMeta[] {
  const entityIdByKey = new Map<string, number>()
  const symCounterByEntity = new Map<number, number>()
  const out: ChainMeta[] = []
  for (let i = 0; i < chains.length; i++) {
    const c = chains[i]
    if (c.type !== 'protein' && c.type !== 'rna' && c.type !== 'dna' && c.type !== 'ligand') {
      throw new Error(`Unsupported chain type: ${c.type}`)
    }
    if (c.type === 'ligand' && !c.blob) {
      throw new Error(`Ligand chain '${c.sequence}' is missing its blob — call loadLigandBlob() first`)
    }
    // Homomer detection keys: polymer uses (type, sequence); ligand uses (type, ccd).
    const key = c.type === 'ligand' ? `ligand:${c.blob!.ccd}` : `${c.type}:${c.sequence}`
    let entityId = entityIdByKey.get(key)
    if (entityId === undefined) {
      entityId = entityIdByKey.size
      entityIdByKey.set(key, entityId)
    }
    const sym = symCounterByEntity.get(entityId) ?? 0
    symCounterByEntity.set(entityId, sym + 1)
    out.push({
      asymId: i,
      entityId,
      symId: sym,
      molTypeId: chainTypeId(c.type),
      type: c.type,
    })
  }
  return out
}

/**
 * Apply per-residue centering + Haar rotation + N(0, s_trans) translation
 * to ref_pos, in place. Mirrors Python's `center_random_augmentation` called
 * inside `featurizerv2.py:process` on every ref_space_uid group:
 *
 *   for i in range(max(ref_space_uid)):
 *       included = ref_space_uid == i
 *       ref_pos[included] = center_random_augmentation(
 *           ref_pos[included][None], resolved_mask[included][None],
 *           centering=True
 *       )[0]
 *
 * Without this step the trunk receives raw CCD coords, while the trained
 * model expects each residue's atoms in an augmented local frame. The
 * trained net IS invariant to which rotation we pick, so we just need to
 * apply *some* per-residue rotation+translation — Python uses fresh
 * torch.randn each call, we do the same with our own RNG.
 */
function augmentRefPosPerResidue(
  refPosArr: Float32Array,
  byResidue: number[][],
): void {
  const rng = makeRng(Date.now() >>> 0)
  const S_TRANS = 1.0
  for (const atoms of byResidue) {
    if (atoms.length === 0) continue
    // Centroid (uniform weights — every atom in a from-sequence CCD residue is "resolved").
    let cx = 0, cy = 0, cz = 0
    for (const a of atoms) {
      cx += refPosArr[a * 3]
      cy += refPosArr[a * 3 + 1]
      cz += refPosArr[a * 3 + 2]
    }
    const inv = 1 / atoms.length
    cx *= inv; cy *= inv; cz *= inv
    // Subtract centroid.
    for (const a of atoms) {
      refPosArr[a * 3]     -= cx
      refPosArr[a * 3 + 1] -= cy
      refPosArr[a * 3 + 2] -= cz
    }
    // Random Haar rotation (one per residue group).
    const R = randomRotation(rng)
    // Random translation (one per residue group).
    const t = randomTranslation(rng, S_TRANS)
    for (const a of atoms) {
      const x = refPosArr[a * 3]
      const y = refPosArr[a * 3 + 1]
      const z = refPosArr[a * 3 + 2]
      // Boltz convention: out = X @ R + t (row-vector convention; applyAffine uses this).
      refPosArr[a * 3]     = x * R[0] + y * R[3] + z * R[6] + t[0]
      refPosArr[a * 3 + 1] = x * R[1] + y * R[4] + z * R[7] + t[1]
      refPosArr[a * 3 + 2] = x * R[2] + y * R[5] + z * R[8] + t[2]
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Atom enumeration

interface AtomEntry {
  /** Token this atom maps to (atom_to_token target). 1:1 for ligand, many:1 for polymer. */
  tokenIdx: number
  /** Augmentation-group id (= ref_space_uid). One per polymer residue, one per
   *  ligand chain. Atoms in the same group share a Haar rotation + translation. */
  augGroupId: number
  /** Original index into the topology's `atoms[]` list. -1 for ligand atoms (they have no topology). */
  atomOrigIdx: number
  name: string
  element: number
  charge: number
  chiralityId: number
  refPos: [number, number, number]
}

interface TokenMeta {
  /** Owning chain's index into the input list. */
  chainIdx: number
  /** Position within the owning chain (0-based, restarts per chain). For
   *  ligand tokens all atoms in one ligand share residueInChainIdx=0
   *  (matches Boltz: a CCD ligand is one residue, even though each atom is a token). */
  residueInChainIdx: number
  /** Single-letter code for polymer tokens; '' for ligand. */
  letter: string
}

interface LigandSlice {
  asymId: number
  /** First atom in the global atoms[] array for this ligand chain. */
  atomOffset: number
  /** First token in the global token stream for this ligand chain. */
  tokenOffset: number
  blob: LigandBlob
}

interface AtomEnumeration {
  atoms: AtomEntry[]
  /** Atom indices grouped per augmentation group. byAugGroup[g] is the list
   *  of atom indices that rotate/translate together. */
  byAugGroup: number[][]
  /** For each token n: the first emitted atom index of that token. For ligand
   *  tokens, the token's single atom. For polymer tokens, the residue's first
   *  surviving atom. */
  tokenAtomStart: number[]
  /** For each polymer token: map from topo-orig-atom-idx → emitted-atom-idx
   *  (or -1 if dropped). For ligand tokens: empty (unused). */
  origToEmittedPerToken: number[][]
  /** Per-token metadata, length = N. */
  tokens: TokenMeta[]
  /** Per-chain metadata. */
  chains: ChainMeta[]
  /** Ligand chains in order — needed for Section H constraint assembly. */
  ligandSlices: LigandSlice[]
}

function enumerateAtoms(chains: ChainInput[]): AtomEnumeration {
  const chainMeta = assignChainMeta(chains)
  const atoms: AtomEntry[] = []
  const byAugGroup: number[][] = []
  const tokenAtomStart: number[] = []
  const origToEmittedPerToken: number[][] = []
  const tokens: TokenMeta[] = []
  const ligandSlices: LigandSlice[] = []

  for (let c = 0; c < chains.length; c++) {
    const ch = chains[c]
    if (ch.type === 'ligand') {
      // Ligand chain: each atom in the blob becomes both a token AND an atom.
      // All atoms share one augmentation group (the ligand frame).
      const blob = ch.blob!
      const augGroupId = byAugGroup.length
      const groupAtoms: number[] = []
      byAugGroup.push(groupAtoms)
      const slice: LigandSlice = {
        asymId: c,
        atomOffset: atoms.length,
        tokenOffset: tokens.length,
        blob,
      }
      ligandSlices.push(slice)
      for (let i = 0; i < blob.atoms.length; i++) {
        const a = blob.atoms[i]
        const tokenIdx = tokens.length
        tokens.push({ chainIdx: c, residueInChainIdx: 0, letter: '' })
        tokenAtomStart.push(atoms.length)
        origToEmittedPerToken.push([])
        groupAtoms.push(atoms.length)
        atoms.push({
          tokenIdx,
          augGroupId,
          atomOrigIdx: -1,
          name: a.name,
          element: a.element,
          charge: a.charge,
          chiralityId: a.chirality_id,
          refPos: a.ref_pos,
        })
      }
      continue
    }

    // Polymer chain: walk residues, emit one token per residue and one atom
    // per non-leaving topology atom. augGroup is per residue.
    const seq = ch.sequence
    for (let r = 0; r < seq.length; r++) {
      const tokenIdx = tokens.length
      const augGroupId = byAugGroup.length
      tokens.push({ chainIdx: c, residueInChainIdx: r, letter: seq[r] })
      tokenAtomStart.push(atoms.length)
      const groupAtoms: number[] = []
      byAugGroup.push(groupAtoms)
      const resName = letterToResName(seq[r], ch.type)
      const topo = residueTopology(resName, ch.type)
      const map = new Array<number>(topo.atoms.length).fill(-1)
      origToEmittedPerToken.push(map)
      for (let aIdx = 0; aIdx < topo.atoms.length; aIdx++) {
        const a = topo.atoms[aIdx]
        // Drop leaving atoms (OXT for proteins) — every residue including
        // C-terminal. Matches Boltz's structure-from-sequence path.
        if (a.leaving) continue
        map[aIdx] = atoms.length
        groupAtoms.push(atoms.length)
        atoms.push({
          tokenIdx,
          augGroupId,
          atomOrigIdx: aIdx,
          name: a.name,
          element: a.element,
          charge: a.charge,
          chiralityId: a.chirality_id,
          refPos: a.ref_pos,
        })
      }
    }
  }

  return {
    atoms,
    byAugGroup,
    tokenAtomStart,
    origToEmittedPerToken,
    tokens,
    chains: chainMeta,
    ligandSlices,
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tensor allocation helpers

function tensor(name: string, shape: number[], dtype: FeatsDtype): FeatsTensor {
  const numel = shape.reduce((a, b) => a * b, 1)
  let data: FeatsTensor['data']
  switch (dtype) {
    case 'float32':
      data = new Float32Array(numel)
      break
    case 'float16':
      data = new Uint16Array(numel)
      break
    case 'int64':
      data = new BigInt64Array(numel)
      break
    case 'int32':
      data = new Int32Array(numel)
      break
    case 'int16':
      data = new Int16Array(numel)
      break
    case 'int8':
      data = new Int8Array(numel)
      break
    case 'uint8':
    case 'bool':
      data = new Uint8Array(numel)
      break
  }
  return { name, shape, dtype, data }
}

function fillFloat(t: FeatsTensor, v: number): FeatsTensor {
  ;(t.data as Float32Array).fill(v)
  return t
}

function fillInt64(t: FeatsTensor, v: bigint): FeatsTensor {
  ;(t.data as BigInt64Array).fill(v)
  return t
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point

/**
 * Single-sequence convenience wrapper. The previous public API; preserved
 * verbatim so existing callers (BoltzAct, validate harness, single-chain
 * golden tests) work unchanged. A single-chain input is byte-identical to
 * passing a one-element list with chain type 'protein'.
 */
export function featurize(sequence: string): FeatsBundle {
  const seq = sequence.replace(/\s+/g, '').toUpperCase()
  if (seq.length === 0) throw new Error('Empty sequence')
  return featurizeChains([{ sequence: seq, type: 'protein' }])
}

export function featurizeChains(chainsInput: ChainInput[]): FeatsBundle {
  if (chainsInput.length === 0) throw new Error('No chains')
  const chains = chainsInput.map((c) => ({
    ...c,
    sequence: c.sequence.replace(/\s+/g, '').toUpperCase(),
  }))
  for (const c of chains) {
    if (c.sequence.length === 0) throw new Error('Empty sequence in chain')
  }
  const enum_ = enumerateAtoms(chains)
  // Total token count: one per polymer residue, one per ligand atom. Read from
  // the enumeration so the ligand branch (where N != sequence.length) lines up.
  const N = enum_.tokens.length
  const A_real = enum_.atoms.length
  const A = Math.ceil(A_real / ATOM_WINDOW_W) * ATOM_WINDOW_W
  const K = A / ATOM_WINDOW_W
  const B = 1

  const tensors: Record<string, FeatsTensor> = {}
  const add = (t: FeatsTensor) => {
    tensors[t.name] = t
  }

  // ── A. Token features ────────────────────────────────────────────────────
  {
    const resType = tensor('res_type', [B, N, NUM_TOKENS], 'int64')
    const tokIdx = tensor('token_index', [B, N], 'int64')
    const resIdx = tensor('residue_index', [B, N], 'int64')
    const molType = tensor('mol_type', [B, N], 'int64')
    const asymId = tensor('asym_id', [B, N], 'int64')
    const entityId = tensor('entity_id', [B, N], 'int64')
    const symId = tensor('sym_id', [B, N], 'int64')
    const tokenPad = fillFloat(tensor('token_pad_mask', [B, N], 'float32'), 1)
    const tokenResolved = fillFloat(tensor('token_resolved_mask', [B, N], 'float32'), 1)
    const tokenDisto = fillFloat(tensor('token_disto_mask', [B, N], 'float32'), 1)
    const cyclicPeriod = tensor('cyclic_period', [B, N], 'float32')
    const modified = tensor('modified', [B, N], 'int64')
    // Boltz default: method_types_ids["x-ray diffraction"] = 1. The "other"
    // path is only taken when override_method is explicitly passed.
    void METHOD_TYPE_OTHER_ID
    const methodFeature = fillInt64(
      tensor('method_feature', [B, N], 'int64'),
      1n,
    )
    const affinityTokenMask = tensor('affinity_token_mask', [B, N], 'float32')

    const resTypeArr = resType.data as BigInt64Array
    const tokIdxArr = tokIdx.data as BigInt64Array
    const resIdxArr = resIdx.data as BigInt64Array
    const molTypeArr = molType.data as BigInt64Array
    const asymIdArr = asymId.data as BigInt64Array
    const entityIdArr = entityId.data as BigInt64Array
    const symIdArr = symId.data as BigInt64Array
    for (let n = 0; n < N; n++) {
      const tok = enum_.tokens[n]
      const cm = enum_.chains[tok.chainIdx]
      const id = letterToTokenId(tok.letter, cm.type)
      resTypeArr[n * NUM_TOKENS + id] = 1n
      tokIdxArr[n] = BigInt(n)
      // residue_index restarts per chain (Boltz convention — distinct chains
      // share token_index but have independent residue numbering).
      resIdxArr[n] = BigInt(tok.residueInChainIdx)
      molTypeArr[n] = BigInt(cm.molTypeId)
      asymIdArr[n] = BigInt(cm.asymId)
      entityIdArr[n] = BigInt(cm.entityId)
      symIdArr[n] = BigInt(cm.symId)
    }
    void cyclicPeriod
    void modified
    void affinityTokenMask

    add(resType)
    add(tokIdx)
    add(resIdx)
    add(molType)
    add(asymId)
    add(entityId)
    add(symId)
    add(tokenPad)
    add(tokenResolved)
    add(tokenDisto)
    add(cyclicPeriod)
    add(modified)
    add(methodFeature)
    add(affinityTokenMask)
  }

  // ── B. Atom features ─────────────────────────────────────────────────────
  {
    const refPos = tensor('ref_pos', [B, A, 3], 'float32')
    const refElement = tensor('ref_element', [B, A, NUM_ELEMENTS], 'int64')
    const refCharge = tensor('ref_charge', [B, A], 'float32')
    const refChirality = tensor('ref_chirality', [B, A], 'int64')
    const refAtomNameChars = tensor('ref_atom_name_chars', [B, A, 4, 64], 'int64')
    const refSpaceUid = tensor('ref_space_uid', [B, A], 'int64')
    const atomPadMask = tensor('atom_pad_mask', [B, A], 'float32')
    const atomResolvedMask = tensor('atom_resolved_mask', [B, A], 'bool')
    const atomBackboneFeat = tensor('atom_backbone_feat', [B, A, ATOM_BACKBONE_FEAT_DIM], 'int64')
    const bfactor = tensor('bfactor', [B, A], 'float32')
    // Boltz from-sequence path initialises plddt = 1.0 for every real atom
    // (schema.py:1756). Padding rows remain 0.
    const plddt = tensor('plddt', [B, A], 'float32')
    const coords = tensor('coords', [B, 1, A, 3], 'float32')

    const refPosArr = refPos.data as Float32Array
    const refElementArr = refElement.data as BigInt64Array
    const refChargeArr = refCharge.data as Float32Array
    const refChiralityArr = refChirality.data as BigInt64Array
    const refAtomNameCharsArr = refAtomNameChars.data as BigInt64Array
    const refSpaceUidArr = refSpaceUid.data as BigInt64Array
    const atomPadArr = atomPadMask.data as Float32Array
    const atomResolvedArr = atomResolvedMask.data as Uint8Array
    const atomBackboneArr = atomBackboneFeat.data as BigInt64Array
    const plddtArr = plddt.data as Float32Array

    for (let a = 0; a < A_real; a++) {
      const at = enum_.atoms[a]
      const ownerType = enum_.chains[enum_.tokens[at.tokenIdx].chainIdx].type
      plddtArr[a] = 1
      refPosArr[a * 3]     = at.refPos[0]
      refPosArr[a * 3 + 1] = at.refPos[1]
      refPosArr[a * 3 + 2] = at.refPos[2]
      refElementArr[a * NUM_ELEMENTS + at.element] = 1n
      refChargeArr[a] = at.charge
      refChiralityArr[a] = BigInt(at.chiralityId)
      refSpaceUidArr[a] = BigInt(at.augGroupId)
      atomPadArr[a] = 1
      atomResolvedArr[a] = 1
      // ref_atom_name_chars: 4 chars × 64 vocab; pad name with spaces.
      const name = at.name.padEnd(4, ' ').slice(0, 4)
      for (let c = 0; c < 4; c++) {
        const charCode = name.charCodeAt(c) - 32
        const idx = charCode >= 0 && charCode < 64 ? charCode : 0
        refAtomNameCharsArr[a * 4 * 64 + c * 64 + idx] = 1n
      }
      // atom_backbone_feat one-hot. Channel 0 = sidechain / off-list.
      // Protein backbone N/CA/C/O → 1..4, nucleic P/OP1/OP2/O5'/C5'/C4'/O4'/
      // C3'/O3'/C2'/O2'/C1' → 5..16. Dispatch by chain type so a protein CA
      // doesn't get confused with the nucleic C-prefixed atoms.
      const channel = atomBackboneChannel(at.name, ownerType)
      atomBackboneArr[a * ATOM_BACKBONE_FEAT_DIM + channel] = 1n
    }
    // Per-residue ref_pos augmentation. Match featurizerv2.py:1494-1499 —
    // center each residue's atoms around their own centroid, then apply a
    // random Haar rotation + N(0, 1) translation. Trained net is invariant
    // to which rotation we pick; without this step the model receives
    // un-augmented input and produces collapsed coords.
    augmentRefPosPerResidue(refPosArr, enum_.byAugGroup)
    // Padding rows already zero-initialised. bfactor / plddt / coords stay zero.

    add(refPos)
    add(refElement)
    add(refCharge)
    add(refChirality)
    add(refAtomNameChars)
    add(refSpaceUid)
    add(atomPadMask)
    add(atomResolvedMask)
    add(atomBackboneFeat)
    add(bfactor)
    add(plddt)
    add(coords)
  }

  // ── C. Mapping tensors ───────────────────────────────────────────────────
  {
    const atomToToken = tensor('atom_to_token', [B, A, N], 'int64')
    const tokenToCenter = tensor('token_to_center_atom', [B, N, A], 'int64')
    const tokenToRep = tensor('token_to_rep_atom', [B, N, A], 'int64')
    const rSetToRep = tensor('r_set_to_rep_atom', [B, N, A], 'int64')
    const framesIdx = tensor('frames_idx', [B, 1, N, 3], 'int64')
    // frame_resolved_mask is False for from-sequence inputs because
    // token_atoms["is_present"] is False (no input coords). See
    // featurizerv2.py:1285+.
    const frameResolvedMask = tensor('frame_resolved_mask', [B, 1, N], 'bool')

    const atomToTokenArr = atomToToken.data as BigInt64Array
    const tokenToCenterArr = tokenToCenter.data as BigInt64Array
    const tokenToRepArr = tokenToRep.data as BigInt64Array
    const rSetToRepArr = rSetToRep.data as BigInt64Array
    const framesIdxArr = framesIdx.data as BigInt64Array

    for (let a = 0; a < A_real; a++) {
      const at = enum_.atoms[a]
      atomToTokenArr[a * N + at.tokenIdx] = 1n
    }

    for (let n = 0; n < N; n++) {
      const tok = enum_.tokens[n]
      const type = enum_.chains[tok.chainIdx].type
      if (type === 'ligand') {
        // Each ligand atom IS a token (1:1). center/rep/r-set all point at
        // the token's own atom. frames_idx isn't used (frame_resolved_mask
        // stays false), but we still need to write something — zeros match
        // Boltz's convention for non-polymer frames.
        const atomIdx = enum_.tokenAtomStart[n]
        if (atomIdx >= 0) {
          tokenToCenterArr[n * A + atomIdx] = 1n
          tokenToRepArr[n * A + atomIdx] = 1n
          rSetToRepArr[n * A + atomIdx] = 1n
        }
        framesIdxArr[n * 3]     = 0n
        framesIdxArr[n * 3 + 1] = 0n
        framesIdxArr[n * 3 + 2] = 0n
        continue
      }
      // Polymer token: pick center/disto/backbone via residue topology.
      const resName = letterToResName(tok.letter, type)
      const topo = residueTopology(resName, type)
      const map = enum_.origToEmittedPerToken[n]
      const centerEmitted = map[topo.center_atom_idx]
      const distoEmitted = map[topo.disto_atom_idx]
      if (centerEmitted >= 0) tokenToCenterArr[n * A + centerEmitted] = 1n
      if (distoEmitted >= 0) {
        tokenToRepArr[n * A + distoEmitted] = 1n
      }
      // r_set_to_rep_atom uses the *center* atom (Cα), not the disto atom.
      // Verified against 1L2Y golden that it mirrors token_to_center_atom.
      if (centerEmitted >= 0) {
        rSetToRepArr[n * A + centerEmitted] = 1n
      }
      // frames_idx: (N, CA, C) absolute atom indices per residue.
      const bb = topo.backbone_atom_idx
      const Nemit = map[bb[0]]
      const CAemit = map[bb[1]]
      const Cemit = map[bb[2]]
      framesIdxArr[n * 3]     = BigInt(Nemit >= 0 ? Nemit : 0)
      framesIdxArr[n * 3 + 1] = BigInt(CAemit >= 0 ? CAemit : 0)
      framesIdxArr[n * 3 + 2] = BigInt(Cemit >= 0 ? Cemit : 0)
    }

    add(atomToToken)
    add(tokenToCenter)
    add(tokenToRep)
    add(rSetToRep)
    add(framesIdx)
    add(frameResolvedMask)
  }

  // ── D. Token-pair features ───────────────────────────────────────────────
  {
    // token_bonds and type_bonds carry *non-polymer* (CCD-level) bonds and
    // explicit user constraints. The peptide / nucleic backbone is implicit
    // via residue_index ordering — for those we leave the matrices at 0.
    // For ligand chains every atom is its own token, so the bond graph the
    // trunk needs lives entirely in these two NxN matrices. Without this,
    // ligand atoms get scattered (the trunk can't tell which atoms are
    // bonded; HEM emerges as a cloud of free atoms rather than a porphyrin).
    const tokenBonds = tensor('token_bonds', [B, N, N, 1], 'float32')
    const typeBonds = tensor('type_bonds', [B, N, N], 'int64')
    const tokenBondsArr = tokenBonds.data as Float32Array
    const typeBondsArr = typeBonds.data as BigInt64Array
    for (const slice of enum_.ligandSlices) {
      for (const b of slice.blob.bonds) {
        // Each ligand atom IS a token; offset by tokenOffset to land in the
        // global token index space.
        const ti = slice.tokenOffset + b.i
        const tj = slice.tokenOffset + b.j
        // Symmetric upper/lower fill so the trunk's attention sees both
        // directions of the bond.
        tokenBondsArr[ti * N + tj] = 1
        tokenBondsArr[tj * N + ti] = 1
        typeBondsArr[ti * N + tj] = BigInt(b.type_id)
        typeBondsArr[tj * N + ti] = BigInt(b.type_id)
      }
    }
    void BOND_TYPE_SINGLE_ID

    // contact_conditioning: one-hot at channel 0 for every pair in the
    // no-constraint case. The const table says UNSELECTED=1 but the path
    // that produces the actual ONNX-input tensor for single-chain protein
    // ends up at channel 0 — verified byte-exact against the 1L2Y golden.
    // (Likely the np.zeros + UNSELECTED gets remapped downstream, or
    // UNSPECIFIED=0 is the value when no pocket/contact constraints exist.)
    const contactConditioning = tensor('contact_conditioning', [B, N, N, 5], 'int64')
    const contactConditioningArr = contactConditioning.data as BigInt64Array
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        contactConditioningArr[(i * N + j) * 5] = 1n
      }
    }
    const contactThreshold = tensor('contact_threshold', [B, N, N], 'float32')

    // disto_target is the one-hot of the binned pairwise distogram. With
    // from-sequence inputs all coords are zero, so all pairwise distances
    // are 0 → bin 0 → one-hot at channel 0. Shape [B, N, N, 1, 64].
    const distoTarget = tensor('disto_target', [B, N, N, 1, 64], 'float32')
    const distoTargetArr = distoTarget.data as Float32Array
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        distoTargetArr[((i * N + j) * 1) * 64] = 1
      }
    }

    void contactThreshold

    add(tokenBonds)
    add(typeBonds)
    add(contactConditioning)
    add(contactThreshold)
    add(distoTarget)
  }

  // ── E. MSA (single-row) ──────────────────────────────────────────────────
  {
    const msa = tensor('msa', [B, 1, N], 'int64')
    const msaMask = fillInt64(tensor('msa_mask', [B, 1, N], 'int64'), 1n)
    // The single MSA row is the query itself; Boltz marks it as paired
    // with the chain on row 0 (msa.py:336 — is_paired starts as {c: 1, ...}).
    const msaPaired = fillFloat(tensor('msa_paired', [B, 1, N], 'float32'), 1)
    const hasDeletion = tensor('has_deletion', [B, 1, N], 'bool')
    const deletionValue = tensor('deletion_value', [B, 1, N], 'float32')
    const deletionMean = tensor('deletion_mean', [B, N], 'float32')
    const profile = tensor('profile', [B, N, NUM_TOKENS], 'float32')

    const msaArr = msa.data as BigInt64Array
    const profileArr = profile.data as Float32Array
    for (let n = 0; n < N; n++) {
      const tok = enum_.tokens[n]
      const id = letterToTokenId(tok.letter, enum_.chains[tok.chainIdx].type)
      msaArr[n] = BigInt(id)
      profileArr[n * NUM_TOKENS + id] = 1
    }
    void msaPaired
    void hasDeletion
    void deletionValue
    void deletionMean

    add(msa)
    add(msaMask)
    add(msaPaired)
    add(hasDeletion)
    add(deletionValue)
    add(deletionMean)
    add(profile)
  }

  // ── F. Template features (templates absent) ──────────────────────────────
  {
    add(tensor('template_mask', [B, 1, N], 'float32'))
    add(tensor('template_mask_cb', [B, 1, N], 'float32'))
    add(tensor('template_mask_frame', [B, 1, N], 'float32'))
    // template_restype: when no template is supplied Boltz fills it with
    // one-hot of <pad> (channel 0) for every residue position. Verified
    // against 1L2Y golden.
    const templateRestype = tensor('template_restype', [B, 1, N, NUM_TOKENS], 'int64')
    const templateRestypeArr = templateRestype.data as BigInt64Array
    for (let n = 0; n < N; n++) {
      templateRestypeArr[n * NUM_TOKENS + 0] = 1n
    }
    add(templateRestype)
    add(tensor('template_ca', [B, 1, N, 3], 'float32'))
    add(tensor('template_cb', [B, 1, N, 3], 'float32'))
    add(tensor('template_frame_t', [B, 1, N, 3], 'float32'))
    add(tensor('template_frame_rot', [B, 1, N, 3, 3], 'float32'))
    add(tensor('query_to_template', [B, 1, N], 'int64'))
    add(tensor('visibility_ids', [B, 1, N], 'float32'))
  }

  // ── G. Ensemble (one conformer, trivial) ─────────────────────────────────
  {
    add(tensor('disto_center', [B, N, 3], 'float32'))
    add(tensor('disto_coords_ensemble', [B, 1, N, 3], 'float32'))
    const ensembleRefIdxs = tensor('ensemble_ref_idxs', [B, 1], 'int64')
    // Already zero — meta says [[0]] which is what we have.
    add(ensembleRefIdxs)
  }

  // ── H. Per-ligand geometry constraints ───────────────────────────────────
  // The five per-molecule constraint groups (rdkit bounds, chiral atoms,
  // stereo bonds, planar bonds, planar rings 5/6) are aggregated across all
  // ligand chains with atom indices shifted by each ligand's atomOffset.
  // When no ligands are present, every K collapses to 0 and we fall back to
  // the F11 hotfix: size-1 zero-filled trailing axis so ORT-Web's Concat
  // kernels can compile. Polymer-only inputs are byte-identical to the
  // pre-ligand path through the trunk.
  //
  // The cross-chain / user-constraint group (connected_*, contact_*,
  // symmetric_chain_index) stays size-1 zero-filled. Inter-chain disulfides
  // and pocket constraints are deferred to a later phase; empirically the
  // model tolerates these zero-padded for everything we test.
  {
    // Aggregate from all ligand slices, applying atomOffset to each index.
    const chiralRows: number[][] = []          // [a, b, c, d] per row
    const chiralRef: number[] = []
    const chiralOrient: number[] = []
    const stereoRows: number[][] = []          // [a, b, c, d]
    const stereoCheck: number[] = []
    const stereoOrient: number[] = []
    const planarBondRows: number[][] = []      // [6 atom indices]
    const planar5Rows: number[][] = []         // [5]
    const planar6Rows: number[][] = []         // [6]
    const rdkitRows: number[][] = []           // [i, j]
    const rdkitBondMask: number[] = []
    const rdkitAngleMask: number[] = []
    const rdkitLower: number[] = []
    const rdkitUpper: number[] = []

    for (const slice of enum_.ligandSlices) {
      const off = slice.atomOffset
      for (const c of slice.blob.chiral_atoms) {
        chiralRows.push(c.atoms.map((i) => i + off))
        chiralRef.push(c.is_reference ? 1 : 0)
        chiralOrient.push(c.is_r ? 1 : 0)
      }
      for (const s of slice.blob.stereo_bonds) {
        stereoRows.push(s.atoms.map((i) => i + off))
        stereoCheck.push(s.is_check ? 1 : 0)
        stereoOrient.push(s.is_e ? 1 : 0)
      }
      for (const p of slice.blob.planar_bonds) {
        planarBondRows.push(p.atoms.map((i) => i + off))
      }
      for (const p of slice.blob.planar_rings_5) {
        planar5Rows.push(p.atoms.map((i) => i + off))
      }
      for (const p of slice.blob.planar_rings_6) {
        planar6Rows.push(p.atoms.map((i) => i + off))
      }
      for (const r of slice.blob.rdkit_bounds) {
        rdkitRows.push([r.i + off, r.j + off])
        rdkitBondMask.push(r.is_bond ? 1 : 0)
        rdkitAngleMask.push(r.is_angle ? 1 : 0)
        rdkitLower.push(r.lower)
        rdkitUpper.push(r.upper)
      }
    }

    // K=1 when empty so the trunk's existing exported shape (F11 hotfix)
    // continues to satisfy ORT-Web's Concat kernel.
    const padK = (k: number) => Math.max(k, 1)

    // [B, axis, K] writer: writes row k's `axis`-tuple along the trailing dim.
    const writeRows = (t: FeatsTensor, rows: number[][], rowLen: number) => {
      const arr = t.data as BigInt64Array
      const K = t.shape[2]
      for (let k = 0; k < rows.length; k++) {
        for (let i = 0; i < rowLen; i++) {
          arr[i * K + k] = BigInt(rows[k][i])
        }
      }
    }
    // [B, K] writer: bool / float / int64 masks indexed by k.
    const writeMaskBool = (t: FeatsTensor, values: number[]) => {
      const arr = t.data as Uint8Array
      for (let k = 0; k < values.length; k++) arr[k] = values[k]
    }
    const writeMaskF32 = (t: FeatsTensor, values: number[]) => {
      const arr = t.data as Float32Array
      for (let k = 0; k < values.length; k++) arr[k] = values[k]
    }

    // Chiral
    {
      const K_ = padK(chiralRows.length)
      const idx = tensor('chiral_atom_index', [B, 4, K_], 'int64')
      writeRows(idx, chiralRows, 4)
      add(idx)
      const orient = tensor('chiral_atom_orientations', [B, K_], 'bool')
      writeMaskBool(orient, chiralOrient)
      add(orient)
      const ref = tensor('chiral_reference_mask', [B, K_], 'bool')
      writeMaskBool(ref, chiralRef)
      add(ref)
    }
    // Stereo
    {
      const K_ = padK(stereoRows.length)
      const idx = tensor('stereo_bond_index', [B, 4, K_], 'int64')
      writeRows(idx, stereoRows, 4)
      add(idx)
      const orient = tensor('stereo_bond_orientations', [B, K_], 'bool')
      writeMaskBool(orient, stereoOrient)
      add(orient)
      const ref = tensor('stereo_reference_mask', [B, K_], 'bool')
      writeMaskBool(ref, stereoCheck)
      add(ref)
    }
    // Planar bonds / rings
    {
      const K_ = padK(planarBondRows.length)
      const idx = tensor('planar_bond_index', [B, 6, K_], 'int64')
      writeRows(idx, planarBondRows, 6)
      add(idx)
    }
    {
      const K_ = padK(planar5Rows.length)
      const idx = tensor('planar_ring_5_index', [B, 5, K_], 'int64')
      writeRows(idx, planar5Rows, 5)
      add(idx)
    }
    {
      const K_ = padK(planar6Rows.length)
      const idx = tensor('planar_ring_6_index', [B, 6, K_], 'int64')
      writeRows(idx, planar6Rows, 6)
      add(idx)
    }
    // RDKit bounds
    {
      const K_ = padK(rdkitRows.length)
      const idx = tensor('rdkit_bounds_index', [B, 2, K_], 'int64')
      writeRows(idx, rdkitRows, 2)
      add(idx)
      const bondMask = tensor('rdkit_bounds_bond_mask', [B, K_], 'bool')
      writeMaskBool(bondMask, rdkitBondMask)
      add(bondMask)
      const angleMask = tensor('rdkit_bounds_angle_mask', [B, K_], 'bool')
      writeMaskBool(angleMask, rdkitAngleMask)
      add(angleMask)
      const lower = tensor('rdkit_lower_bounds', [B, K_], 'float32')
      writeMaskF32(lower, rdkitLower)
      add(lower)
      const upper = tensor('rdkit_upper_bounds', [B, K_], 'float32')
      writeMaskF32(upper, rdkitUpper)
      add(upper)
    }
    // Cross-chain / user-constraint group — kept zero-filled. Confirmed
    // polymer-invariant in the polymer phase; for ligand prediction the
    // small molecule's intra-graph constraints are what matters.
    add(tensor('connected_atom_index',  [B, 2, 1], 'int64'))
    add(tensor('connected_chain_index', [B, 2, 1], 'int64'))
    add(tensor('contact_negation_mask', [B, 1],    'bool'))
    add(tensor('contact_pair_index',    [B, 2, 1], 'int64'))
    add(tensor('contact_thresholds',    [B, 1],    'float32'))
    add(tensor('contact_union_index',   [B, 1],    'int64'))
    add(tensor('symmetric_chain_index', [B, 2, 1], 'int64'))
  }

  return {
    schemaVersion: '0.1',
    B,
    N,
    A,
    K,
    tensors,
  }
}
