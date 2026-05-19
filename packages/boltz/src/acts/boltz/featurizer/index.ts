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

// ─────────────────────────────────────────────────────────────────────────────
// Chain input model
//
// The featurizer is parameterised over a list of entities. A "chain" here is
// one polymer (Phase 1: protein only). Single-sequence callers go through the
// `featurize(sequence)` wrapper at the bottom of this file, which packages
// the input as a single-element chain list. Multi-chain callers use
// `featurizeChains(chains)` directly.

export interface ChainInput {
  /** Raw sequence text. 1-letter codes for protein/RNA/DNA. Whitespace stripped. */
  sequence: string
  /** Entity type. Protein, RNA, or DNA. */
  type: ChainType
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
    if (c.type !== 'protein' && c.type !== 'rna' && c.type !== 'dna') {
      throw new Error(`Unsupported chain type: ${c.type}`)
    }
    const key = `${c.type}:${c.sequence}`
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
  /** Global token index (0..N-1 across all chains). Used for ref_space_uid and atom_to_token. */
  residueIdx: number
  /** Original index into the topology's `atoms[]` list (load-bearing for center/disto/backbone maps). */
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
  /** Position within the owning chain (0-based, restarts per chain). */
  residueInChainIdx: number
  /** Single-letter code for this token (for letterToTokenId / letterToResName). */
  letter: string
}

interface AtomEnumeration {
  atoms: AtomEntry[]
  /** atom indices grouped per global token. `byResidue[n]` lists emitted-atom-indices belonging to token n. */
  byResidue: number[][]
  /** First emitted atom index for each global token. */
  residueStart: number[]
  /**
   * Per token: orig-atom-idx → emitted-atom-idx (or -1 if dropped).
   * Used to translate center/disto/backbone indices through the leaving-atom filter.
   */
  origToEmittedPerResidue: number[][]
  /** Per-token metadata, length = N (sum of chain lengths). */
  tokens: TokenMeta[]
  /** Per-chain metadata, length = chains.length. */
  chains: ChainMeta[]
}

function enumerateAtoms(chains: ChainInput[]): AtomEnumeration {
  const chainMeta = assignChainMeta(chains)
  const atoms: AtomEntry[] = []
  const byResidue: number[][] = []
  const residueStart: number[] = []
  const origToEmittedPerResidue: number[][] = []
  const tokens: TokenMeta[] = []

  for (let c = 0; c < chains.length; c++) {
    const seq = chains[c].sequence
    const type = chains[c].type
    for (let r = 0; r < seq.length; r++) {
      const globalN = tokens.length
      tokens.push({ chainIdx: c, residueInChainIdx: r, letter: seq[r] })
      residueStart.push(atoms.length)
      const indicesHere: number[] = []
      byResidue.push(indicesHere)
      const resName = letterToResName(seq[r], type)
      const topo = residueTopology(resName, type)
      const map = new Array<number>(topo.atoms.length).fill(-1)
      origToEmittedPerResidue.push(map)
      for (let aIdx = 0; aIdx < topo.atoms.length; aIdx++) {
        const a = topo.atoms[aIdx]
        // Drop leaving atoms (OXT for proteins) — *every* residue, including
        // the C-terminal. Empirically matches the v0.1 Python featurizer's
        // behavior (the SPEC.md said "keep on C-term" but Boltz's own
        // featurizer flow drops it during the structure-from-sequence path).
        if (a.leaving) continue
        map[aIdx] = atoms.length
        indicesHere.push(atoms.length)
        atoms.push({
          residueIdx: globalN,
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

  return { atoms, byResidue, residueStart, origToEmittedPerResidue, tokens, chains: chainMeta }
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
  const N = chains.reduce((acc, c) => acc + c.sequence.length, 0)
  const enum_ = enumerateAtoms(chains)
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
      const ownerType = enum_.chains[enum_.tokens[at.residueIdx].chainIdx].type
      plddtArr[a] = 1
      refPosArr[a * 3]     = at.refPos[0]
      refPosArr[a * 3 + 1] = at.refPos[1]
      refPosArr[a * 3 + 2] = at.refPos[2]
      refElementArr[a * NUM_ELEMENTS + at.element] = 1n
      refChargeArr[a] = at.charge
      refChiralityArr[a] = BigInt(at.chiralityId)
      refSpaceUidArr[a] = BigInt(at.residueIdx)
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
    augmentRefPosPerResidue(refPosArr, enum_.byResidue)
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
      atomToTokenArr[a * N + at.residueIdx] = 1n
    }

    for (let n = 0; n < N; n++) {
      const tok = enum_.tokens[n]
      const type = enum_.chains[tok.chainIdx].type
      const resName = letterToResName(tok.letter, type)
      const topo = residueTopology(resName, type)
      const map = enum_.origToEmittedPerResidue[n]
      const centerEmitted = map[topo.center_atom_idx]
      const distoEmitted = map[topo.disto_atom_idx]
      if (centerEmitted >= 0) tokenToCenterArr[n * A + centerEmitted] = 1n
      if (distoEmitted >= 0) {
        tokenToRepArr[n * A + distoEmitted] = 1n
      }
      // r_set_to_rep_atom uses the *center* atom (Cα), not the disto atom.
      // SPEC.md said it was identical to token_to_rep_atom; verified against
      // 1L2Y golden that it actually mirrors token_to_center_atom.
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
    // token_bonds and type_bonds carry only *non-polymer* (CCD-level) bonds
    // and explicit user constraints. The peptide backbone is implicit via
    // residue_index ordering; do NOT mark adjacent residues here.
    const tokenBonds = tensor('token_bonds', [B, N, N, 1], 'float32')
    const typeBonds = tensor('type_bonds', [B, N, N], 'int64')
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

    void tokenBonds
    void typeBonds
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

  // ── H. Stub tensors (last axis = 1, zero-filled) ─────────────────────────
  // v0.1 F11 hotfix: the trunk/diffusion/confidence graphs were re-exported
  // with size-1 padded axes (rather than size-0) so ORT-Web's WebGPU backend
  // can compile their Concat kernels. The model is invariant to this dummy
  // dimension — pLDDT matches the size-0 baseline to within calibration noise.
  {
    add(tensor('chiral_atom_index',          [B, 4, 1], 'int64'))
    add(tensor('chiral_atom_orientations',   [B, 1],    'bool'))
    add(tensor('chiral_reference_mask',      [B, 1],    'bool'))
    add(tensor('connected_atom_index',       [B, 2, 1], 'int64'))
    add(tensor('connected_chain_index',      [B, 2, 1], 'int64'))
    add(tensor('contact_negation_mask',      [B, 1],    'bool'))
    add(tensor('contact_pair_index',         [B, 2, 1], 'int64'))
    add(tensor('contact_thresholds',         [B, 1],    'float32'))
    add(tensor('contact_union_index',        [B, 1],    'int64'))
    add(tensor('planar_bond_index',          [B, 6, 1], 'int64'))
    add(tensor('planar_ring_5_index',        [B, 5, 1], 'int64'))
    add(tensor('planar_ring_6_index',        [B, 6, 1], 'int64'))
    add(tensor('rdkit_bounds_angle_mask',    [B, 1],    'bool'))
    add(tensor('rdkit_bounds_bond_mask',     [B, 1],    'bool'))
    add(tensor('rdkit_bounds_index',         [B, 2, 1], 'int64'))
    add(tensor('rdkit_lower_bounds',         [B, 1],    'float32'))
    add(tensor('rdkit_upper_bounds',         [B, 1],    'float32'))
    add(tensor('stereo_bond_index',          [B, 4, 1], 'int64'))
    add(tensor('stereo_bond_orientations',   [B, 1],    'bool'))
    add(tensor('stereo_reference_mask',      [B, 1],    'bool'))
    add(tensor('symmetric_chain_index',      [B, 2, 1], 'int64'))
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
