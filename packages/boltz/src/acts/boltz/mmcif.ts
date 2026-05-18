/**
 * Minimal mmCIF writer.
 *
 * Writes an all-atom `_atom_site` loop with the standard 22 columns Mol* expects,
 * placing per-residue pLDDT × 100 in the B_iso_or_equiv column (Mol*'s default
 * "By B-factor" coloring picks it up automatically).
 *
 * Inputs are decoded from the captured feats tensors:
 *   - `atom_pad_mask[A]`             : 1 for real atoms, 0 for padding
 *   - `atom_to_token[A, N]`          : one-hot per atom → residue index
 *   - `ref_atom_name_chars[A, 4, 64]`: one-hot per atom × 4 chars × 64 ascii values
 *   - `ref_element[A, 128]`          : one-hot per atom → atomic number
 *
 * Coordinates are `[A * 3]` (single batch). pLDDT is per-residue `[N]` in [0, 100].
 *
 * Residue identity comes from the sequence string the caller supplies (single
 * chain). Chain ID is hard-coded to 'A' for v0.1.
 */
import { argmaxLast, type Rng as _Rng } from './math'
import type { FeatsBundle, FeatsTensor } from './featsLoader'

/**
 * Coerce a feats-tensor backing store into a plain numeric ArrayLike.
 * BigInt64Array values must be converted to Number — passing them to a
 * non-bigint TypedArray constructor throws
 *   "Content types of source and new typed array are different".
 */
function asNumberArray(t: FeatsTensor): ArrayLike<number> {
  const d = t.data
  if (d instanceof BigInt64Array) {
    const out = new Float64Array(d.length)
    for (let i = 0; i < d.length; i++) out[i] = Number(d[i])
    return out
  }
  return d as ArrayLike<number>
}

const THREE_LETTER: Record<string, string> = {
  A: 'ALA', R: 'ARG', N: 'ASN', D: 'ASP', C: 'CYS',
  E: 'GLU', Q: 'GLN', G: 'GLY', H: 'HIS', I: 'ILE',
  L: 'LEU', K: 'LYS', M: 'MET', F: 'PHE', P: 'PRO',
  S: 'SER', T: 'THR', W: 'TRP', Y: 'TYR', V: 'VAL',
  X: 'UNK', B: 'ASX', Z: 'GLX', U: 'SEC', O: 'PYL',
}

const ATOMIC_NUMBER_TO_SYMBOL: string[] = [
  '?',  'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne', // 0–10
  'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca', // 11–20
  'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', // 21–30
  'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr', // 31–40
  'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', // 41–50
  'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', // 51–60
  'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', // 61–70
  'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', // 71–80
  'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', // 81–86
]

function atomicNumberToSymbol(z: number): string {
  return ATOMIC_NUMBER_TO_SYMBOL[z] ?? '?'
}

/** Decode atom name from `ref_atom_name_chars[a, 4, 64]` one-hot. Strips trailing spaces. */
function decodeAtomName(charsOneHot: ArrayLike<number>, a: number): string {
  // chars are 4 positions, each one-hot over 64 ASCII values starting at chr(32) (space).
  const base = a * 4 * 64
  let out = ''
  for (let c = 0; c < 4; c++) {
    const charBase = base + c * 64
    let bestIdx = 0
    let bestVal = charsOneHot[charBase]
    for (let i = 1; i < 64; i++) {
      const v = charsOneHot[charBase + i]
      if (v > bestVal) { bestVal = v; bestIdx = i }
    }
    out += String.fromCharCode(bestIdx + 32)
  }
  return out.trim()
}

/**
 * Build an mmCIF string from the prediction outputs + the feats that defined
 * what each atom is. pLDDT is per-residue, [0, 100] scale.
 */
export function writeMmcif(opts: {
  feats: FeatsBundle
  atomCoords: Float32Array     // [A * 3]
  plddt: Float32Array          // [N]
  sequence: string             // length N
  modelId?: string
}): string {
  const { feats, atomCoords, plddt, sequence } = opts
  const id = opts.modelId ?? 'predicted'
  const A = feats.A
  const N = feats.N

  const atomPadMaskT = feats.tensors['atom_pad_mask']
  const atomToTokenT = feats.tensors['atom_to_token']
  const refElement = feats.tensors['ref_element']
  const refAtomNameChars = feats.tensors['ref_atom_name_chars']

  // atom_pad_mask dtype is float32 in our featurizer; treat any storage as
  // a numeric ArrayLike so `!atomPadMask[a]` works regardless.
  const atomPadMask = asNumberArray(atomPadMaskT)

  // atom_to_token shape [A, N], one-hot along last dim → residue index per atom.
  const atomResidue = argmaxLast(asNumberArray(atomToTokenT), N)

  // ref_element [A, K_elem] one-hot → atomic number.
  const elementDim = refElement.shape[refElement.shape.length - 1]
  const atomicNum = argmaxLast(asNumberArray(refElement), elementDim)

  // ref_atom_name_chars [A, 4, 64] one-hot over ASCII offset.
  const charBuf = asNumberArray(refAtomNameChars)

  // Build the loop.
  const lines: string[] = []
  lines.push(`data_${id}`)
  lines.push('_entry.id ' + id)
  lines.push('loop_')
  lines.push('_atom_site.group_PDB')
  lines.push('_atom_site.id')
  lines.push('_atom_site.type_symbol')
  lines.push('_atom_site.label_atom_id')
  lines.push('_atom_site.label_alt_id')
  lines.push('_atom_site.label_comp_id')
  lines.push('_atom_site.label_asym_id')
  lines.push('_atom_site.label_entity_id')
  lines.push('_atom_site.label_seq_id')
  lines.push('_atom_site.pdbx_PDB_ins_code')
  lines.push('_atom_site.Cartn_x')
  lines.push('_atom_site.Cartn_y')
  lines.push('_atom_site.Cartn_z')
  lines.push('_atom_site.occupancy')
  lines.push('_atom_site.B_iso_or_equiv')
  lines.push('_atom_site.auth_seq_id')
  lines.push('_atom_site.auth_asym_id')
  lines.push('_atom_site.pdbx_PDB_model_num')

  let serial = 0
  for (let a = 0; a < A; a++) {
    if (!atomPadMask[a]) continue
    serial++
    const residueIdx = atomResidue[a]
    if (residueIdx < 0 || residueIdx >= N) continue
    const aaLetter = sequence[residueIdx] ?? 'X'
    const compId = THREE_LETTER[aaLetter] ?? 'UNK'
    const seqId = residueIdx + 1
    const element = atomicNumberToSymbol(atomicNum[a])
    const atomName = decodeAtomName(charBuf, a) || element
    const x = atomCoords[a * 3]
    const y = atomCoords[a * 3 + 1]
    const z = atomCoords[a * 3 + 2]
    const b = plddt[residueIdx]
    const occupancy = 1.0
    lines.push(
      [
        'ATOM',
        serial,
        element,
        atomName,
        '.',
        compId,
        'A',
        '1',
        seqId,
        '?',
        x.toFixed(3),
        y.toFixed(3),
        z.toFixed(3),
        occupancy.toFixed(2),
        b.toFixed(2),
        seqId,
        'A',
        '1',
      ].join(' '),
    )
  }
  lines.push('#')
  return lines.join('\n') + '\n'
}
