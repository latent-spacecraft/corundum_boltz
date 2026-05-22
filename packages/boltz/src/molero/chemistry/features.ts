/**
 * Feature flags — per-atom booleans packed into a single Uint8Array
 * bitfield. Inputs come from the valence model + element table; outputs
 * drive material channels (Phase 2) and interaction detection (Phase 5).
 *
 * Bits (see scene.ts → AtomFlag):
 *   bit 0: HydrogenDonor    — N/O/S with totalH > 0
 *   bit 1: HydrogenAcceptor — O (always); N (no H, not ammonium)
 *   bit 2: AromaticRing     — from valence model (sp² ring member)
 *   bit 3: HydrophobicAtom  — C with no polar neighbors (approximated
 *                              by residue class for slice 1.1)
 *   bit 4: PositiveCharge   — formalCharge > 0.25
 *   bit 5: NegativeCharge   — formalCharge < -0.25
 *   bit 6: TransitionMetal  — element table
 *   bit 7: Backbone         — atom name in {N, CA, C, O, OXT, P, O5', C5',
 *                              C4', C3', O3'} for protein/nucleic
 */
import { atomNameFromId } from '../parsers/mmcif'
import { AtomFlag, Hybridization } from '../scene/scene'
import { isTransitionMetal } from './elements'

// Hydrophobic residues (Kyte-Doolittle positives). Slice 1.1 uses the
// residue-class heuristic for the HydrophobicAtom flag; Phase 2 will
// replace this with the per-atom Mol*-style heuristic (C with no polar
// neighbors).
const HYDROPHOBIC_RESIDUES = new Set([
  'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'CYS', 'GLY',
])

// Backbone atom names per polymer type.
const PROTEIN_BACKBONE = new Set(['N', 'CA', 'C', 'O', 'OXT', 'HA'])
const NUCLEIC_BACKBONE = new Set([
  'P', 'OP1', 'OP2', 'OP3',
  'O5\'', 'C5\'', 'C4\'', 'O4\'', 'C3\'', 'O3\'', 'C2\'', 'O2\'', 'C1\'',
])

export interface FeatureFlagsInput {
  atomCount: number
  atomicNumber: Uint8Array
  atomNameId: Uint32Array
  residueIndex: Uint32Array
  residueCompIds: string[]
  totalH: Uint8Array
  formalCharge: Float32Array
  /** Output bitfield. Must be length === atomCount. */
  out: Uint8Array
}

export function computeFeatureFlags(input: FeatureFlagsInput): void {
  const {
    atomCount,
    atomicNumber,
    atomNameId,
    residueIndex,
    residueCompIds,
    totalH,
    formalCharge,
    out,
  } = input

  let lastResIdx = -1
  let lastIsHydrophobic = false
  let lastBackbone: ReadonlySet<string> | undefined

  for (let i = 0; i < atomCount; i++) {
    const z = atomicNumber[i]
    const resIdx = residueIndex[i]
    if (resIdx !== lastResIdx) {
      const comp = residueCompIds[resIdx]
      lastIsHydrophobic = HYDROPHOBIC_RESIDUES.has(comp)
      // Crude: protein vs nucleic backbone selection. Treats any 3-letter
      // residue as protein; 1-2 letter as nucleic. Good enough for the
      // standard residues we care about now.
      lastBackbone = comp.length <= 2 ? NUCLEIC_BACKBONE : PROTEIN_BACKBONE
      lastResIdx = resIdx
    }
    const name = atomNameFromId(atomNameId[i])
    let f: number = AtomFlag.None

    // HydrogenDonor: N/O/S carrying H.
    if (totalH[i] > 0 && (z === 7 || z === 8 || z === 16)) {
      f |= AtomFlag.HydrogenDonor
    }
    // HydrogenAcceptor: O always (lone pair). N with no H and no positive charge.
    if (z === 8) {
      f |= AtomFlag.HydrogenAcceptor
    } else if (z === 7 && totalH[i] === 0 && formalCharge[i] <= 0) {
      f |= AtomFlag.HydrogenAcceptor
    }
    // AromaticRing — from valence-model hybridization (sp² ring carbons / N).
    // We can't know "ring" without ring perception, but sp² is a strong
    // proxy that matches our valence-model lookup table.
    // (Slice 1.2 will add real ring perception via Mol*'s scheme.)
    // For Phase 1.1 we treat sp² C/N as aromatic when residue is aromatic.
    // The lookup we want is already in valence-model; here we approximate
    // by checking sp² + element + residue membership.
    if ((z === 6 || z === 7) && AROMATIC_RESIDUES.has(residueCompIds[resIdx])) {
      // Use hybridization passed via formalCharge sidechannel? No — we
      // don't have hybridization here. Conservative: just mark sp²-likely
      // atoms in aromatic residues if they're a C or N. The valence model
      // has already tagged the specific ring atoms as sp²; we'd need it
      // here for precision. Mark all C/N in aromatic residues as
      // aromatic — the visual signal is approximate but consistent.
      f |= AtomFlag.AromaticRing
    }
    // HydrophobicAtom: C in a hydrophobic residue.
    if (z === 6 && lastIsHydrophobic) {
      f |= AtomFlag.HydrophobicAtom
    }
    // Charge flags.
    if (formalCharge[i] > 0.25) f |= AtomFlag.PositiveCharge
    else if (formalCharge[i] < -0.25) f |= AtomFlag.NegativeCharge
    // Metal.
    if (isTransitionMetal(z)) f |= AtomFlag.TransitionMetal
    // Backbone.
    if (lastBackbone?.has(name)) f |= AtomFlag.Backbone

    out[i] = f
  }
}

const AROMATIC_RESIDUES = new Set([
  'HIS', 'PHE', 'TYR', 'TRP',
  'A', 'G', 'C', 'U', 'T',
  'DA', 'DG', 'DC', 'DT', 'DU',
])

// Re-export so consumers can read flags symbolically.
export { AtomFlag, Hybridization }
