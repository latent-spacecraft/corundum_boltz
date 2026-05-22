/**
 * Valence model — per-atom formal charge, totalH, hybridization.
 *
 * Slice-1.1 implementation: small lookup tables for the canonical
 * functional groups in the 20 standard amino acids and 5 nucleotides;
 * element-based defaults for everything else. This is enough to drive
 * the per-atom material channels (charge → emission, hybridization →
 * roughness) for typical biomolecules.
 *
 * Slice 1.2+ will replace this with full CCD lookup + bond-perception-
 * derived computation, mirroring Mol*'s `ValenceModelProvider`. The
 * outputs/shape stay the same so downstream code is forward-compatible.
 */
import { atomNameFromId } from '../parsers/mmcif'
import { Hybridization } from '../scene/scene'

// ─────────────────────────────────────────────────────────────────────────────
// Charged sidechain atoms — formal charge per (residue, atom-name).
// Carboxylates are delocalized between the two oxygens; we tag both as
// -0.5 so emission contributions sum to -1 over the group.

const SPECIAL_CHARGES: Record<string, Record<string, number>> = {
  ARG: { NH1: +0.5, NH2: +0.5 },   // guanidinium delocalized over the two terminal N
  LYS: { NZ:  +1.0 },
  HIS: { },                        // ambiguous protonation; leave 0
  ASP: { OD1: -0.5, OD2: -0.5 },
  GLU: { OE1: -0.5, OE2: -0.5 },
  // Backbone termini — handled below by structural position rather than
  // residue identity, so left out here.
}

// ─────────────────────────────────────────────────────────────────────────────
// Aromatic-ring atoms by residue. Membership drives sp² hybridization and
// the AromaticRing feature flag downstream.

const AROMATIC_ATOMS: Record<string, ReadonlySet<string>> = {
  HIS: new Set(['CG', 'ND1', 'CD2', 'CE1', 'NE2']),
  PHE: new Set(['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
  TYR: new Set(['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']),
  TRP: new Set(['CG', 'CD1', 'NE1', 'CE2', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2']),
  // Purine bases (A, G, DA, DG) — same atom names.
  A:  new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']),
  G:  new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']),
  DA: new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']),
  DG: new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']),
  // Pyrimidines (C, U, T, DC, DT).
  C:  new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6']),
  U:  new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6']),
  T:  new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6']),
  DC: new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6']),
  DT: new Set(['N1', 'C2', 'N3', 'C4', 'C5', 'C6']),
}

// Carbonyl carbons in standard residues — sp² but not aromatic. Drives
// roughness mapping (sp² reads more polished than sp³).
const CARBONYL_CARBONS: Record<string, ReadonlySet<string>> = {
  // Backbone — every residue has a C carbonyl + O.
  ALA: new Set(['C']), ARG: new Set(['C', 'CZ']), ASN: new Set(['C', 'CG']),
  ASP: new Set(['C', 'CG']), CYS: new Set(['C']), GLN: new Set(['C', 'CD']),
  GLU: new Set(['C', 'CD']), GLY: new Set(['C']), HIS: new Set(['C']),
  ILE: new Set(['C']), LEU: new Set(['C']), LYS: new Set(['C']),
  MET: new Set(['C']), PHE: new Set(['C']), PRO: new Set(['C']),
  SER: new Set(['C']), THR: new Set(['C']), TRP: new Set(['C']),
  TYR: new Set(['C']), VAL: new Set(['C']),
}

// Hydroxyl oxygens get totalH = 1; carbonyl oxygens get totalH = 0.
const HYDROXYL_OXYGENS: Record<string, ReadonlySet<string>> = {
  SER: new Set(['OG']),
  THR: new Set(['OG1']),
  TYR: new Set(['OH']),
  // Nucleotide sugar 2'-OH (RNA only).
  A: new Set(['O2\'', 'O3\'', 'O5\'']),
  G: new Set(['O2\'', 'O3\'', 'O5\'']),
  C: new Set(['O2\'', 'O3\'', 'O5\'']),
  U: new Set(['O2\'', 'O3\'', 'O5\'']),
  DA: new Set(['O3\'', 'O5\'']),
  DG: new Set(['O3\'', 'O5\'']),
  DC: new Set(['O3\'', 'O5\'']),
  DT: new Set(['O3\'', 'O5\'']),
}

// ─────────────────────────────────────────────────────────────────────────────
// Element defaults — used when we have no residue-specific knowledge.

function defaultHybridization(atomicNumber: number): number {
  switch (atomicNumber) {
    case 1:  return Hybridization.Sp3   // H (trivial)
    case 6:  return Hybridization.Sp3   // C — sp3 unless ring/carbonyl
    case 7:  return Hybridization.Sp3   // N — usually sp3 amine
    case 8:  return Hybridization.Sp3   // O — sp3 hydroxyl default
    case 15: return Hybridization.Sp3   // P
    case 16: return Hybridization.Sp3   // S
    default: return Hybridization.Unknown
  }
}

function defaultTotalH(atomicNumber: number): number {
  switch (atomicNumber) {
    case 1:  return 0  // H atom itself
    case 6:  return 1  // approximate — most aliphatic Cs have 1+ H
    case 7:  return 1  // most Ns in protein are NH or NH2
    case 8:  return 0  // most Os are carbonyl; hydroxyls overridden
    case 16: return 1  // SH (Cys) — most Ss in protein are this
    default: return 0
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-atom dispatch.

export interface ValenceModelInput {
  atomCount: number
  atomicNumber: Uint8Array
  atomNameId: Uint32Array
  residueIndex: Uint32Array
  residueCompIds: string[]
  out: {
    formalCharge: Float32Array
    hybridization: Uint8Array
    totalH: Uint8Array
  }
}

export function computeValenceModel(input: ValenceModelInput): void {
  const { atomCount, atomicNumber, atomNameId, residueIndex, residueCompIds, out } = input

  // Cache the last residue's lookups so we don't hash-map per atom in a row.
  let lastResIdx = -1
  let lastChargeMap: Record<string, number> | undefined
  let lastAromaticSet: ReadonlySet<string> | undefined
  let lastCarbonylSet: ReadonlySet<string> | undefined
  let lastHydroxylSet: ReadonlySet<string> | undefined

  for (let i = 0; i < atomCount; i++) {
    const z = atomicNumber[i]
    const resIdx = residueIndex[i]
    if (resIdx !== lastResIdx) {
      const comp = residueCompIds[resIdx]
      lastChargeMap = SPECIAL_CHARGES[comp]
      lastAromaticSet = AROMATIC_ATOMS[comp]
      lastCarbonylSet = CARBONYL_CARBONS[comp]
      lastHydroxylSet = HYDROXYL_OXYGENS[comp]
      lastResIdx = resIdx
    }
    const name = atomNameFromId(atomNameId[i])

    // Formal charge.
    const ch = lastChargeMap?.[name]
    out.formalCharge[i] = ch !== undefined ? ch : 0

    // Hybridization.
    let hyb = defaultHybridization(z)
    if (lastAromaticSet?.has(name)) {
      hyb = Hybridization.Sp2
    } else if (z === 6 /* C */ && lastCarbonylSet?.has(name)) {
      hyb = Hybridization.Sp2
    } else if (z === 8 /* O */ && lastCarbonylSet) {
      // Backbone carbonyl O ('O') is sp2; sidechain hydroxyls stay sp3.
      // We approximate: if a residue has a carbonyl C 'C' and the atom
      // name is 'O', mark sp2.
      if (name === 'O' || name === 'OXT') hyb = Hybridization.Sp2
    }
    out.hybridization[i] = hyb

    // Total H. Hydroxyls override, then element default, then sp² oxygen → 0.
    let h = defaultTotalH(z)
    if (z === 8) {
      h = lastHydroxylSet?.has(name) ? 1 : 0
    } else if (z === 7) {
      // Charged amines / guanidiniums have more H than the default 1.
      if (lastChargeMap?.[name] && lastChargeMap[name] > 0) h = 3
    }
    out.totalH[i] = h
  }
}
