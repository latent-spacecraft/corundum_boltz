/**
 * Hydrophobicity — Kyte-Doolittle scale, per residue.
 *
 * Reference: Kyte, J. & Doolittle, R. F. (1982).
 *   A simple method for displaying the hydropathic character of a protein.
 *   J. Mol. Biol. 157, 105-132.
 *
 * Range: +4.5 (Ile, most hydrophobic) to -4.5 (Arg, most hydrophilic).
 * Standard nucleotides return 0 — hydrophobicity is a protein concept
 * here; PRIs / RNA / DNA get neutral treatment. Unknown residues also
 * return 0.
 *
 * The per-atom variant assigns each atom its residue's score. A finer
 * per-atom weighting (Mol*-style: polar groups boost hydrophilicity even
 * inside hydrophobic residues) is a Phase-2.5 follow-up.
 */
import type { Scene as MoleroScene } from '../scene/scene'

const KYTE_DOOLITTLE: Record<string, number> = {
  ILE:  4.5,
  VAL:  4.2,
  LEU:  3.8,
  PHE:  2.8,
  CYS:  2.5,
  MET:  1.9,
  ALA:  1.8,
  GLY: -0.4,
  THR: -0.7,
  SER: -0.8,
  TRP: -0.9,
  TYR: -1.3,
  PRO: -1.6,
  HIS: -3.2,
  GLU: -3.5,
  GLN: -3.5,
  ASP: -3.5,
  ASN: -3.5,
  LYS: -3.9,
  ARG: -4.5,
  // Non-standard amino acids — best-effort estimates.
  SEC:  2.5, // selenocysteine ≈ Cys
  PYL: -3.9, // pyrrolysine ≈ Lys
  MSE:  1.9, // selenomethionine ≈ Met
}

export function residueHydrophobicity(compId: string): number {
  return KYTE_DOOLITTLE[compId] ?? 0
}

/**
 * Returns a Float32Array of per-atom hydrophobicity (one entry per atom
 * in scene.attrs). Atoms in non-protein chains get 0.
 */
export function computeHydrophobicity(scene: MoleroScene): Float32Array {
  const A = scene.attrs.count
  const out = new Float32Array(A)
  const residueIndex = scene.attrs.residueIndex
  const residues = scene.residues
  const chains = scene.chains

  // Precompute per-residue scores so we don't redo the map lookup per atom.
  const perResidue = new Float32Array(residues.length)
  for (let r = 0; r < residues.length; r++) {
    const res = residues[r]
    const chain = chains[res.chainIndex]
    if (chain?.entityType === 'protein') {
      perResidue[r] = residueHydrophobicity(res.compId)
    }
  }
  for (let i = 0; i < A; i++) {
    out[i] = perResidue[residueIndex[i]]
  }
  return out
}

/** Normalize a Kyte-Doolittle score to [0, 1] (4.5 → 1, -4.5 → 0).
 *  Useful for material-channel mapping where the GPU wants 0..1. */
export function hydrophobicityNorm(score: number): number {
  return (score + 4.5) / 9.0
}
