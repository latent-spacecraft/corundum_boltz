/**
 * Backbone extraction — pulls the trace atoms for ribbon rendering.
 *
 * Protein chains: Cα atom of each residue.
 * Nucleic acids:  C4' (sugar) when present; falls back to P.
 *
 * Chains break naturally on missing residues or large Cα-Cα gaps (more
 * than `gapThreshold` Å between consecutive trace atoms). The output is
 * a list of *segments*: continuous polylines, one per uninterrupted
 * stretch. Each segment carries the atom indices behind each trace
 * point so downstream passes can color by per-atom properties
 * (pLDDT, conservation, etc.).
 *
 * Ligand / water / unknown chains have no backbone trace and are
 * skipped — they should be rendered with sphere + stick passes only.
 */
import { internAtomName } from '../parsers/mmcif'
import type { Scene as MoleroScene } from '../scene/scene'

export interface BackboneSegment {
  /** Source chain index. */
  chainIndex: number
  /** Sequential trace points along the backbone. */
  positions: Float32Array        // [N * 3]
  /** Atom index in PropertyAttributes for each trace point. */
  atomIndex: Uint32Array         // [N]
  /** Residue index for each trace point. */
  residueIndex: Uint32Array      // [N]
  /** Entity type from the source chain — used by passes that style
   *  protein vs nucleic differently. */
  entityType: 'protein' | 'rna' | 'dna'
}

export interface BackboneOptions {
  /** Protein Cα-Cα distance above which we split into a new segment (Å).
   *  Healthy peptide bond ≈ 3.8 Å; 4.5 catches missing residues without
   *  triggering on tight turns. */
  proteinGapThreshold: number
  /** Nucleic C4'-C4' (or P-P) distance above which we split (Å).
   *  Consecutive nucleotides sit ≈ 5–7 Å apart depending on geometry;
   *  8 Å is comfortably above that without merging across chain breaks. */
  nucleicGapThreshold: number
}

export const DEFAULT_BACKBONE_OPTIONS: BackboneOptions = {
  proteinGapThreshold: 4.5,
  nucleicGapThreshold: 8.0,
}

const PROTEIN_TRACE_NAMES = ['CA']
const NUCLEIC_TRACE_NAMES = ['C4\'', 'P']

export function extractBackbones(
  scene: MoleroScene,
  partial?: Partial<BackboneOptions>,
): BackboneSegment[] {
  const opts = { ...DEFAULT_BACKBONE_OPTIONS, ...partial }
  const segments: BackboneSegment[] = []
  const { residues, chains, attrs } = scene
  const { position, atomNameId } = attrs

  for (const chain of chains) {
    let entityType: 'protein' | 'rna' | 'dna'
    let traceNames: string[]
    let gapThreshold: number
    if (chain.entityType === 'protein') {
      entityType = 'protein'
      traceNames = PROTEIN_TRACE_NAMES
      gapThreshold = opts.proteinGapThreshold
    } else if (chain.entityType === 'rna') {
      entityType = 'rna'
      traceNames = NUCLEIC_TRACE_NAMES
      gapThreshold = opts.nucleicGapThreshold
    } else if (chain.entityType === 'dna') {
      entityType = 'dna'
      traceNames = NUCLEIC_TRACE_NAMES
      gapThreshold = opts.nucleicGapThreshold
    } else {
      continue
    }
    const traceNameIds = traceNames.map((n) => internAtomName(n))
    const gapThreshold2 = gapThreshold * gapThreshold

    // Walk residues in order; for each, scan its atoms for a trace atom.
    let currentPositions: number[] = []
    let currentAtoms: number[] = []
    let currentResidues: number[] = []
    let prevX = 0, prevY = 0, prevZ = 0
    let havePrev = false

    const finalizeSegment = () => {
      if (currentPositions.length >= 3 /* at least one point */) {
        if (currentPositions.length / 3 >= 2) {
          segments.push({
            chainIndex: chain.index,
            positions: Float32Array.from(currentPositions),
            atomIndex: Uint32Array.from(currentAtoms),
            residueIndex: Uint32Array.from(currentResidues),
            entityType,
          })
        }
      }
      currentPositions = []
      currentAtoms = []
      currentResidues = []
      havePrev = false
    }

    for (let r = chain.residueStart; r < chain.residueEnd; r++) {
      const res = residues[r]
      const traceAtomIdx = findTraceAtom(
        res.atomStart,
        res.atomEnd,
        traceNameIds,
        atomNameId,
      )
      if (traceAtomIdx < 0) {
        // Missing trace atom in this residue — break the segment here.
        finalizeSegment()
        continue
      }
      const px = position[traceAtomIdx * 3]
      const py = position[traceAtomIdx * 3 + 1]
      const pz = position[traceAtomIdx * 3 + 2]
      if (havePrev) {
        const dx = px - prevX, dy = py - prevY, dz = pz - prevZ
        if (dx * dx + dy * dy + dz * dz > gapThreshold2) {
          finalizeSegment()
        }
      }
      currentPositions.push(px, py, pz)
      currentAtoms.push(traceAtomIdx)
      currentResidues.push(r)
      prevX = px; prevY = py; prevZ = pz
      havePrev = true
    }
    finalizeSegment()
  }
  return segments
}

function findTraceAtom(
  atomStart: number,
  atomEnd: number,
  traceNameIds: number[],
  atomNameId: Uint32Array,
): number {
  // Prefer the first name in the list (e.g. C4' over P for nucleic acids).
  for (const want of traceNameIds) {
    for (let a = atomStart; a < atomEnd; a++) {
      if (atomNameId[a] === want) return a
    }
  }
  return -1
}
