/**
 * Scene — Molero's top-level data container.
 *
 * Two layers:
 *   - Topology (residues, chains, future bonds/interactions) — small,
 *     mostly structure-of-objects since the counts are low.
 *   - PropertyAttributes — a struct-of-typed-arrays keyed by atom index,
 *     designed for direct GPU upload as instance attributes. Per-atom
 *     biochemistry channels (CPK, vdW radius, formal charge, hybridization,
 *     aromatic / hydrophobic / donor / acceptor flags, pLDDT) all live
 *     here in colocated buffers.
 *
 * `RawAtomData`, `RawResidueData`, `RawChainData` are the parser's
 * output — pre-chemistry. `buildScene` runs the chemistry layer (valence
 * model + features) over them and assembles the `PropertyAttributes`
 * table the renderer consumes.
 */
import {
  CPK_COLORS,
  VDW_RADII,
  cpkColorToLinearRGB,
} from '../chemistry/elements'

// ─────────────────────────────────────────────────────────────────────────────
// Parser-output types — what mmCIF/PDB parsers produce before chemistry runs.

export interface RawAtomData {
  count: number
  /** Cartesian coords, one component per array (better cache locality
   *  for the typical "all positions" sweep). Will be packed [x,y,z]
   *  interleaved for GPU upload in PropertyAttributes.position. */
  x?: Float32Array
  y?: Float32Array
  z?: Float32Array
  atomicNumber?: Uint8Array
  atomNameId?: Uint32Array
  residueIndex?: Uint32Array
  chainIndex?: Uint16Array
  bfactor?: Float32Array
  occupancy?: Float32Array
  /** 1 = HETATM, 0 = ATOM. Distinguishes ligands/cofactors from polymer. */
  isHet?: Uint8Array
}

export interface RawResidueData {
  count: number
  chainIndex?: Uint16Array
  /** Three-letter code or CCD ID (e.g., 'ALA', 'HEM'). */
  compId?: string[]
  /** Sequence id within chain. */
  seqId?: Int32Array
  insCode?: string[]
  atomStart?: Uint32Array
  atomEnd?: Uint32Array
}

export interface RawChainData {
  count: number
  asymId?: string[]
  entityType?: ('protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown')[]
  residueStart?: Uint32Array
  residueEnd?: Uint32Array
}

// ─────────────────────────────────────────────────────────────────────────────
// Hybridization enum — used by the valence model and consumed as a
// material channel input. Packed into one byte per atom.

/** Hybridization enum-via-const-object (erasable-syntax compatible). */
export const Hybridization = {
  Unknown: 0,
  Sp: 1,
  Sp2: 2,
  Sp3: 3,
  /** Lone pair on sulfur, halogen, etc. — affects roughness mapping. */
  Lp: 4,
} as const
export type Hybridization = (typeof Hybridization)[keyof typeof Hybridization]

// ─────────────────────────────────────────────────────────────────────────────
// Feature flag bits — one bit per per-atom boolean. Packed into a single
// Uint8Array so a shader can sample one texel and bit-test cheaply.

/** Per-atom boolean bits, packed into one byte. */
export const AtomFlag = {
  None:             0,
  HydrogenDonor:    1 << 0,
  HydrogenAcceptor: 1 << 1,
  AromaticRing:     1 << 2,
  HydrophobicAtom:  1 << 3,
  PositiveCharge:   1 << 4,
  NegativeCharge:   1 << 5,
  TransitionMetal:  1 << 6,
  Backbone:         1 << 7,
} as const
export type AtomFlag = (typeof AtomFlag)[keyof typeof AtomFlag]

// ─────────────────────────────────────────────────────────────────────────────
// PropertyAttributes — the GPU-bound per-atom table.
//
// Layout invariants:
//   - All arrays have the same length = `count`.
//   - `position` is interleaved [x0,y0,z0, x1,y1,z1, …] for one-shot
//     upload as a vec3 attribute.
//   - `color` is interleaved linear-RGB [r0,g0,b0, r1,g1,b1, …].
//   - Per-atom scalars (`radius`, `bfactor`, `formalCharge`) are flat
//     Float32Arrays — one float per atom.
//   - Enum / flag fields use compact integer types so the upload cost is
//     proportional to information content.

export interface PropertyAttributes {
  count: number
  position: Float32Array       // [N * 3]
  /** vdW radius (Å). */
  radius: Float32Array         // [N]
  /** CPK color in linear-RGB. */
  color: Float32Array          // [N * 3]
  /** Per-atom B-factor (= pLDDT * 100 for Boltz output). */
  bfactor: Float32Array        // [N]
  /** Formal charge, signed (typically -2..+2). */
  formalCharge: Float32Array   // [N]
  /** Hybridization enum, one byte per atom. */
  hybridization: Uint8Array    // [N]
  /** Bitfield — AtomFlag combinations. */
  flags: Uint8Array            // [N]
  /** Total hydrogen count (implicit + explicit). */
  totalH: Uint8Array           // [N]
  /** Backreference for picking / labelling. */
  atomicNumber: Uint8Array     // [N]
  /** Interned atom name id (e.g. 'CA' → some int). Resolve via
   *  atomNameFromId in parsers/mmcif. Used by backbone extraction and
   *  any per-atom-name material routing. */
  atomNameId: Uint32Array      // [N]
  residueIndex: Uint32Array    // [N]
  chainIndex: Uint16Array      // [N]
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene — passed from parser → renderer.

export interface Residue {
  chainIndex: number
  compId: string
  seqId: number
  insCode: string
  atomStart: number
  atomEnd: number
  /** Filled later by Phase-2 DSSP-lite or by mmCIF `_struct_conf` parsing. */
  secondaryStructure?: 'helix' | 'sheet' | 'coil' | 'unknown'
}

export interface Chain {
  index: number
  asymId: string
  entityType: 'protein' | 'rna' | 'dna' | 'ligand' | 'water' | 'unknown'
  residueStart: number
  residueEnd: number
}

export interface Scene {
  attrs: PropertyAttributes
  residues: Residue[]
  chains: Chain[]
  /** Axis-aligned bounding box over atom positions. */
  bbox: { min: [number, number, number]; max: [number, number, number] }
  /** Centroid + bounding radius — convenience for camera framing. */
  center: [number, number, number]
  radius: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly — takes parser output, runs chemistry, packs PropertyAttributes.
// Imported here lazily-via-relative to avoid a circular import (scene.ts is
// the type module; chemistry/* implementations depend on it).

import { computeValenceModel } from '../chemistry/valence-model'
import { computeFeatureFlags } from '../chemistry/features'

export interface ParsedEntityGraph {
  atoms: RawAtomData
  residues: RawResidueData
  chains: RawChainData
}

export function buildScene(parsed: ParsedEntityGraph): Scene {
  const A = parsed.atoms.count
  const x = parsed.atoms.x!
  const y = parsed.atoms.y!
  const z = parsed.atoms.z!
  const atomicNumber = parsed.atoms.atomicNumber!
  const residueIndex = parsed.atoms.residueIndex!
  const chainIndex = parsed.atoms.chainIndex!
  const bfactor = parsed.atoms.bfactor!
  const atomNameId = parsed.atoms.atomNameId!

  // ── PropertyAttributes ──────────────────────────────────────────────────
  const position = new Float32Array(A * 3)
  const radius = new Float32Array(A)
  const color = new Float32Array(A * 3)
  const formalCharge = new Float32Array(A)
  const hybridization = new Uint8Array(A)
  const flags = new Uint8Array(A)
  const totalH = new Uint8Array(A)

  // Chemistry derivation — single pass over the atoms.
  const residues = unpackResidues(parsed.residues)
  const chains = unpackChains(parsed.chains)

  computeValenceModel({
    atomCount: A,
    atomicNumber,
    atomNameId,
    residueIndex,
    residueCompIds: residues.map((r) => r.compId),
    out: { formalCharge, hybridization, totalH },
  })

  computeFeatureFlags({
    atomCount: A,
    atomicNumber,
    atomNameId,
    residueIndex,
    residueCompIds: residues.map((r) => r.compId),
    totalH,
    formalCharge,
    out: flags,
  })

  // Pack position + color + radius. bbox running together.
  let minX = Infinity, minY = Infinity, minZ = Infinity
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity
  for (let i = 0; i < A; i++) {
    const px = x[i], py = y[i], pz = z[i]
    position[i * 3]     = px
    position[i * 3 + 1] = py
    position[i * 3 + 2] = pz
    radius[i] = VDW_RADII[atomicNumber[i]] ?? VDW_RADII[0]
    cpkColorToLinearRGB(atomicNumber[i], color, i * 3)
    if (px < minX) minX = px; if (px > maxX) maxX = px
    if (py < minY) minY = py; if (py > maxY) maxY = py
    if (pz < minZ) minZ = pz; if (pz > maxZ) maxZ = pz
  }
  if (A === 0) {
    minX = minY = minZ = -1
    maxX = maxY = maxZ = 1
  }
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2
  const dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ
  const bRadius = 0.5 * Math.hypot(dx, dy, dz)

  const attrs: PropertyAttributes = {
    count: A,
    position,
    radius,
    color,
    bfactor: new Float32Array(bfactor.buffer, bfactor.byteOffset, A).slice(),
    formalCharge,
    hybridization,
    flags,
    totalH,
    atomicNumber: new Uint8Array(atomicNumber.buffer, atomicNumber.byteOffset, A).slice(),
    atomNameId: new Uint32Array(atomNameId.buffer, atomNameId.byteOffset, A).slice(),
    residueIndex: new Uint32Array(residueIndex.buffer, residueIndex.byteOffset, A).slice(),
    chainIndex: new Uint16Array(chainIndex.buffer, chainIndex.byteOffset, A).slice(),
  }

  return {
    attrs,
    residues,
    chains,
    bbox: { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] },
    center: [cx, cy, cz],
    radius: bRadius,
  }
}

function unpackResidues(raw: RawResidueData): Residue[] {
  const n = raw.count
  if (!n) return []
  const out: Residue[] = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = {
      chainIndex: raw.chainIndex![i],
      compId: raw.compId![i],
      seqId: raw.seqId![i],
      insCode: raw.insCode![i],
      atomStart: raw.atomStart![i],
      atomEnd: raw.atomEnd![i],
    }
  }
  return out
}

function unpackChains(raw: RawChainData): Chain[] {
  const n = raw.count
  if (!n) return []
  const out: Chain[] = new Array(n)
  for (let i = 0; i < n; i++) {
    out[i] = {
      index: i,
      asymId: raw.asymId![i],
      entityType: raw.entityType![i],
      residueStart: raw.residueStart![i],
      residueEnd: raw.residueEnd![i],
    }
  }
  return out
}

// Suppress the unused-import lint on CPK_COLORS — we re-export it for
// downstream theming code.
export { CPK_COLORS, VDW_RADII }
