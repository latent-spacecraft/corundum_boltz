/**
 * Element data — atomic number ↔ symbol, vdW radii, covalent radii, CPK
 * color palette. Pure data + branchless lookups; no allocations after
 * module load.
 *
 * Atomic numbers are 1-indexed (H = 1). Index 0 is reserved as the
 * "unknown element" slot with sensible defaults.
 *
 * Sources:
 *   - vdW radii: Bondi (1964), with later additions from Mantina et al.
 *     (2009). Values in Å.
 *   - Covalent radii: Cordero et al. (2008), single-bond radii. In Å.
 *   - CPK colors: standard Corey-Pauling-Koltun + Jmol extensions.
 *     Hex RGB, 0xRRGGBB.
 *
 * The tables cover Z=1..103 (H through Lr). Atoms beyond that fall back
 * to the unknown slot — fine for biomolecules.
 */

// ─────────────────────────────────────────────────────────────────────────────
// Symbol ↔ atomic number

/** Index = atomic number (1-based). Index 0 = unknown. */
export const ELEMENT_SYMBOLS: readonly string[] = [
  'X',
  'H',  'He',
  'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
  'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni',
  'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
  'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
  'Cs', 'Ba',
  'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
  'Ho', 'Er', 'Tm', 'Yb', 'Lu',
  'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
  'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
  'Fr', 'Ra',
  'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
  'Es', 'Fm', 'Md', 'No', 'Lr',
]

const SYMBOL_TO_NUMBER: ReadonlyMap<string, number> = (() => {
  const m = new Map<string, number>()
  for (let z = 0; z < ELEMENT_SYMBOLS.length; z++) {
    m.set(ELEMENT_SYMBOLS[z], z)
    // Allow uppercase lookup ('FE' as well as 'Fe') for sloppy PDB input.
    m.set(ELEMENT_SYMBOLS[z].toUpperCase(), z)
  }
  return m
})()

/** Returns the atomic number for an element symbol, or 0 if unknown. */
export function atomicNumberFromSymbol(symbol: string): number {
  if (!symbol) return 0
  // Try the canonical case first (cheap path).
  const direct = SYMBOL_TO_NUMBER.get(symbol)
  if (direct !== undefined) return direct
  // PDB element columns are sometimes uppercase ('CA' for calcium vs Cα atom
  // confusion is the caller's problem); we still try the upper variant.
  return SYMBOL_TO_NUMBER.get(symbol.toUpperCase()) ?? 0
}

// ─────────────────────────────────────────────────────────────────────────────
// vdW radii (Å) — Bondi 1964 + Mantina 2009 extensions.
// Index 0 is the unknown-element fallback (1.7 Å, a carbon-ish guess).
// Zero entries below mean "no published value"; we backfill at the bottom.

const VDW_DATA: number[] = (() => {
  const r = new Array<number>(104).fill(0)
  // Common biomolecule elements first.
  r[0]  = 1.7  // unknown fallback
  r[1]  = 1.20  // H
  r[2]  = 1.40  // He
  r[3]  = 1.82; r[4]  = 1.53; r[5]  = 1.92; r[6]  = 1.70  // Li Be B C
  r[7]  = 1.55; r[8]  = 1.52; r[9]  = 1.47; r[10] = 1.54  // N O F Ne
  r[11] = 2.27; r[12] = 1.73; r[13] = 1.84; r[14] = 2.10  // Na Mg Al Si
  r[15] = 1.80; r[16] = 1.80; r[17] = 1.75; r[18] = 1.88  // P S Cl Ar
  r[19] = 2.75; r[20] = 2.31                              // K Ca
  // First-row transition metals
  r[21] = 2.11; r[22] = 1.95; r[23] = 1.93; r[24] = 1.96
  r[25] = 1.96; r[26] = 1.96; r[27] = 1.92; r[28] = 1.63
  r[29] = 1.40; r[30] = 1.39                              // Cu Zn
  r[31] = 1.87; r[32] = 2.11; r[33] = 1.85; r[34] = 1.90  // Ga Ge As Se
  r[35] = 1.85; r[36] = 2.02                              // Br Kr
  r[37] = 3.03; r[38] = 2.49
  r[42] = 2.00  // Mo (a guess from Mantina)
  r[44] = 2.05  // Ru
  r[46] = 2.05  // Pd
  r[47] = 1.72  // Ag
  r[48] = 1.58  // Cd
  r[49] = 1.93; r[50] = 2.17; r[51] = 2.06; r[52] = 2.06  // In Sn Sb Te
  r[53] = 1.98; r[54] = 2.16                              // I Xe
  r[55] = 3.43; r[56] = 2.68
  r[78] = 1.75  // Pt
  r[79] = 1.66  // Au
  r[80] = 1.55  // Hg
  r[82] = 2.02  // Pb
  // Backfill remaining transition metals + lanthanides/actinides with 2.0 Å.
  for (let z = 1; z < r.length; z++) if (r[z] === 0) r[z] = 2.0
  return r
})()

export const VDW_RADII: Float32Array = Float32Array.from(VDW_DATA)

/** Returns vdW radius in Å. Unknown elements fall back to 1.7 Å. */
export function vdwRadius(atomicNumber: number): number {
  if (atomicNumber < 0 || atomicNumber >= VDW_RADII.length) return VDW_RADII[0]
  return VDW_RADII[atomicNumber]
}

// ─────────────────────────────────────────────────────────────────────────────
// Covalent radii (Å) — Cordero et al. 2008. Used for distance-based bond
// perception in slice 1.2.

const COV_DATA: number[] = (() => {
  const r = new Array<number>(104).fill(0)
  r[0]  = 0.76  // unknown ≈ carbon
  r[1]  = 0.31
  r[2]  = 0.28
  r[3]  = 1.28; r[4]  = 0.96; r[5]  = 0.84; r[6]  = 0.76
  r[7]  = 0.71; r[8]  = 0.66; r[9]  = 0.57; r[10] = 0.58
  r[11] = 1.66; r[12] = 1.41; r[13] = 1.21; r[14] = 1.11
  r[15] = 1.07; r[16] = 1.05; r[17] = 1.02; r[18] = 1.06
  r[19] = 2.03; r[20] = 1.76
  r[21] = 1.70; r[22] = 1.60; r[23] = 1.53; r[24] = 1.39
  r[25] = 1.50; r[26] = 1.42; r[27] = 1.38; r[28] = 1.24
  r[29] = 1.32; r[30] = 1.22
  r[31] = 1.22; r[32] = 1.20; r[33] = 1.19; r[34] = 1.20
  r[35] = 1.20; r[36] = 1.16
  r[37] = 2.20; r[38] = 1.95
  r[42] = 1.54; r[44] = 1.46; r[46] = 1.39; r[47] = 1.45; r[48] = 1.44
  r[49] = 1.42; r[50] = 1.39; r[51] = 1.39; r[52] = 1.38
  r[53] = 1.39; r[54] = 1.40
  r[55] = 2.44; r[56] = 2.15
  r[78] = 1.36; r[79] = 1.36; r[80] = 1.32
  r[82] = 1.46
  for (let z = 1; z < r.length; z++) if (r[z] === 0) r[z] = 1.50
  return r
})()

export const COVALENT_RADII: Float32Array = Float32Array.from(COV_DATA)

export function covalentRadius(atomicNumber: number): number {
  if (atomicNumber < 0 || atomicNumber >= COVALENT_RADII.length) return COVALENT_RADII[0]
  return COVALENT_RADII[atomicNumber]
}

// ─────────────────────────────────────────────────────────────────────────────
// CPK + Jmol color palette. Hex 0xRRGGBB.
//
// Index 0 = unknown (default pink). Values follow Jmol's published table
// (https://jmol.sourceforge.net/jscolors/) which extends classic CPK with
// distinguishable hues for transition metals and lanthanides.

const CPK_DATA: number[] = (() => {
  const c = new Array<number>(104).fill(0xff1493) // unknown = deep pink
  c[1]  = 0xffffff  // H — white
  c[2]  = 0xd9ffff  // He
  c[3]  = 0xcc80ff; c[4]  = 0xc2ff00; c[5]  = 0xffb5b5; c[6]  = 0x909090  // Li Be B C
  c[7]  = 0x3050f8; c[8]  = 0xff0d0d; c[9]  = 0x90e050; c[10] = 0xb3e3f5  // N O F Ne
  c[11] = 0xab5cf2; c[12] = 0x8aff00; c[13] = 0xbfa6a6; c[14] = 0xf0c8a0  // Na Mg Al Si
  c[15] = 0xff8000; c[16] = 0xffff30; c[17] = 0x1ff01f; c[18] = 0x80d1e3  // P S Cl Ar
  c[19] = 0x8f40d4; c[20] = 0x3dff00                                       // K Ca
  c[21] = 0xe6e6e6; c[22] = 0xbfc2c7; c[23] = 0xa6a6ab; c[24] = 0x8a99c7
  c[25] = 0x9c7ac7; c[26] = 0xe06633; c[27] = 0xf090a0; c[28] = 0x50d050
  c[29] = 0xc88033; c[30] = 0x7d80b0                                       // Cu Zn
  c[31] = 0xc28f8f; c[32] = 0x668f8f; c[33] = 0xbd80e3; c[34] = 0xffa100
  c[35] = 0xa62929; c[36] = 0x5cb8d1
  c[37] = 0x702eb0; c[38] = 0x00ff00
  c[42] = 0x54b5b5; c[44] = 0x248f8f; c[46] = 0x006985; c[47] = 0xc0c0c0
  c[48] = 0xffd98f
  c[49] = 0xa67573; c[50] = 0x668080; c[51] = 0x9e63b5; c[52] = 0xd47a00
  c[53] = 0x940094; c[54] = 0x429eb0
  c[55] = 0x57178f; c[56] = 0x00c900
  c[78] = 0xd0d0e0; c[79] = 0xffd123; c[80] = 0xb8b8d0                     // Pt Au Hg
  c[82] = 0x575961
  for (let z = 1; z < c.length; z++) if (c[z] === 0) c[z] = 0xff1493
  return c
})()

export const CPK_COLORS: Uint32Array = Uint32Array.from(CPK_DATA)

/** Returns the CPK color as a packed 0xRRGGBB integer. */
export function cpkColor(atomicNumber: number): number {
  if (atomicNumber < 0 || atomicNumber >= CPK_COLORS.length) return CPK_COLORS[0]
  return CPK_COLORS[atomicNumber]
}

/** Writes the CPK color into out[offset..offset+3] as linear-RGB [0,1] floats. */
export function cpkColorToLinearRGB(
  atomicNumber: number,
  out: Float32Array,
  offset: number,
): void {
  const hex = cpkColor(atomicNumber)
  // sRGB → linear conversion: x ≤ 0.04045 ? x/12.92 : ((x + 0.055)/1.055)^2.4
  // Inline for speed since this fires per-atom at load time.
  const r = ((hex >> 16) & 0xff) / 255
  const g = ((hex >> 8) & 0xff) / 255
  const b = (hex & 0xff) / 255
  out[offset]     = srgbToLinear(r)
  out[offset + 1] = srgbToLinear(g)
  out[offset + 2] = srgbToLinear(b)
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

// ─────────────────────────────────────────────────────────────────────────────
// Element classification helpers — used by valence model + features.

const METAL_SET = new Set<number>([
  3, 4, 11, 12, 13, 19, 20, // alkali, alkaline-earth, Al
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, // 1st-row TM
  37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, // 2nd-row
  55, 56, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, // 3rd-row
])
const TRANSITION_METAL_SET = new Set<number>([
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
  39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  72, 73, 74, 75, 76, 77, 78, 79, 80,
])

export function isMetal(atomicNumber: number): boolean {
  return METAL_SET.has(atomicNumber)
}
export function isTransitionMetal(atomicNumber: number): boolean {
  return TRANSITION_METAL_SET.has(atomicNumber)
}
export function isHydrogen(atomicNumber: number): boolean {
  return atomicNumber === 1
}
export function isHalogen(atomicNumber: number): boolean {
  return atomicNumber === 9 || atomicNumber === 17 || atomicNumber === 35 || atomicNumber === 53
}
