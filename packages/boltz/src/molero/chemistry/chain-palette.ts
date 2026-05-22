/**
 * Chain palette — eight perceptually distinct hues for distinguishing
 * polymer chains. Single source of truth for both the cartoon ribbon
 * and any other pass that wants per-chain coloring (currently: the
 * glass surface's per-vertex tint).
 *
 * Cycles after eight chains. Biological assemblies with more chains
 * usually want a custom palette anyway, but the cycle is at least
 * deterministic (chain 0 and chain 8 share a hue).
 */

export const CHAIN_PALETTE_HEX: readonly number[] = [
  0xd4a557, // warm gold
  0x4fa3c7, // teal blue
  0xc15a7c, // dusty rose
  0x6fc76f, // mint green
  0xb18ad0, // soft violet
  0xe0985a, // burnt orange
  0x82a8e0, // pale azure
  0xc7d44f, // chartreuse
] as const

export function chainColorHex(chainIndex: number): number {
  return CHAIN_PALETTE_HEX[chainIndex % CHAIN_PALETTE_HEX.length]
}

/** sRGB hex → linear-RGB float triple. */
export function chainColorLinearRGB(chainIndex: number): [number, number, number] {
  const hex = chainColorHex(chainIndex)
  return [
    srgbToLinear(((hex >> 16) & 0xff) / 255),
    srgbToLinear(((hex >> 8) & 0xff) / 255),
    srgbToLinear((hex & 0xff) / 255),
  ]
}

function srgbToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/** Precomputed linear-RGB palette as a Float32Array of [r0,g0,b0,r1,g1,b1,…]
 *  — handy when you need to index a lot per-atom or per-vertex. */
export const CHAIN_PALETTE_LINEAR: Float32Array = (() => {
  const arr = new Float32Array(CHAIN_PALETTE_HEX.length * 3)
  for (let i = 0; i < CHAIN_PALETTE_HEX.length; i++) {
    const [r, g, b] = chainColorLinearRGB(i)
    arr[i * 3]     = r
    arr[i * 3 + 1] = g
    arr[i * 3 + 2] = b
  }
  return arr
})()
