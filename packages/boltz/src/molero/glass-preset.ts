/**
 * Glass preset — bundled defaults for the SASA-modulated refractive
 * shell. Read at module-load from `glass-preset.json` and exposed both
 * as the bundled fallback and as the param shape the panel mutates.
 *
 * Two halves:
 *   - surface*: geometry-building params consumed by `buildGaussianSurface`
 *   - material*: PBR / TSL params consumed by `createGlassPass`
 *
 * The settings panel mutates a single flat object; `splitPreset` carves
 * it into the two halves for the call sites.
 */
import defaultPreset from './glass-preset.json'
import type { GaussianSurfaceOptions } from './passes/gaussian-surface'
import type { GlassPassOptions } from './passes/glass'

export interface GlassPreset {
  // Surface
  resolution: number
  probeRadius: number
  sigmaFactor: number
  isolation: number
  padding: number
  // Material
  ior: number
  thickness: number
  maxTransmission: number
  minTransmission: number
  roughnessMin: number
  roughnessMax: number
  clearcoat: number
  clearcoatRoughness: number
  envMapIntensity: number
  chainTintStrength: number
}

export const BUNDLED_GLASS_PRESET: GlassPreset = defaultPreset as GlassPreset

export function splitPreset(p: GlassPreset): {
  surface: Partial<GaussianSurfaceOptions>
  material: Partial<GlassPassOptions>
} {
  return {
    surface: {
      resolution: p.resolution,
      probeRadius: p.probeRadius,
      sigmaFactor: p.sigmaFactor,
      isolation: p.isolation,
      padding: p.padding,
    },
    material: {
      ior: p.ior,
      thickness: p.thickness,
      maxTransmission: p.maxTransmission,
      minTransmission: p.minTransmission,
      roughnessRange: { min: p.roughnessMin, max: p.roughnessMax },
      clearcoat: p.clearcoat,
      clearcoatRoughness: p.clearcoatRoughness,
      envMapIntensity: p.envMapIntensity,
      chainTintStrength: p.chainTintStrength,
    },
  }
}
