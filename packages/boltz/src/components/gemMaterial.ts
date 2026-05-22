/**
 * Shared gem material factory — Threekit-style PBR transmission, tuned for
 * a single convex hull rendered in our local-unit scene (≈ 1 unit across).
 *
 * Threekit's gem shader knobs (see the platform-documentation "gem material"
 * page) collapse cleanly onto Three.js `MeshPhysicalMaterial`:
 *
 *   Threekit             →  Three.js
 *   ─────────────────────────────────────────────────────────────────────
 *   Color                →  color + attenuationColor (refraction tint)
 *   Attenuation Distance →  attenuationDistance × thickness (path length)
 *   IOR                  →  ior          (default 2.417 = diamond;
 *                                         1.762–1.778 for corundum)
 *   Roughness            →  roughness    (subtle; pure 0 over-mirrors)
 *   Abbe Number          →  dispersion   (inverse — high Abbe = low dispersion)
 *
 * Critical gotcha: Three.js applies attenuation along the volumetric path
 * through the mesh, scaled by `thickness`. Setting thickness=0 disables the
 * absorption entirely no matter what attenuationColor / attenuationDistance
 * say. Default here is 1.0 — proportional to the local-unit gem size.
 *
 * Use the same factory for the structure-viewer "gem shell" so logo and
 * cofold render share one visual register.
 */
import * as THREE from 'three'

export interface GemMaterialOpts {
  /** Base/refraction tint. Required. */
  color: THREE.ColorRepresentation
  /**
   * Distance over which transmitted light loses ~63% (1/e) of its intensity
   * to the attenuationColor. In local scene units. Threekit recommends
   * 0.001–0.1 m for real gems; for our ≈ 1-unit gem a value of 0.3–0.8 gives
   * comparable saturation. Default 0.5.
   */
  attenuationDistance?: number
  /**
   * Colour absorbed along the transmitted path. Defaults to `color` — pick
   * a slightly more saturated value if you want a darker edge / lighter
   * centre look (ruby, garnet); equal-to-color gives a uniform tint.
   */
  attenuationColor?: THREE.ColorRepresentation
  /**
   * Index of refraction. 1.77 for corundum (ruby & sapphire). Diamond 2.417.
   */
  ior?: number
  /**
   * Microsurface roughness on the facets. Pure 0 produces a mirror-like
   * specular that's hard to read on a small thumbnail. ~0.02 gives a soft
   * facet highlight without losing the polished look.
   */
  roughness?: number
  /**
   * Chromatic dispersion — fakes Abbe-number behaviour. Three.js wires this
   * into the transmission shader directly. 0 = none; 0.2 = noticeable spectral
   * fringe at facet edges; 1+ = obvious rainbow (good for diamond, garish
   * for corundum). Default 0.2.
   */
  dispersion?: number
  /**
   * Local-unit thickness used to scale the attenuation path. Should be on
   * the same order as the mesh's local size — 1.0 for our ≈ 1-unit gems.
   * Set to 0 to disable absorption (rendering as a colored prism only).
   */
  thickness?: number
  /**
   * 0..1 fraction of incident light that refracts through. 1.0 = full glass.
   * Lower values let some specular reflection / diffuse term show through.
   */
  transmission?: number
  /**
   * Multiplier on the environment contribution. 1.0 matches the PMREM probe
   * intensity; raise for showier highlights on a small thumbnail.
   */
  envMapIntensity?: number
  /**
   * Render both sides — required so internal reflections off the back of
   * the gem read correctly through the front facets.
   */
  side?: THREE.Side
}

/**
 * Build a fresh MeshPhysicalMaterial configured for gem-style transmission.
 * Returns a new instance every call — caller is responsible for disposal.
 */
export function createGemMaterial(opts: GemMaterialOpts): THREE.MeshPhysicalMaterial {
  const color = new THREE.Color(opts.color)
  const attenuationColor = new THREE.Color(opts.attenuationColor ?? color)
  return new THREE.MeshPhysicalMaterial({
    color,
    metalness: 0,
    roughness: opts.roughness ?? 0.02,
    ior: opts.ior ?? 1.77,
    transmission: opts.transmission ?? 1.0,
    thickness: opts.thickness ?? 1.0,
    attenuationDistance: opts.attenuationDistance ?? 0.5,
    attenuationColor,
    dispersion: opts.dispersion ?? 0.2,
    envMapIntensity: opts.envMapIntensity ?? 1.0,
    transparent: true,
    side: opts.side ?? THREE.DoubleSide,
  })
}

/**
 * Pre-tuned presets matching the Corundum palette. The `ruby` tone is the
 * one the logo uses — kept here so the structure-shell overlay can pick
 * the same constants without redefining.
 */
export const GEM_PRESETS = {
  ruby: {
    color: new THREE.Color(0.85, 0.04, 0.07),
    attenuationColor: new THREE.Color(1.0, 0.05, 0.08),
    ior: 1.77,
    attenuationDistance: 0.5,
    dispersion: 0.2,
  },
  sapphire: {
    color: new THREE.Color(0.05, 0.12, 0.55),
    attenuationColor: new THREE.Color(0.04, 0.10, 0.65),
    ior: 1.77,
    attenuationDistance: 0.5,
    dispersion: 0.2,
  },
  diamond: {
    color: new THREE.Color(1.0, 1.0, 1.0),
    attenuationColor: new THREE.Color(1.0, 1.0, 1.0),
    ior: 2.42,
    attenuationDistance: 5.0, // very low absorption — diamond is near-perfect
    dispersion: 1.0,           // diamond's signature fire
  },
  quartz: {
    // Clear quartz — near-colorless, glass-like IOR, mild dispersion.
    // Reads as "polished crystal lens" rather than tinted gem; the chain
    // inside refracts through unobscured.
    color: new THREE.Color(0.96, 0.98, 1.0),
    attenuationColor: new THREE.Color(0.94, 0.97, 1.0),
    ior: 1.54,
    attenuationDistance: 8.0, // very low absorption
    dispersion: 0.05,
  },
  emerald: {
    color: new THREE.Color(0.05, 0.55, 0.20),
    attenuationColor: new THREE.Color(0.10, 0.70, 0.30),
    ior: 1.58,
    attenuationDistance: 0.8,
    dispersion: 0.1,
  },
} satisfies Record<string, Partial<GemMaterialOpts> & { color: THREE.Color }>

export type GemPresetName = keyof typeof GEM_PRESETS
