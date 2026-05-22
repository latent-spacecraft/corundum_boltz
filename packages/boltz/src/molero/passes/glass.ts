/**
 * Glass pass — physically-refractive gaussian-surface shell.
 *
 * Reads the per-vertex `aGlass` vec4 baked by `buildGaussianSurface`
 * (now Gaussian-blended across nearby atoms, smooth transitions):
 *   .x = SASA / sasaReference          (0 buried .. 1 fully exposed)
 *   .y = hydrophobicity Kyte-Doolittle (0 hydrophilic .. 1 hydrophobic)
 *   .z = formalCharge biased to [0, 1] (0.5 = neutral)
 *   .w = aromaticness fraction         (0 = no aromatic / metal atoms
 *                                       contributing; 1 = vertex sits
 *                                       fully under aromatic / metal density)
 *
 * Routes them into MeshPhysicalNodeMaterial:
 *   - transmission    ← SASA           (exposed faces are clearer glass;
 *                                       deeply buried faces stay opaque so
 *                                       the protein core reads as solid mass)
 *   - roughness       ← hydrophobicity (Kyte-Doolittle: hydrophobic →
 *                                       polished gem; hydrophilic → frosted)
 *   - metalness       ← aromaticness × 0.5 (capped — full metal would
 *                                       drown the per-vertex tint)
 *   - emission tint   ← formalCharge   (positive → cool blue halo,
 *                                       negative → warm red halo)
 */
import {
  Color,
  Mesh,
  MeshPhysicalNodeMaterial,
  type BufferGeometry,
} from 'three/webgpu'
import {
  abs,
  clamp,
  float,
  mix,
  nodeObject,
  saturate,
  select,
  uniform,
  vec3,
  vertexColor,
} from 'three/tsl'

export interface GlassPassOptions {
  /** Base material color (white = let chemistry drive). */
  baseColor: number
  /** IOR — water 1.33, quartz 1.46, sapphire 1.77. 1.52 = window glass. */
  ior: number
  /** Volumetric thickness for refraction depth (Å). */
  thickness: number
  /** Max transmission (multiplied by SASA). */
  maxTransmission: number
  /** Min transmission applied even to buried atoms (so the shell isn't
   *  fully opaque in the protein core). */
  minTransmission: number
  /** Roughness band — `roughnessFromHydrophobicity` blends between these.
   *  Hydrophilic (frosted) → max; hydrophobic (polished gem) → min. */
  roughnessRange: { min: number; max: number }
  /** Clearcoat over the gem — second polished layer. */
  clearcoat: number
  clearcoatRoughness: number
  envMapIntensity: number
  /**
   * 0..1 blend amount for the per-vertex chain tint. 0 = pure white
   * (chemistry colors only); 1 = full chain color (gem tinted by chain).
   * The tint is multiplicative against the chemistry-driven base, so
   * SASA / hydrophobicity / charge still show through.
   */
  chainTintStrength: number
}

export const DEFAULT_GLASS_OPTIONS: GlassPassOptions = {
  baseColor: 0xffffff,
  ior: 1.52,
  thickness: 0.4,
  maxTransmission: 0.92,
  minTransmission: 0.25,
  roughnessRange: { min: 0.05, max: 0.55 },
  clearcoat: 1.0,
  clearcoatRoughness: 0.08,
  envMapIntensity: 1.2,
  chainTintStrength: 0.35,
}

export interface GlassPassResources {
  mesh: Mesh
  material: MeshPhysicalNodeMaterial
  geometry: BufferGeometry
  dispose: () => void
}

export function createGlassPass(
  geometry: BufferGeometry,
  partial?: Partial<GlassPassOptions>,
): GlassPassResources {
  const opts = { ...DEFAULT_GLASS_OPTIONS, ...partial }

  const material = new MeshPhysicalNodeMaterial({
    color: new Color(opts.baseColor),
    transparent: false, // transmission backend handles alpha
    side: 2, // DoubleSide enum value — render both faces (cavities, etc.)
  })
  material.ior = opts.ior
  material.thickness = opts.thickness
  material.envMapIntensity = opts.envMapIntensity

  // ── Per-vertex chemistry vec4 ────────────────────────────────────────
  // The custom `aGlass` attribute is registered with BufferGeometry as a
  // standard vertex attribute (not instanced), so `vertexColor('aGlass')`
  // — sorry, vertexColor reads named color attribute via the TSL helper.
  // Three.js's TSL exposes `attribute(name)` for arbitrary attributes.
  // Since `attribute` from three/tsl isn't typed to itemSize, we cast.
  const aGlass = nodeObject(
    (attributeNode as any)('aGlass') as any,
  ) as any
  const aChainTint = nodeObject(
    (attributeNode as any)('aChainTint') as any,
  ) as any
  const sasaN = aGlass.x
  const hydroN = aGlass.y
  const chargeN = aGlass.z
  const aromaticN = aGlass.w

  // ── Transmission ─────────────────────────────────────────────────────
  // mix(minTransmission, maxTransmission, sasaN)
  const transmissionNode = mix(
    float(opts.minTransmission),
    float(opts.maxTransmission),
    saturate(sasaN),
  )
  material.transmissionNode = transmissionNode

  // ── Roughness ────────────────────────────────────────────────────────
  // Hydrophobic (hydroN → 1) = polished (roughness.min).
  // Hydrophilic (hydroN → 0) = frosted (roughness.max).
  const roughnessNode = mix(
    float(opts.roughnessRange.max),
    float(opts.roughnessRange.min),
    saturate(hydroN),
  )
  material.roughnessNode = clamp(roughnessNode, float(0.02), float(0.95))

  // ── Metalness ────────────────────────────────────────────────────────
  // aGlass.w is the in-neighborhood aromatic fraction (with transition-
  // metal atoms contributing a 2.5× weight bump on the rasterizer side,
  // pre-clamped to [0, 1]). Half-strength here keeps the vertex tint
  // legible — full metalness would mute the chemistry colors entirely.
  material.metalnessNode = clamp(saturate(aromaticN).mul(0.5), float(0), float(1))

  // ── Emission — charge halo ──────────────────────────────────────────
  // chargeN: 0.5 = neutral, > 0.5 = positive (cool tint), < 0.5 = negative (warm).
  // Emission strength scales with |charge|.
  const chargeSigned = chargeN.sub(0.5).mul(2) // [-1, 1]
  const chargeMag = saturate(abs(chargeSigned))
  const positiveTint = vec3(0.20, 0.40, 1.00)
  const negativeTint = vec3(1.00, 0.30, 0.20)
  const chargeTint = select(chargeSigned.greaterThan(0), positiveTint, negativeTint)
  material.emissiveNode = chargeTint.mul(chargeMag).mul(0.3)

  // ── Base color tint — per-vertex chain palette ───────────────────────
  // The default white base (material.color = 0xffffff) means the gem is
  // chemistry-tinted only. Blending the chain palette in multiplicatively
  // means SASA / hydrophobicity / charge channels stay visible; the
  // chain just biases the underlying hue.
  const whiteVec = vec3(1, 1, 1)
  material.colorNode = mix(whiteVec, aChainTint, uniform(opts.chainTintStrength))

  // ── Clearcoat — applied uniformly via uniforms ───────────────────────
  material.clearcoatNode = uniform(opts.clearcoat)
  material.clearcoatRoughnessNode = uniform(opts.clearcoatRoughness)

  // Suppress unused-import warning in the bundler when vertexColor isn't
  // taken; we keep the import in case a follow-up uses it for chain hue.
  void vertexColor

  const mesh = new Mesh(geometry, material)
  mesh.frustumCulled = false
  // Render after opaque geometry so transmission samples the correct
  // composited backdrop.
  mesh.renderOrder = 10

  return {
    mesh,
    material,
    geometry,
    dispose: () => {
      material.dispose()
      // Caller owns the geometry (came from buildGaussianSurface) — they
      // dispose it via the surface's own dispose handle.
    },
  }
}

// TSL's `attribute(name)` helper isn't typed in our `three/tsl` re-export
// — it's exported as `attribute` from the runtime. We import it here under
// a different name to avoid colliding with the React `attribute` HTML attr
// noun used elsewhere; the cast in createGlassPass narrows it to ShaderNodeObject.
import { attribute as attributeNode } from 'three/tsl'
