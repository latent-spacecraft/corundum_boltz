# Project Molero

*A bespoke WebGL rendering engine for biomolecules — where biochemistry is the material.*

---

## Vision

Existing browser molecular viewers (Mol\*, JSmol, NGL) render molecules as inventoried atoms with discrete property→colour mappings. Each property gets a "mode": colour-by-chain, colour-by-pLDDT, colour-by-hydrophobicity. Pick one; the rest collapse to neutral.

Molero treats biochemical properties as **first-class material channels**. Where a polished gem reads as glass and matte plastic reads as plastic because the surface, transmission, absorption, and microroughness encode that *physically*, Molero composes biological molecules from PBR primitives whose roughness, metalness, emission, transmission, dispersion, and volumetric channels each carry a continuous biochemical signal. Multiple properties read simultaneously — a single rendering shows colour=secondary-structure, roughness=hydrophobicity, emission=pLDDT, transmission=SASA, with a Gaussian electrostatic glow bleeding into the solvent space around the protein.

The result is rendering that doesn't merely *display* the molecule but *makes its biology visible* without legends or toggling.

This document is the spec. Implementation lands incrementally as numbered phases.

---

## Design principles

### 1. Property channels, not switches.
Visualization "modes" elsewhere are discrete. Molero composes channels — a scene runs N maps simultaneously. The user assigns properties to material channels via a configuration layer; the renderer interprets the composite continuously.

### 2. Continuous space, not just discrete primitives.
Atoms are spheres and bonds are sticks when that's the right rendering. But the molecule is also a 3D scalar/vector field — atom density, electrostatic potential, partial-charge distribution, conservation, SASA, B-factor. Molero renders iso-surfaces, raymarched volumes, and streamlines through that field as first-class passes. Discrete primitives become one render mode among many.

### 3. Physics-based light.
Real image-based lighting (PMREM probe). Real transmission with IOR and dispersion. Real volumetric absorption. Real subsurface scattering. The image is shaded by photometric light interacting with material — exactly how the eye reads a polished gem vs. a frosted plastic. Biological properties piggyback on the channels every rendering artist already understands; the visual vocabulary doesn't need to be re-learned.

### 4. Browser-native, BYOD, WebGPU-required.
No server, no plugin, no install — same runtime ethos as the rest of Corundum. WebGPU is a **hard requirement**, not a progressive enhancement: anyone running Boltz already has a capable GPU (Apple Silicon, discrete or modern integrated), and accepting that floor lets every render pass assume compute shaders, storage buffers, and float-32 3D textures without WebGL2 fallback complexity. Devices without WebGPU fall back to the existing Mol\* path until they upgrade.

### 5. Composable, not monolithic.
Render passes are pluggable. Property-channel mappings are configuration. New maps (e.g. "encode mutation rate as metalness") land as data, not engine changes. The spec is the contract; passes implement an interface.

### 6. Read-only.
Molero is a viewer. No modeling, no bond editing, no cheminformatics computation. RDKit-WASM, prediction engines, etc. live elsewhere in the stack and feed Molero their outputs.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Input layer                                                    │
│   - mmCIF / PDB / SDF parser → entity graph                    │
│   - Property loaders (per-atom, per-residue, per-pair)         │
│   - External scalar/vector fields (DX, CCP4, MAP grids)        │
│   - Trajectory loaders (XTC, DCD, multi-model PDB)             │
├────────────────────────────────────────────────────────────────┤
│ Scene graph                                                    │
│   - Entities: chain, residue, atom, ligand, bond, interaction  │
│   - Property maps: each entity owns N channels                 │
│   - Spatial index: KD-tree (atom queries), BVH (ray casts)     │
├────────────────────────────────────────────────────────────────┤
│ Render passes                                                  │
│   Discrete:                                                    │
│     · sphere (atoms)                                           │
│     · stick (bonds)                                            │
│     · ribbon / cartoon (secondary structure)                   │
│     · putty (variable-thickness tube)                          │
│   Continuous:                                                  │
│     · marching-cubes (Gaussian atomic density)                 │
│     · SES / SAS (Connolly probe rolling)                       │
│     · alpha-shape                                              │
│     · raymarched volume (3D texture, e.g. electrostatics)      │
│     · Gaussian splat (volumetric per-atom contribution)        │
│     · streamline (vector field, e.g. electric field lines)     │
│   Composite:                                                   │
│     · silhouette / outline                                     │
│     · H-bond / contact beams                                   │
│     · distance-field labels                                    │
│     · pocket / void glow                                       │
├────────────────────────────────────────────────────────────────┤
│ Material system                                                │
│   - PBR base (MeshPhysicalMaterial-derived)                    │
│     transmission, IOR, dispersion, attenuation, SSS, clearcoat │
│   - Per-vertex / per-fragment property attributes              │
│   - 3D-texture lookups for volumetric props                    │
│   - onBeforeCompile shader chunks for biochemistry maths       │
├────────────────────────────────────────────────────────────────┤
│ Compositor                                                     │
│   - HDR rendering, ACES tone mapping                           │
│   - Bloom, depth-of-field, SSAO                                │
│   - Selection halos, motion blur                               │
└────────────────────────────────────────────────────────────────┘
```

---

## Property → channel mapping vocabulary

The eye reads these in approximately this order, so the mapping budget should respect it: colour first, then geometric form, then highlight/spec, then emission, then volumetrics.

### Colour / albedo
| Property            | Map                                              |
|---------------------|--------------------------------------------------|
| Element             | CPK palette (default)                            |
| Chain ID            | distinguishable per-chain hue                    |
| Residue type        | polar/hydrophobic/charged class colouring        |
| Custom scalar       | any property + configurable colormap (viridis, …)|

### Roughness  *(microsurface — how rough the eye reads the substance)*
| Property             | Map                                                |
|----------------------|----------------------------------------------------|
| Hydrophobicity       | hydrophobic → polished glass; hydrophilic → matte  |
| SASA                 | buried atoms smooth; exposed atoms textured        |
| B-factor             | high-motion → frosted                              |
| Mutation rate        | conserved → polished, variable → eroded            |

### Metalness  *(dielectric vs metal — also reads as "weight" / "importance")*
| Property              | Map                                              |
|-----------------------|--------------------------------------------------|
| Conservation          | highly-conserved → gold/copper; variable → matte |
| Catalytic / functional| active-site residues opt into full metalness     |
| Disulfides / coord.   | literal metal look at coordination sites         |

### Emission  *(self-luminance — direct the eye)*
| Property                  | Map                                            |
|---------------------------|------------------------------------------------|
| pLDDT / model confidence  | low-confidence → soft red pulse                |
| Stabilisation energy      | strained residues → hot orange                 |
| Active site / pharmacophore | subtle pulsing tag                           |
| Spectroscopic chromophore | emit at the chromophore's absorption peak      |

### Transmission / opacity  *(physical see-through)*
| Property              | Map                                              |
|-----------------------|--------------------------------------------------|
| SASA                  | deeply buried opaque; surface translucent        |
| Selection state       | unselected entities semi-transparent             |
| Depth gating          | distance-from-camera fade for through-views      |

### Volumetric  *(raymarched 3D fields — the "biology in the empty space")*
| Property                 | Map                                            |
|--------------------------|------------------------------------------------|
| Electrostatic potential  | positive vol → blue, negative → red, iso glow  |
| Partial charge density   | Gaussian splat per atom weighted by charge     |
| Pocket / cavity          | hollow regions render as inverted glow         |
| CryoEM / X-ray density   | direct volumetric from MAP/CCP4 grid           |

### Geometry modulation  *(shape conveys signal too)*
| Property             | Map                                                  |
|----------------------|------------------------------------------------------|
| Secondary structure  | distinct helix / sheet / coil ribbon profiles        |
| pLDDT                | ribbon thickness — low confidence tapers to nothing  |
| Mutation hotspots    | small displacement bumps at hotspot residues         |
| Bond strain          | angle-violating bonds visibly bend                   |

### Animation
| Property                       | Map                                       |
|--------------------------------|-------------------------------------------|
| MD trajectory / NMR ensemble   | smooth inter-frame interpolation          |
| Normal modes                   | breathing animation with motion blur      |
| Thermal motion (B-factor)      | noise-driven micro-displacement           |

---

## Marquee passes

These are the headline visuals — the things a viewer can do that Mol\*/JSmol cannot. Each is also a roadmap milestone.

### 1. Gaussian electrostatic field
Per-atom partial charge becomes a Gaussian splat in a screen-space-accumulated 3D texture. Volumetric raymarch renders the field as a coloured glow that "bleeds" into the empty space around the protein. The user sees the dipole moment, the catalytic patch, the binding-pocket charge complementarity between protein and ligand — properties that are otherwise only legible by switching mode and squinting at a coloured surface.

### 2. Hydrophobic refractive shell
The marching-cubes surface, but with roughness modulated per-vertex by the nearest-residue hydrophobicity. Hydrophobic patches read as polished glass; hydrophilic regions read as frosted. The hydrophobic core's silhouette glows through the translucent shell.

### 3. Cofactor refractive lens
Each ligand renders inside its own gem-material shell, tuned to a property of the ligand. Heme = ruby. Copper ion = orange topaz. Iron-sulfur cluster = polished steel. Cofactors become structurally and visually the centerpiece — exactly what they are biochemically.

### 4. Confidence cloud
For predicted structures: pLDDT rendered as Perlin-noise-modulated volumetric mist surrounding low-confidence regions. Uncertainty becomes fog — denser where the model is less sure, clear where it's confident. The image carries the model's epistemic state directly.

### 5. H-bond / contact lattice
Hydrogen bonds and non-covalent contacts render as thin glowing beams between donor and acceptor. Bond properties (distance, angle, donor-acceptor identity) modulate beam intensity and hue. The interaction network becomes a luminous lattice through the molecule — a way to see how the protein is held together.

### 6. Breathing modes
For NMR ensembles or MD trajectories: render N frames smoothly interpolated, with motion vectors driving anisotropic blur in the material. The molecule visibly breathes; high-motion regions blur smoothly, rigid cores stay sharp. Conveys protein dynamics as motion, not as separate-frame comparison.

### 7. Pocket void glow
Detected cavities (alpha shapes, Fpocket, MDpocket) render as **inverted** glow — light *coming out of* the void rather than landing on the surface. Druggable pockets become visually irresistible.

---

## Phased delivery

### Phase 0 — Foundations *(largely in place)*
- WebGL2 + Three.js baseline ✓
- PBR `gemMaterial` factory with transmission, IOR, dispersion ✓
- `RefractiveShell` Three.js overlay over Mol\* ✓
- Marching-cubes molecular surface ✓
- Tuning UI scaffold (`GemShellDrawer`) ✓
- mmCIF / PDB parser → atom + bond + residue entity graph (partial — `atomParser.ts` extracts positions today)

### Phase 1 — Discrete primitives *(replace Mol\* for the basic reps)*
- Sphere render pass (instanced; atomic-radius-aware)
- Stick render pass (bond cylinders, order-aware geometry)
- Ribbon / cartoon (Frenet-frame backbone tubing with secondary-structure profile)
- Putty (variable thickness from per-residue scalar)
- Per-entity material slot — material per chain / residue / atom
- Selection system (atom / residue / chain / arbitrary group)

**Exit criteria**: Molero can render any predicted structure end-to-end with no Mol\* dependency, matching the current jewelry-register armature look.

### Phase 2 — Property channels
- Property loader interface: per-atom, per-residue, per-pair value tables (CSV, JSON, embedded in mmCIF B-factor column, computed at load)
- Built-in computed properties: hydrophobicity (Kyte-Doolittle), SASA (Shrake-Rupley), secondary structure (DSSP-lite)
- Channel-mapping UI: assign property → material channel (drag/drop)
- Two flagship maps shipped: hydrophobicity → roughness, pLDDT → emission

**Exit criteria**: a user can load a structure, pick "hydrophobicity → roughness" and "pLDDT → emission" from menus, and see both encoded simultaneously without re-render flicker.

### Phase 3 — Volumetric
- 3D texture infrastructure (float-32 where available, half-float fallback)
- Raymarched volume render pass (back-to-front, alpha-blend)
- Gaussian splat compositor (atom positions + radii → volumetric density)
- DX / CCP4 / MAP grid loaders
- Electrostatic potential pass (Poisson-Boltzmann or APBS output consumed; later, in-browser via WebGPU compute)

**Exit criteria**: load a protein + APBS .dx file → see the electrostatic field as a coloured volumetric glow surrounding the structure.

### Phase 4 — Continuous surfaces
- SES / SAS Connolly probe-rolling surface
- Per-vertex property painting on the surface (hydrophobicity, charge mapped to the surface itself)
- Pocket detection (alpha-shape or grid-based void enumeration)
- Pocket render pass with inverted glow

**Exit criteria**: a binding pocket can be detected automatically and rendered as a glowing void, with the surrounding surface coloured by electrostatic potential.

### Phase 5 — Composite passes
- H-bond detection + beam rendering
- Hydrophobic-contact, salt-bridge, π-stacking detection
- Outline / silhouette pass
- Distance-field-rendered text labels (always-readable, depth-aware)
- Streamline pass for vector fields (gradient of electrostatic potential)

**Exit criteria**: the user can flip on "show me how this protein is held together" and the H-bond + contact lattice renders as luminous beams.

### Phase 6 — Animation / dynamics
- Trajectory loader (XTC, DCD, multi-model PDB, GRO+TRR)
- Inter-frame interpolation with motion vectors
- Motion-vector-driven anisotropy in the material
- Breathing-mode visualization (normal-mode-derived)
- Timeline scrubber UI

**Exit criteria**: load an MD trajectory, hit play, watch the protein breathe — with high-motion regions rendering as motion-blurred, rigid cores rendering crisp.

### Phase 7 — Performance + WebGPU
- Compute-shader Gaussian splatting (10–100× over CPU/fragment)
- WebGPU pipeline for raymarched volumes
- Instanced sphere/stick rendering via compute-driven culling
- LOD system (drop volumetric / surface passes when the molecule is small on-screen)

**Exit criteria**: a 100k-atom assembly (ribosome, capsid, MD snapshot) renders interactively (>30 fps) on consumer hardware.

---

## Data model sketch

```ts
// Spatial entity graph
interface Atom {
  id: number          // global atom index
  element: number     // atomic number
  resId: number       // index into residues[]
  chainId: number     // index into chains[]
  pos: Vec3
  // property channels — keyed by channel name; values are scalars
  // (extended later to vec3 / vec4 for vector fields)
  properties: Map<string, number>
}

interface Residue {
  id: number
  chainId: number
  type: string        // 3-letter code or CCD code
  atomStart: number   // [start, end) into atoms[]
  atomEnd: number
  secondaryStructure: 'helix' | 'sheet' | 'coil' | 'unknown'
  properties: Map<string, number>
}

interface Chain {
  id: number
  asymId: string      // mmCIF label_asym_id (A, B, ...)
  entityType: 'protein' | 'rna' | 'dna' | 'ligand' | 'water'
  residueStart: number
  residueEnd: number
}

interface Bond {
  i: number           // atom index
  j: number           // atom index
  order: 1 | 1.5 | 2 | 3
  type: 'covalent' | 'hbond' | 'metal' | 'disulfide' | 'salt-bridge'
}

interface Scene {
  atoms: Atom[]
  residues: Residue[]
  chains: Chain[]
  bonds: Bond[]
  volumetricFields: VolumetricField[]  // 3D scalar / vector grids
  passes: RenderPass[]                 // active passes
  channelMappings: ChannelMapping[]    // property → material channel routes
}

// A channel mapping: "for entities matching <selector>, map the value of
// <property> via <colormap or curve> into <material channel>".
interface ChannelMapping {
  selector: EntitySelector          // 'all' | 'chain:A' | 'residue.hydrophobic' | ...
  property: string                  // 'hydrophobicity', 'plddt', ...
  channel: MaterialChannel          // 'roughness' | 'emission' | 'transmission' | ...
  transform: ScalarTransform        // linear | sigmoid | clamp | colormap('viridis')
}
```

---

## Non-goals

- **Structure editing.** Molero is a viewer. Building bonds, modeling missing loops, mutating residues — out of scope. Other tools own that.
- **Cheminformatics.** No bond perception from coordinates alone, no automatic chemistry inference. Inputs are well-formed mmCIF / PDB / SDF; bonds come from upstream.
- **Mol\* extension compatibility.** Molero's vocabulary is intentionally different. Migration tools / mmCIF-comment-based mapping can come later if needed.
- **Macroscopic assemblies (Phase 7+).** Capsids, ribosomes, full assemblies at atomic detail — out of scope until WebGPU compute lands. Until then, asset budgets cap at ~100k atoms.

---

## Tech notes

- **WebGPU** is the hard floor (see Design principle 4). All compute paths assume storage buffers and compute shaders; all volumetric paths assume float-32 3D textures. Devices without WebGPU stay on Mol\*.
- **Three.js WebGPURenderer** as the platform — own all custom shaders, treat Three.js as the scene-graph / camera / pipeline layer rather than a renderer that dictates style. (Three.js's WebGPU backend exposes the same `MeshPhysicalMaterial` API we already use, plus TSL for compute pipelines.)
- Materials extend `MeshPhysicalMaterial` via `onBeforeCompile` shader injection where built-in channels aren't enough (e.g. per-vertex hydrophobicity attribute → custom roughness term).
- Per-vertex property packing into custom `BufferAttribute`s; per-fragment interpolation via barycentric. No CPU-side texture writes per frame.
- 3D textures: `RedFormat` / `R32F` where supported (electrostatic potential needs the dynamic range); `R16F` fallback.
- Bond order ≥ 2 rendered as parallel cylinders; aromaticity rendered as dashed ring inside the ring residue.
- Selection picking via instance-id render targets, not raycasting (scales better with atom count).

---

## Naming + identity

**Molero**. Pronunciation: *moh-LEH-roh*.

A foundry for biomolecular vision — beyond inventory rendering, into materials that carry biology in every photon.

---

## Decisions log

- **2026-05-21 — Module location**: stays inside `packages/boltz/` as an internal renderer through Phase 1+. Spins out to a standalone package (target: `packages/molero/`) once the API surface is stable enough that the rest of Corundum is just one consumer.
- **2026-05-21 — WebGPU floor**: WebGPU is a **hard requirement** for Molero. Anyone running Boltz already has the GPU class for it (Apple Silicon, discrete, modern integrated). No WebGL2 fallback inside Molero — devices without WebGPU stay on the Mol\* path until their browser/hardware catches up. This unlocks every render pass to assume compute shaders, storage buffers, and float-32 3D textures from day one.

## Still-open questions

1. **Mol\* coexistence horizon**: how long do we keep Mol\* alive as a fallback? Useful for diff-testing throughout Phases 1–3; can probably retire after Phase 4 once the SES surface lands.

2. **External property formats**: which file formats matter most for Phase 2? mmCIF B-factor-column is the most universal carrier; CSV-per-atom is the most extensible. Probably ship both, prioritise CSV for the channel-mapping demo.
