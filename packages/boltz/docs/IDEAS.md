# Corundum — Forward Ideas

*Things worth doing later, with enough scaffolding that future-you can
pick them up without re-deriving why.*

---

## Streaming prediction visualizer ("wire forming → heating → enclosing")

Logged 2026-05-17, during the vacuum-mode session.

### Premise

The 1–5 minute prediction wait is currently dead time behind a progress
bar. The vacuum-mode rendering (glowing wire inside thin gaussian-
surface shell, see `MolViewer.tsx`) is already the visual vocabulary;
extending it into the wait turns the loading state INTO the science.

Three phases map cleanly to the orchestrator's existing stages:

1. **Wire forming** (~50 diffusion steps). Per-step Cα coordinates push
   into a thick polyline. The line whips from noise to ordered
   structure, frame-by-frame, while the diffusion sampler converges.
   Glow uniform-cold throughout — pLDDT not yet known.
2. **Heating** (confidence head, 1 pass). Per-residue emissive on the
   polyline ramps from uniform to pLDDT-modulated brightness. ~1s
   transition. This is the science beat: the model now knows what it
   knows.
3. **Enclosing** (gaussian-surface compute, ~1s on small structures).
   Shell fades in around the wire, `alpha 0 → 0.15`. Cross-fade to the
   real Mol\* vacuum mode at the end, where full interactivity (rotate,
   select, hover) resumes.

### Architecture sketch

- **Renderer choice**: a dedicated Three.js (or raw WebGPU) canvas
  layered with the Mol\* canvas inside the viewer card. Mol\* is not
  built for 50 frames/s coordinate swaps; a custom canvas drawing a
  single thick polyline through Cα positions takes Float32Array
  directly and runs at native frame rate. Hand off to Mol\* vacuum mode
  only at the end. Both canvases stack in the same card so the cross-
  fade is free.
- **Data flow**: extend `ProgressEvent` in `orchestrate.ts` to carry a
  `phase` discriminant and, for diffusion-step events, the current Cα
  coords. Send them across the worker boundary as transferable
  `Float32Array`s — zero copy, no JSON serialization. The Cα subset for
  a typical 300-residue protein is `300 * 3 * 4 = 3.6 kB` per step ×
  50 steps = 180 kB total, well within budget.
- **State**: a `predictionPhase: 'idle' | 'wire-forming' | 'heating' |
  'enclosing' | 'done'` field in the boltz store. The viewer chooses
  which renderer is active by reading it.

### Scope ladder

- **Tier 1 — data flow only** (~half a day). Wire up the worker→UI
  channel, verify per-step Cα coords arrive at 50 steps/30s without
  stalling the diffusion loop. Render nothing streaming yet. Proves
  the architecture before sinking renderer time.
- **Tier 2 — wire + heating** (~2–3 days). Custom canvas draws the
  polyline, updates per step. Heating phase modulates emissive by
  pLDDT once confidence head fires. No shell yet; Mol\* vacuum mode
  picks up after the heating moment.
- **Tier 3 — full envelope** (~4–5 days total). Wire + heating + shell
  fade-in + cross-fade hand-off to Mol\* vacuum. The version that
  ships.

### Notes for future-you

- The diffusion sampler in `orchestrate.ts` already iterates per step
  with a callback hook (used for the progress bar). Adding Cα-coord
  emission to that callback is small — the math to extract Cα from the
  packed atom tensor lives in `mmcif.ts:writeMmcif` (search for
  `token_to_center_atom`), reuse it.
- `unit.conformation` updates on the Mol\* side were considered and
  rejected (Tier B above). Mol\* re-derives geometry per change and the
  re-tessellation cost dominates. Custom canvas is the right scope.
- Bioart polish (Tier 4, deferred): slow camera tumble during wire
  forming; a brief particle/spark pulse at the heating moment; subtle
  bloom kick when the shell closes. None required for the core idea.

### Why not now

Vacuum mode is the visual vocabulary the loader will speak. Building
the loader before the vocabulary feels confident is premature. The
streaming work also needs (a) the orchestration loop on a real
predicted structure end-to-end in the browser, which is in flight, and
(b) the per-step callback hook to be stable, which means letting the
current orchestrate.ts settle past v0.1.
