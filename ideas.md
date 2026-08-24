# NeuroForge Design Directions

## Chosen direction: Signal Observatory

**Modern scientific instrumentation** with editorial data-visualization influences. The interface is designed as an asymmetric observation desk: a narrow operations rail, a central computation canvas, and a lower temporal loss trace. Its visual system uses ink-black fields, Measured Teal (`#31D3C6`) for active computation, and restrained amber for loss or caution. Space Grotesk provides interface hierarchy, while IBM Plex Mono makes numeric states, formulas, and parameters legible.

The design prioritizes traceable computation, purposeful hierarchy, quiet precision, and learning through motion. Its signature motifs are fine crosshair grid texture, signal nodes carrying values and gradients, directional edge pulses during backpropagation, and an amber loss trace. All nonessential movement respects reduced-motion preferences.

**Brand essence:** NeuroForge makes neural-network mechanics observable for builders who want to understand the math, not just run it.

## Style Decisions

- The computation graph is the primary hero object: node values, edge direction, gradient state, and loss consequence visually dominate the first viewport.
- The forward/reverse node-and-arc geometry recurs in signal spines, edge chevrons, state markers, and graph headings as NeuroForge’s visual fingerprint.
- Operational copy names the observable action or evidence state, such as “Trace next backward pass,” rather than generic UI verbs.
