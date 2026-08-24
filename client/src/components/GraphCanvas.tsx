/** Signal Observatory: central computation canvas with traceable values and restrained teal backpropagation pulses. */
import type { TrainingSnapshot } from "../engine/training";

type GraphCanvasProps = {
  snapshot: TrainingSnapshot;
  activeStep: number;
  reducedMotion: boolean;
};

const nodes = [
  { id: "x", title: "input x", x: 38, y: 82, formula: "x", value: "[x₁, x₂]" },
  { id: "w1", title: "linear 1", x: 222, y: 82, formula: "xW + b", value: "8 units" },
  { id: "a1", title: "tanh", x: 406, y: 82, formula: "tanh(z)", value: "activation" },
  { id: "w2", title: "linear 2", x: 590, y: 82, formula: "aW + b", value: "8 units" },
  { id: "a2", title: "tanh", x: 774, y: 82, formula: "tanh(z)", value: "activation" },
  { id: "out", title: "output", x: 958, y: 82, formula: "logits", value: "ŷ" },
  { id: "loss", title: "loss", x: 1112, y: 82, formula: "L(ŷ, y)", value: "objective" },
];

const edges = nodes.slice(0, -1).map((node, index) => ({ from: node, to: nodes[index + 1] }));

export function GraphCanvas({ snapshot, activeStep, reducedMotion }: GraphCanvasProps) {
  const progress = activeStep >= 0 ? nodes.length - 1 - (activeStep % nodes.length) : -1;
  return (
    <section className="graph-panel" aria-label="Live computation graph">
      <header className="panel-heading">
        <div>
          <p className="eyebrow"><span className="mini-flow" aria-hidden="true" /> live structure</p>
          <h2>Computation graph</h2>
        </div>
        <div className="graph-legend" aria-label="Graph legend">
          <span><i className="legend-node" /> forward value</span>
          <span><i className="legend-pulse" /> reverse gradient</span>
        </div>
      </header>
      <div className="graph-wrap">
        <svg className="graph-svg" viewBox="0 0 1236 300" role="img" aria-label="Neural network operations linked from input through loss">
          <defs>
            <pattern id="crosshatch" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M 24 0 L 0 0 0 24" fill="none" stroke="rgba(170, 198, 200, 0.09)" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width="1236" height="300" fill="url(#crosshatch)" rx="12" />
          <path className="graph-spine" d="M 36 46 H 1198" />
          <path className="graph-spine-return" d="M 1198 238 H 36" />
          {edges.map(({ from, to }, index) => {
            const pulsing = progress === index || progress === index + 1;
            return (
              <g key={`${from.id}-${to.id}`}>
                <path className="graph-edge" d={`M ${from.x + 120} ${from.y + 38} H ${to.x - 16}`} />
                <path className="edge-chevron" d={`M ${to.x - 26} ${to.y + 32} l 10 6 l -10 6`} />
                {pulsing && !reducedMotion && <path className="graph-edge-pulse" d={`M ${to.x - 16} ${to.y + 38} H ${from.x + 120}`} />}
              </g>
            );
          })}
          {nodes.map((node, index) => {
            const active = progress === index;
            const isLoss = node.id === "loss";
            const displayValue = isLoss ? snapshot.loss.toFixed(4) : node.id === "out" ? snapshot.output.slice(0, 2).map((value) => value.toFixed(2)).join(" · ") : node.value;
            return (
              <g key={node.id} className={active ? "active-graph-node" : ""} transform={`translate(${node.x} ${node.y})`}>
                <rect className={`graph-node ${isLoss ? "loss-node" : ""}`} width="120" height="76" rx="8" />
                <text className="graph-node-title" x="13" y="23">{node.title}</text>
                <text className="graph-node-formula" x="13" y="45">{node.formula}</text>
                <text className="graph-node-value" x="13" y="64">{displayValue}</text>
                {active && <circle className="node-signal" cx="111" cy="11" r="3.5" />}
              </g>
            );
          })}
          <text className="backward-label" x="38" y="262">∂L / ∂θ  ←  gradient signal returns through local derivative rules</text>
        </svg>
        <p className="graph-scroll-hint">Swipe the canvas to trace every operation <span aria-hidden="true">→</span></p>
      </div>
      <footer className="graph-readout">
        <span>epoch <b>{snapshot.epoch}</b></span>
        <span>loss <b>{snapshot.loss.toFixed(5)}</b></span>
        <span>∥∇θ∥ <b>{snapshot.gradientMagnitude.toFixed(5)}</b></span>
        {snapshot.accuracy !== undefined && <span>accuracy <b>{(snapshot.accuracy * 100).toFixed(0)}%</b></span>}
      </footer>
    </section>
  );
}
