/** Signal Observatory hero: a proprietary forward/reverse signal map that establishes NeuroForge as a large learning instrument, not a dashboard. */
export function HeroSignalMap() {
  const nodes = [
    { x: 90, y: 128, label: "INPUT", value: "x" },
    { x: 252, y: 204, label: "AFFINE", value: "W₁x + b₁" },
    { x: 425, y: 130, label: "ACTIVATE", value: "tanh" },
    { x: 575, y: 222, label: "AFFINE", value: "W₂a + b₂" },
    { x: 700, y: 136, label: "LOSS", value: "L" },
  ];
  return (
    <div className="hero-signal-map" aria-label="A stylized forward and reverse automatic differentiation signal map">
      <div className="signal-map-topline"><span><i /> LIVE DERIVATIVE FIELD</span><span>TRACE 01 / 01</span></div>
      <svg viewBox="0 0 790 350" role="img" aria-hidden="true">
        <defs>
          <pattern id="hero-grid" width="34" height="34" patternUnits="userSpaceOnUse"><path d="M34 0H0V34" fill="none" stroke="rgba(180,222,217,.09)" /></pattern>
          <filter id="soft-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>
        <rect width="790" height="350" fill="url(#hero-grid)" />
        <path className="hero-contour contour-one" d="M-20 242 C 120 192, 148 294, 295 235 S 494 152, 810 202" />
        <path className="hero-contour contour-two" d="M-10 78 C 145 136, 190 54, 350 105 S 584 280, 805 244" />
        <path className="hero-forward-route" d="M90 128 C155 128 174 204 252 204 S359 130 425 130 S511 222 575 222 S650 136 700 136" />
        <path className="hero-reverse-route" d="M700 162 C650 162 631 248 575 248 S491 156 425 156 S333 230 252 230 S155 154 90 154" />
        {nodes.map((node, index) => <g key={node.label} transform={`translate(${node.x} ${node.y})`} className={index === 4 ? "hero-node hero-node-loss" : "hero-node"}>
          <circle r="23" />
          <circle r="6" />
          <text x="0" y="-38" textAnchor="middle" className="hero-node-label">{node.label}</text>
          <text x="0" y="48" textAnchor="middle" className="hero-node-value">{node.value}</text>
        </g>)}
        <g className="hero-packet" filter="url(#soft-glow)"><circle cx="481" cy="178" r="5" /><circle cx="363" cy="168" r="3.4" /><circle cx="176" cy="166" r="2.8" /></g>
        <text x="40" y="320" className="hero-axis-copy">FORWARD EVALUATION  →</text>
        <text x="750" y="320" textAnchor="end" className="hero-axis-copy reverse">←  REVERSE ACCUMULATION</text>
      </svg>
      <div className="signal-map-bottomline"><span>LOCAL RULES: 15</span><span className="map-teal">∂L / ∂θ ACCUMULATING</span></div>
    </div>
  );
}
