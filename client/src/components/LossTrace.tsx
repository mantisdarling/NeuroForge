/** Signal Observatory: restrained amber temporal trace, designed for comparison rather than decoration. */
type LossTraceProps = { history: number[] };

export function LossTrace({ history }: LossTraceProps) {
  const series = history.length ? history : [0.9, 0.74, 0.62, 0.54, 0.47, 0.42, 0.36, 0.31, 0.27];
  const width = 720;
  const height = 178;
  const padding = { top: 14, right: 12, bottom: 28, left: 40 };
  const max = Math.max(...series) * 1.1;
  const min = Math.min(...series) * 0.9;
  const x = (index: number) => padding.left + (index / Math.max(series.length - 1, 1)) * (width - padding.left - padding.right);
  const y = (value: number) => padding.top + (1 - (value - min) / Math.max(max - min, 1e-9)) * (height - padding.top - padding.bottom);
  const points = series.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  return (
    <section className="loss-panel" aria-label="Loss over epochs">
      <header className="panel-heading compact-heading">
        <div>
          <p className="eyebrow">temporal trace</p>
          <h2>Training loss</h2>
        </div>
        <span className="trace-stat">{series.at(-1)?.toFixed(5)}</span>
      </header>
      <svg className="loss-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Loss curve decreasing over training">
        {[0.25, 0.5, 0.75].map((ratio) => <line key={ratio} x1={padding.left} x2={width - padding.right} y1={padding.top + ratio * (height - padding.top - padding.bottom)} y2={padding.top + ratio * (height - padding.top - padding.bottom)} className="loss-guide" />)}
        <polyline points={points} className="loss-line" />
        <circle cx={x(series.length - 1)} cy={y(series.at(-1) ?? 0)} r="4" className="loss-point" />
        <text className="axis-label" x={padding.left} y={height - 8}>epoch 0</text>
        <text className="axis-label" x={width - padding.right} y={height - 8} textAnchor="end">epoch {Math.max(series.length - 1, 0)}</text>
      </svg>
    </section>
  );
}

