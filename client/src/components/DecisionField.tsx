/** Signal Observatory: an SVG classification field that makes model predictions inspectable without external chart dependencies. */
import type { Dataset } from "../engine/training";

type DecisionFieldProps = { dataset: Dataset; predict: (points: number[][]) => number[] };

export function DecisionField({ dataset, predict }: DecisionFieldProps) {
  if (dataset.task === "regression") return <RegressionPreview dataset={dataset} predict={predict} />;
  const resolution = 16;
  const points = Array.from({ length: resolution * resolution }, (_, index) => {
    const x = -1.1 + (2.2 * (index % resolution)) / (resolution - 1);
    const y = -1.1 + (2.2 * Math.floor(index / resolution)) / (resolution - 1);
    return [x, y];
  });
  const probabilities = predict(points);
  const size = 260;
  const cell = size / resolution;
  return (
    <section className="field-panel" aria-label="Model prediction field">
      <header className="panel-heading compact-heading"><div><p className="eyebrow">model readout</p><h2>Decision field</h2></div></header>
      <svg className="field-svg" viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Current classification decision field">
        {points.map((_point, index) => {
          const probability = probabilities[index * 2 + 1] ?? 0.5;
          return <rect key={index} x={(index % resolution) * cell} y={Math.floor(index / resolution) * cell} width={cell + 0.2} height={cell + 0.2} fill={`rgba(49, 211, 198, ${0.08 + probability * 0.56})`} />;
        })}
        {dataset.inputs.map(([x, y], index) => <circle key={index} cx={((x + 1.1) / 2.2) * size} cy={size - ((y + 1.1) / 2.2) * size} r="4.5" className={dataset.labels[index] ? "class-one" : "class-zero"} />)}
      </svg>
    </section>
  );
}

function RegressionPreview({ dataset, predict }: DecisionFieldProps) {
  const sample = Array.from({ length: 62 }, (_, index) => -1 + (2 * index) / 61).map((value) => [value]);
  const predicted = predict(sample);
  const toPoint = (x: number, y: number) => `${(x + 1) * 130},${130 - y * 100}`;
  return (
    <section className="field-panel" aria-label="Regression preview">
      <header className="panel-heading compact-heading"><div><p className="eyebrow">model readout</p><h2>Regression fit</h2></div></header>
      <svg className="field-svg" viewBox="0 0 260 260" role="img" aria-label="Predicted sine regression curve">
        <line x1="0" x2="260" y1="130" y2="130" className="axis-line" /><line x1="130" x2="130" y1="0" y2="260" className="axis-line" />
        <polyline points={sample.map(([x], index) => toPoint(x, predicted[index])).join(" ")} className="prediction-line" />
        {dataset.inputs.map(([x], index) => <circle key={index} cx={(x + 1) * 130} cy={130 - dataset.targets[index][0] * 100} r="2.2" className="class-one" />)}
      </svg>
    </section>
  );
}

