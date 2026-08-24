/** Signal Observatory: asymmetric observation desk, ink field, Measured Teal computation signals, and Space Grotesk/IBM Plex Mono hierarchy. */
import { useEffect, useMemo, useState } from "react";
import { DecisionField } from "./components/DecisionField";
import { GraphCanvas } from "./components/GraphCanvas";
import { LossTrace } from "./components/LossTrace";
import { type DatasetId, TrainingSession, type TrainingSnapshot } from "./engine/training";

const DATASETS: { id: DatasetId; title: string; description: string }[] = [
  { id: "xor", title: "XOR", description: "A minimal nonlinear boundary." },
  { id: "spiral", title: "Spiral", description: "Two classes braided by radius." },
  { id: "sine", title: "Sine", description: "A smooth function fitted from samples." },
];

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

function createSession(dataset: DatasetId, width: number): TrainingSession {
  return new TrainingSession(dataset, width, 2026);
}

export default function App() {
  const [datasetId, setDatasetId] = useState<DatasetId>("xor");
  const [width, setWidth] = useState(8);
  const [session, setSession] = useState(() => createSession("xor", 8));
  const [snapshot, setSnapshot] = useState<TrainingSnapshot>(() => session.snapshot());
  const [history, setHistory] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const reducedMotion = useReducedMotion();

  const dataset = session.dataset;
  const description = useMemo(() => DATASETS.find((item) => item.id === datasetId)?.description ?? "", [datasetId]);

  useEffect(() => {
    const next = createSession(datasetId, width);
    setSession(next);
    setSnapshot(next.snapshot());
    setHistory([]);
    setIsTraining(false);
    setActiveStep(-1);
  }, [datasetId, width]);

  useEffect(() => {
    if (!isTraining) return;
    let completed = 0;
    const run = () => {
      const next = session.train(8);
      setSnapshot(next);
      setHistory([...session.history]);
      setActiveStep((step) => (step + 1) % 7);
      completed += 8;
      if (completed < 320) window.setTimeout(run, 22);
      else setIsTraining(false);
    };
    const timer = window.setTimeout(run, 0);
    return () => window.clearTimeout(timer);
  }, [isTraining, session]);

  const step = () => {
    const next = session.trainOne();
    setSnapshot(next);
    setHistory([...session.history]);
    setActiveStep((value) => (value + 1) % 7);
  };

  const reset = () => {
    const next = createSession(datasetId, width);
    setSession(next);
    setSnapshot(next.snapshot());
    setHistory([]);
    setIsTraining(false);
    setActiveStep(-1);
  };

  return (
    <main className="app-shell">
      <aside className="operations-rail" aria-label="Training controls">
        <a className="brand" href="#top" aria-label="NeuroForge home">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span>NEURO<span>FORGE</span></span>
        </a>
        <div className="rail-intro">
          <p className="eyebrow">autograd observatory</p>
          <h1>Trace the gradient,<br />not the mystery.</h1>
          <p>Build intuition by watching values, local derivatives, and loss change inside a real reverse-mode engine.</p>
        </div>

        <fieldset className="control-group">
          <legend>dataset</legend>
          <div className="dataset-list">
            {DATASETS.map((item) => <button key={item.id} className={datasetId === item.id ? "dataset-option selected" : "dataset-option"} onClick={() => setDatasetId(item.id)} disabled={isTraining}>
              <strong>{item.title}</strong><span>{item.description}</span>
            </button>)}
          </div>
        </fieldset>

        <fieldset className="control-group">
          <legend>hidden width <output>{width}</output></legend>
          <input aria-label="Hidden layer width" type="range" min="4" max="16" step="2" value={width} onChange={(event) => setWidth(Number(event.target.value))} disabled={isTraining} />
          <div className="range-notes"><span>4 units</span><span>16 units</span></div>
        </fieldset>

        <div className="action-stack">
          <button className="primary-action" onClick={() => setIsTraining(true)} disabled={isTraining}>{isTraining ? "Tracing parameters…" : "Observe 320 epochs"}</button>
          <button className="secondary-action" onClick={step} disabled={isTraining}>Trace next backward pass</button>
          <button className="text-action" onClick={reset} disabled={isTraining}>Reset observed state</button>
        </div>
        <p className="rail-footnote">{description} · runs entirely in your browser</p>
      </aside>

      <section className="observation-desk" id="top">
        <header className="desk-header">
          <div><p className="eyebrow">runtime status</p><p className="status-line"><i className={isTraining ? "status-dot is-running" : "status-dot"} /> {isTraining ? "optimizing parameters" : "ready for observation"}</p></div>
          <div className="method-note"><span>reverse mode</span><span>•</span><span>pure TypeScript</span><span>•</span><span>no ML runtime</span></div>
        </header>
        <GraphCanvas snapshot={snapshot} activeStep={activeStep} reducedMotion={reducedMotion} />
        <div className="lower-observations">
          <LossTrace history={history} />
          <DecisionField dataset={dataset} predict={(points) => session.predict(points)} />
        </div>
        <section className="derivative-note" aria-label="Current backward rule">
          <p className="eyebrow">current local rule</p>
          <div><span className="formula">z = a · b</span><span>→</span><span className="formula accent">∂L/∂a = ∂L/∂z · b</span></div>
          <p>Each operation keeps its parents and local derivative. A topological reverse pass accumulates the chain rule at every leaf.</p>
        </section>
      </section>
    </main>
  );
}
