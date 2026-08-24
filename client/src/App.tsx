/** Signal Observatory: a large-scale, asymmetric learning instrument with an expansive hero, a full live lab, and a technical story from trace to proof. */
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import { AmbientField } from "./components/AmbientField";
import { DecisionField } from "./components/DecisionField";
import { GuidedTrack } from "./components/GuidedTrack";
import { GraphCanvas } from "./components/GraphCanvas";
import { HeroSignalMap } from "./components/HeroSignalMap";
import { LossTrace } from "./components/LossTrace";
import { type DatasetId, TrainingSession, type TrainingSnapshot } from "./engine/training";

const AdvancedLearningLab = lazy(async () => ({ default: (await import("./components/AdvancedLearningLab")).AdvancedLearningLab }));

const DATASETS: { id: DatasetId; title: string; description: string; marker: string }[] = [
  { id: "xor", title: "XOR", description: "A minimal nonlinear boundary.", marker: "01" },
  { id: "spiral", title: "Spiral", description: "Two classes braided by radius.", marker: "02" },
  { id: "sine", title: "Sine", description: "A smooth function fitted from samples.", marker: "03" },
];

const ENGINE_NOTES = [
  ["TAPE", "Parent references + local rules", "The graph stays explicit, so every value has an inspectable origin."],
  ["ORDER", "Topological reverse traversal", "A scalar loss seeds the pass; shared parents accumulate all valid paths."],
  ["PROOF", "Numerical + framework parity", "Finite differences test local rules; a fixed float64 benchmark compares the complete pass with PyTorch."],
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

function createSession(dataset: DatasetId, width: number, batchSize = 4): TrainingSession {
  return new TrainingSession(dataset, width, 2026, { batchSize });
}

function scrollToId(id: string): void {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function initialHighContrast(): boolean {
  try {
    return window.localStorage.getItem("neuroforge-high-contrast") === "true";
  } catch {
    return false;
  }
}

export default function App() {
  const [datasetId, setDatasetId] = useState<DatasetId>("xor");
  const [width, setWidth] = useState(8);
  const [batchSize, setBatchSize] = useState(4);
  const [session, setSession] = useState(() => createSession("xor", 8, 4));
  const [snapshot, setSnapshot] = useState<TrainingSnapshot>(() => session.snapshot());
  const [history, setHistory] = useState<number[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [guideStep, setGuideStep] = useState<number | null>(0);
  const [highContrast, setHighContrast] = useState(initialHighContrast);
  const [advancedReady, setAdvancedReady] = useState(false);
  const advancedBoundaryRef = useRef<HTMLElement>(null);
  const reducedMotion = useReducedMotion();

  const dataset = session.dataset;
  const description = useMemo(() => DATASETS.find((item) => item.id === datasetId)?.description ?? "", [datasetId]);

  useEffect(() => {
    const next = createSession(datasetId, width, batchSize);
    setSession(next);
    setSnapshot(next.snapshot());
    setHistory([]);
    setIsTraining(false);
    setActiveStep(-1);
  }, [datasetId, width, batchSize]);

  useEffect(() => {
    try {
      window.localStorage.setItem("neuroforge-high-contrast", String(highContrast));
    } catch {
      // The visual preference remains optional when browser storage is unavailable.
    }
  }, [highContrast]);

  useEffect(() => {
    const boundary = advancedBoundaryRef.current;
    if (!boundary) return;
    const reveal = () => setAdvancedReady(true);
    if (window.location.hash === "#studies" || !("IntersectionObserver" in window)) {
      reveal();
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        reveal();
        observer.disconnect();
      }
    }, { rootMargin: "900px 0px" });
    observer.observe(boundary);
    return () => observer.disconnect();
  }, []);

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
    setGuideStep((value) => value === 2 ? null : value);
  };

  const reset = () => {
    const next = createSession(datasetId, width, batchSize);
    setSession(next);
    setSnapshot(next.snapshot());
    setHistory([]);
    setIsTraining(false);
    setActiveStep(-1);
    setGuideStep(0);
  };

  const advanceGuide = () => setGuideStep((value) => value === null || value >= 2 ? null : value + 1);

  return (
    <main className={highContrast ? "site-shell high-contrast" : "site-shell"}>
      <AmbientField />
      <header className="global-nav">
        <a className="brand" href="#top" aria-label="NeuroForge home"><span className="brand-mark" aria-hidden="true"><span /><span /><span /></span><span className="brand-lockup"><span>NEURO<span>FORGE</span></span><small>OBSERVATORY</small></span></a>
        <nav aria-label="Primary navigation"><a href="#lab">Live lab</a><a href="#studies">Studies</a><a href="#method">Method</a><a href="#engine">Engine</a></nav>
        <div className="nav-tools"><button className="contrast-toggle" type="button" aria-pressed={highContrast} aria-label={highContrast ? "Disable high contrast" : "Enable high contrast"} onClick={() => setHighContrast((value) => !value)}><span className="contrast-glyph" aria-hidden="true">◐</span><span className="contrast-label">High contrast</span></button><button className="nav-action" onClick={() => scrollToId("lab")}><span aria-hidden="true">↘</span> Inspect live lab</button></div>
      </header>

      <section className="hero-field" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span className="mini-flow" aria-hidden="true" /> A learning instrument for reverse mode</p>
          <h1>Make every<br /><em>derivative</em> visible.</h1>
          <p className="hero-lede">NeuroForge is not another black box. It is a live field guide to the values, local rules, and gradient paths that make a neural network learn.</p>
          <div className="hero-actions"><button className="hero-primary" onClick={() => scrollToId("lab")}>Observe live training <span>↘</span></button><button className="hero-secondary" onClick={() => scrollToId("method")}>Inspect reverse method</button></div>
          <dl className="hero-proof"><div><dt>19</dt><dd>differentiable operations</dd></div><div><dt>5</dt><dd>extended local studies</dd></div><div><dt>0</dt><dd>remote model calls</dd></div></dl>
        </div>
        <HeroSignalMap />
        <div className="hero-index"><span>01</span><i /><span>Observe the consequence of every local rule.</span></div>
      </section>

      <section className="principle-band" aria-label="What NeuroForge makes observable">
        <p className="eyebrow">What the instrument exposes</p>
        <div className="principle-grid"><p><b>Values</b> move forward</p><p><b>Derivatives</b> return backward</p><p><b>Evidence</b> stays inspectable</p></div>
      </section>

      <section className="lab-section" id="lab">
        <header className="section-intro lab-intro">
          <div><p className="eyebrow">02 / operational field</p><h2>Open the training<br />process <em>wide.</em></h2></div>
          <p>This is the live instrument. Select a study, alter the hidden width, and watch a genuine TypeScript reverse pass shape predictions in the browser.</p>
        </header>
        <div className="lab-guide-wrap">
          <GuidedTrack step={guideStep} onAdvance={advanceGuide} onDismiss={() => setGuideStep(null)} onRestart={() => setGuideStep(0)} />
        </div>
        <div className={guideStep === 0 ? "lab-frame guide-controls-active" : guideStep === 1 ? "lab-frame guide-run-active" : guideStep === 2 ? "lab-frame guide-graph-active" : "lab-frame"}>
          <aside className="operations-rail" aria-label="Training controls">
            <div className="rail-heading"><span className="rail-index">LAB 02</span><p>Session controls</p></div>
            <fieldset className="control-group"><legend>Choose a learning study</legend><div className="dataset-list">{DATASETS.map((item) => <button key={item.id} className={datasetId === item.id ? "dataset-option selected" : "dataset-option"} onClick={() => { setDatasetId(item.id); if (guideStep === 0) setGuideStep(1); }} disabled={isTraining}><span className="dataset-marker">{item.marker}</span><span><strong>{item.title}</strong><small>{item.description}</small></span></button>)}</div></fieldset>
            <fieldset className="control-group"><legend>Hidden width <output>{width}</output></legend><input aria-label="Hidden layer width" type="range" min="4" max="16" step="2" value={width} onChange={(event) => setWidth(Number(event.target.value))} disabled={isTraining} /><div className="range-notes"><span>4 units</span><span>16 units</span></div></fieldset>
            <div className="action-stack"><button className={isTraining ? "primary-action is-loading" : "primary-action"} onClick={() => { setIsTraining(true); if (guideStep === 1) setGuideStep(2); }} disabled={isTraining}>{isTraining ? "Tracing parameters…" : "Observe 320 epochs"}</button><button className="secondary-action" onClick={step} disabled={isTraining}>Trace next backward pass</button><button className="text-action" onClick={reset} disabled={isTraining}>Reset observed state</button></div>
            <p className="rail-footnote"><i /> {description}<br />Runs entirely in your browser.</p>
          </aside>

          <section className="observation-desk" aria-label="Live training observatory">
            <header className="desk-header"><div><p className="eyebrow">Runtime status</p><p className="status-line"><i className={isTraining ? "status-dot is-running" : "status-dot"} /> {isTraining ? "optimizing parameters" : "ready for observation"}</p></div><div className="method-note"><span>reverse mode</span><span>•</span><span>pure TypeScript</span><span>•</span><span>no ML runtime</span></div></header>
            <GraphCanvas snapshot={snapshot} activeStep={activeStep} reducedMotion={reducedMotion} />
            <div className="lower-observations"><LossTrace history={history} /><DecisionField dataset={dataset} predict={(points) => session.predict(points)} /></div>
            <section className="derivative-note" aria-label="Current backward rule"><p className="eyebrow">Current local rule</p><div><span className="formula">z = a · b</span><span>→</span><span className="formula accent">∂L/∂a = ∂L/∂z · b</span></div><p>Each operation keeps its parents and local derivative. A topological reverse pass accumulates the chain rule at every leaf.</p></section>
          </section>
        </div>
      </section>

      <section ref={advancedBoundaryRef} className="advanced-boundary" id="studies" aria-label="Advanced studies" aria-busy={!advancedReady}>
        {advancedReady && <Suspense fallback={<div className="advanced-section" aria-label="Loading advanced studies"><p className="eyebrow">Preparing bounded local studies…</p></div>}><AdvancedLearningLab session={session} onBatchSize={setBatchSize} /></Suspense>}
      </section>

      <section className="method-section" id="method">
        <header className="section-intro"><div><p className="eyebrow">03 / from forward trace to proof</p><h2>Not a metaphor.<br /><em>A method.</em></h2></div><p>Autograd becomes useful when you can see the distinction between evaluating a graph and differentiating it. NeuroForge keeps both passes in view.</p></header>
        <div className="method-path"><article className="method-equation"><span>FORWARD PASS</span><strong>x → z → a → ŷ → L</strong><p>Evaluate each operation and remember the local context needed to differentiate it.</p></article><div className="method-vector"><span>01</span><i /><span>02</span><i /><span>03</span></div><article className="method-equation reverse-equation"><span>REVERSE PASS</span><strong>∂L/∂x ← ∂L/∂z ← 1</strong><p>Start at the scalar loss, follow the graph backward, and add every valid gradient contribution.</p></article></div>
        <div className="method-cards">{ENGINE_NOTES.map(([tag, title, text], index) => <article key={tag}><span>{`0${index + 1}`} / {tag}</span><h3>{title}</h3><p>{text}</p></article>)}</div>
      </section>

      <section className="engine-section" id="engine">
        <div className="engine-wash" aria-hidden="true" />
        <header className="section-intro engine-intro"><div><p className="eyebrow">04 / the engine room</p><h2>Small enough to<br /><em>inspect.</em> Real enough to learn from.</h2></div><p>NeuroForge is intentionally compact. The point is not to approximate a production framework; it is to make the essential mechanics legible, testable, and alive.</p></header>
        <div className="spec-matrix"><div className="spec-row spec-heading"><span>Surface</span><span>What is actually implemented</span><span>Why it matters</span></div><div className="spec-row"><span>Tensor tape</span><span>Data, gradients, shape, parents, backward rules, Conv2D, and max pooling</span><span>Every graph edge is traceable, including receptive-field accumulation.</span></div><div className="spec-row"><span>Learning stack</span><span>Dense + convolutional layers, activations, losses, sequential model, SGD, momentum, and Adam</span><span>Enough structure to make real learning behavior comparable.</span></div><div className="spec-row"><span>Verification</span><span>Finite differences, convergence studies, float64 PyTorch parity, type checks, and dependency audit</span><span>Math claims are backed by executable checks and an independent comparison.</span></div></div>
      </section>

      <section className="closing-field"><p className="eyebrow">Next observation</p><h2>Every black box<br />starts as a graph.</h2><p>Open a study, trace one backward pass, then change the system and watch what the gradient does.</p><button className="hero-primary" onClick={() => scrollToId("lab")}>Return to live lab <span>↗</span></button></section>
      <footer className="site-footer"><a className="brand" href="#top"><span className="brand-mark" aria-hidden="true"><span /><span /><span /></span><span>NEURO<span>FORGE</span></span></a><p>Reverse-mode automatic differentiation, made observable.</p><span>TYPE SCRIPT / LOCAL ONLY / 2026</span></footer>
    </main>
  );
}
