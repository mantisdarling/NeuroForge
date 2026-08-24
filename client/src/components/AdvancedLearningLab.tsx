/** Signal Observatory advanced studies: bounded, local, inspectable experiments built on the custom autograd engine. */
import { useEffect, useMemo, useRef, useState } from "react";
import "../advanced.css";
import "../parity.css";
import { DigitLabSession, ShapeCnnSession, compareOptimizers, type OptimizerTrace, type StudySnapshot } from "../engine/advanced";
import { PYTORCH_PARITY } from "../engine/pytorchParity";
import type { TrainingSession } from "../engine/training";

type AdvancedLearningLabProps = { session: TrainingSession; onBatchSize: (size: number) => void; };

const fmt = (value: number) => value.toFixed(value < 0.01 ? 4 : 3);

export function AdvancedLearningLab({ session, onBatchSize }: AdvancedLearningLabProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [digitSession] = useState(() => new DigitLabSession());
  const [digitStudy, setDigitStudy] = useState<StudySnapshot>({ epoch: 0, loss: 2.303, accuracy: 0, history: [] });
  const [digitScores, setDigitScores] = useState<number[]>(Array(10).fill(0.1));
  const [isDigitTraining, setIsDigitTraining] = useState(false);
  const [perturbation, setPerturbation] = useState(0);
  const [batchSize, setBatchSize] = useState(session.batchSize);
  const [traces, setTraces] = useState<OptimizerTrace[]>([]);
  const [isCnnTraining, setIsCnnTraining] = useState(false);
  const [isComparingOptimizers, setIsComparingOptimizers] = useState(false);
  const [cnn] = useState(() => new ShapeCnnSession());
  const [cnnStudy, setCnnStudy] = useState<StudySnapshot>(() => cnn.snapshot());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.fillStyle = "#061113";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  const sensitivity = useMemo(() => session.previewPerturbation(0, perturbation), [session, perturbation]);
  const gradientBars = session.lastBatchGradient.length ? session.lastBatchGradient : Array.from({ length: Math.min(batchSize, 4) }, (_value, index) => (index + 1) * session.snapshot().gradientMagnitude / 4);

  const paint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context || !drawing.current) return;
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * canvas.width / bounds.width;
    const y = (event.clientY - bounds.top) * canvas.height / bounds.height;
    context.fillStyle = "#eefaf7";
    context.beginPath();
    context.arc(x, y, 12, 0, Math.PI * 2);
    context.fill();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.fillStyle = "#061113";
    context.fillRect(0, 0, canvas.width, canvas.height);
    setDigitScores(Array(10).fill(0.1));
  };

  const readDrawing = (): number[] => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return Array(196).fill(0);
    const image = context.getImageData(0, 0, canvas.width, canvas.height).data;
    return Array.from({ length: 196 }, (_value, cell) => {
      const row = Math.floor(cell / 14);
      const column = cell % 14;
      let sum = 0;
      for (let y = 0; y < 14; y += 1) for (let x = 0; x < 14; x += 1) sum += image[((row * 14 + y) * 196 + (column * 14 + x)) * 4] / 255;
      return sum / 196;
    });
  };

  const trainDigits = () => {
    setIsDigitTraining(true);
    window.setTimeout(() => { setDigitStudy(digitSession.train(42)); setIsDigitTraining(false); }, 10);
  };

  const trainCnn = () => {
    setIsCnnTraining(true);
    window.setTimeout(() => {
      setCnnStudy(cnn.train(45));
      setIsCnnTraining(false);
    }, 10);
  };

  const runOptimizerComparison = () => {
    setIsComparingOptimizers(true);
    window.setTimeout(() => {
      setTraces(compareOptimizers("xor", 8, 56));
      setIsComparingOptimizers(false);
    }, 10);
  };

  const bestDigit = digitScores.reduce((best, value, index) => value > digitScores[best] ? index : best, 0);
  return (
    <div className="advanced-section">
      <header className="section-intro advanced-intro"><div><p className="eyebrow">03 / extended studies</p><h2>Change the graph.<br /><em>Watch the consequence.</em></h2></div><p>Every study stays inside the browser. These are small, bounded experiments designed to make learning mechanics visible, not to hide them behind a service.</p></header>
      <div className="study-index" aria-label="Advanced study index"><span>DIGIT FIELD</span><i /><span>SENSITIVITY</span><i /><span>MINI-BATCH</span><i /><span>CNN</span><i /><span>OPTIMIZERS</span></div>

      <div className="study-grid digit-study">
        <article className="study-copy"><p className="eyebrow">Study 01 / digit field</p><h3>Sketch a digit.<br />Inspect a <em>distribution.</em></h3><p>A compact 60-example, browser-local MNIST subset is downsampled to 14×14 before entering the from-scratch dense network. Training is intentionally bounded and never fetches data at runtime.</p><dl className="study-stats"><div><dt>{digitStudy.epoch}</dt><dd>training epochs</dd></div><div><dt>{Math.round(digitStudy.accuracy * 100)}%</dt><dd>subset accuracy</dd></div><div><dt>{fmt(digitStudy.loss)}</dt><dd>cross-entropy</dd></div></dl><button className={isDigitTraining ? "primary-action is-loading" : "primary-action"} onClick={trainDigits} disabled={isDigitTraining}>{isDigitTraining ? "Tracing digit loss…" : "Train 42 local epochs"}</button></article>
        <div className="digit-surface"><div className="surface-heading"><span>INPUT CANVAS / 14 × 14 PROJECTION</span><span className="teal">LOCAL ONLY</span></div><canvas ref={canvasRef} className="digit-canvas" width="196" height="196" tabIndex={0} aria-label="Draw a handwritten digit" aria-describedby="digit-canvas-help" onKeyDown={(event) => { if (event.key === "Enter") setDigitScores(digitSession.predict(readDrawing())); if (event.key === "Escape") clearCanvas(); }} onPointerDown={(event) => { drawing.current = true; event.currentTarget.setPointerCapture(event.pointerId); paint(event); }} onPointerMove={paint} onPointerUp={() => { drawing.current = false; }} onPointerLeave={() => { drawing.current = false; }} /><span id="digit-canvas-help" className="sr-only">Draw with a pointer. Press Enter to inspect the current drawing or Escape to clear it.</span><div className="digit-actions"><button className="secondary-action" onClick={() => setDigitScores(digitSession.predict(readDrawing()))}>Inspect drawing</button><button className="text-action" onClick={clearCanvas}>Clear field</button></div></div>
        <div className="score-surface"><div className="surface-heading"><span>10-CLASS CONFIDENCE</span><strong>{bestDigit}</strong></div><div className="score-list">{digitScores.map((score, digit) => <div key={digit} className={digit === bestDigit ? "score-row is-best" : "score-row"}><span>{digit}</span><i><b style={{ width: `${Math.max(2, score * 100)}%` }} /></i><output>{Math.round(score * 100)}%</output></div>)}</div></div>
      </div>

      <div className="study-grid sensitivity-study">
        <article className="study-copy"><p className="eyebrow">Study 02 / sensitivity field</p><h3>Move one weight.<br />Trace the <em>ripple.</em></h3><p>The probe temporarily shifts the first visible input weight, evaluates the current model, then restores the parameter. No training state is overwritten.</p><label className="sensitivity-label" htmlFor="sensitivity">Weight perturbation <output>{perturbation >= 0 ? "+" : ""}{perturbation.toFixed(2)}</output></label><input id="sensitivity" type="range" min="-1" max="1" step="0.02" value={perturbation} onChange={(event) => setPerturbation(Number(event.target.value))} /></article>
        <div className="ripple-surface"><div className="surface-heading"><span>{sensitivity.parameter}</span><span className="teal">PROBE ONLY</span></div><div className="ripple-path"><span>weight</span><i /><span>activation</span><i /><span>output</span><i /><span>loss</span></div><div className="ripple-values"><div><small>baseline loss</small><b>{fmt(sensitivity.before.loss)}</b></div><div><small>probe loss</small><b className="teal">{fmt(sensitivity.after.loss)}</b></div><div><small>Δ loss</small><b className="amber">{(sensitivity.after.loss - sensitivity.before.loss).toExponential(2)}</b></div></div></div>
      </div>

      <div className="study-grid batch-study"><article className="study-copy"><p className="eyebrow">Study 03 / mini-batch lens</p><h3>Many examples.<br />One <em>update.</em></h3><p>Choose how many examples contribute to the next Live Lab update. Their individual gradients are represented here before the average reaches the parameter.</p><div className="batch-switches">{[1, 2, 4, 8].map((size) => <button key={size} className={batchSize === size ? "is-selected" : ""} onClick={() => { setBatchSize(size); onBatchSize(size); }}>{size}</button>)}</div></article><div className="batch-surface"><div className="surface-heading"><span>EXAMPLE GRADIENTS → MEAN UPDATE</span><span>{batchSize} / batch</span></div><div className="gradient-stream">{gradientBars.map((gradient, index) => <div key={`${index}-${gradient}`} className="gradient-particle" style={{ "--height": `${Math.max(11, Math.min(100, Math.abs(gradient) * 7200))}%`, "--delay": `${index * 55}ms` } as React.CSSProperties}><span>g{index + 1}</span><i /></div>)}<div className="gradient-mean"><span>Σg / n</span><b>{fmt(session.snapshot().gradientMagnitude)}</b></div></div></div></div>

      <div className="study-grid cnn-study"><article className="study-copy"><p className="eyebrow">Study 04 / convolution field</p><h3>Slide a kernel.<br />Keep the <em>evidence.</em></h3><p>A valid 3×3 Conv2D layer, ReLU, 2×2 max pooling, flattening, and dense classifier learn three generated geometric classes. The convolution backward pass is covered by finite differences.</p><button className={isCnnTraining ? "primary-action is-loading" : "primary-action"} onClick={trainCnn} disabled={isCnnTraining}>{isCnnTraining ? "Tracing convolution loss…" : "Train 45 CNN epochs"}</button></article><div className="cnn-surface"><div className="surface-heading"><span>6 × 6 SHAPES → 3 × 3 KERNELS</span><span className="teal">VALID CONV</span></div><div className="shape-strip">{ShapeCnnSession.names().map((shape, index) => <div key={shape} className={`shape-token shape-${index}`}><i /><span>{shape}</span></div>)}</div><div className="cnn-metrics"><div><small>epoch</small><b>{cnnStudy.epoch}</b></div><div><small>loss</small><b className="amber">{fmt(cnnStudy.loss)}</b></div><div><small>accuracy</small><b className="teal">{Math.round(cnnStudy.accuracy * 100)}%</b></div></div></div></div>

      <div className="study-grid optimizer-study"><article className="study-copy"><p className="eyebrow">Study 05 / optimizer evidence</p><h3>Same start.<br />Different <em>trajectory.</em></h3><p>SGD, momentum, and Adam start from an identical seed on the XOR task. Their loss traces are calculated locally and overlaid here.</p><button className={isComparingOptimizers ? "secondary-action is-loading" : "secondary-action"} onClick={runOptimizerComparison} disabled={isComparingOptimizers}>{isComparingOptimizers ? "Comparing local traces…" : "Run 56 matched steps"}</button></article><div className="optimizer-surface"><div className="surface-heading"><span>LOSS TRACE / IDENTICAL INITIALIZATION</span><span>56 steps</span></div>{traces.length ? <svg className="optimizer-svg" viewBox="0 0 560 190" role="img" aria-label="Optimizer loss comparison">{traces.map((trace, index) => { const max = Math.max(...traces.flatMap((candidate) => candidate.history)); const points = trace.history.map((loss, step) => `${20 + step * 9.2},${170 - (loss / max) * 132}`).join(" "); return <g key={trace.name}><polyline points={points} className={`optimizer-line line-${index}`} /><text x={425} y={32 + index * 22} className={`optimizer-label line-${index}`}>{trace.name.toUpperCase()} {fmt(trace.finalLoss)}</text></g>; })}</svg> : <div className="optimizer-empty">Run the matched study to reveal three real trajectories.</div>}</div></div>

      <section className="parity-panel"><div><p className="eyebrow">Independent parity benchmark</p><h3>Verified against<br /><em>{PYTORCH_PARITY.framework}.</em></h3><p>A fixed two-layer float64 network uses identical inputs, parameters, targets, forward operations, and reverse gradients in both implementations.</p></div><dl><div><dt>{PYTORCH_PARITY.max_abs_difference.toExponential(2)}</dt><dd>maximum absolute difference</dd></div><div><dt>{PYTORCH_PARITY.dtype}</dt><dd>comparison precision</dd></div><div><dt>{PYTORCH_PARITY.status.toUpperCase()}</dt><dd>parity status</dd></div></dl></section>

      <section className="how-section" id="how"><p className="eyebrow">How this works / illustrated by the live graph above</p><div><h3>Automatic differentiation records local rules during the forward pass, then applies the chain rule in reverse.</h3><p>Each Tensor keeps its parents and a small backward rule. Starting with a scalar loss gradient of one, the engine visits graph nodes in reverse topological order and adds every contribution to each parent gradient. Conv2D follows the same contract: it records where every receptive field touched the input and kernel, then accumulates those local derivatives during the backward pass.</p></div></section>
    </div>
  );
}
