/** Signal Observatory: a non-blocking first-visit guide that orients learners without hiding the live instrument. */
type GuidedTrackProps = {
  step: number | null;
  onAdvance: () => void;
  onDismiss: () => void;
  onRestart: () => void;
};

const STEPS = [
  { label: "Choose a study", copy: "Select XOR, Spiral, or Sine. Each exposes a different kind of learning signal." },
  { label: "Run the system", copy: "Observe a bounded training run. The loss, predictions, and gradients update together." },
  { label: "Trace the evidence", copy: "Read the graph from output to input, then step one backward pass when you are ready." },
];

export function GuidedTrack({ step, onAdvance, onDismiss, onRestart }: GuidedTrackProps) {
  if (step === null) {
    return <button className="guide-restart" onClick={onRestart}><span aria-hidden="true">↗</span> Restart first-visit guide</button>;
  }
  const current = STEPS[step];
  const isLast = step === STEPS.length - 1;
  return (
    <aside className="guided-track" aria-label="First-visit guide" aria-live="polite">
      <div className="guided-track-top"><p className="eyebrow"><span className="mini-flow" aria-hidden="true" /> First-visit route</p><button className="guide-dismiss" onClick={onDismiss} aria-label="Dismiss first-visit guide">Dismiss</button></div>
      <div className="guided-track-body"><span className="guide-count">0{step + 1} / 03</span><div><h3>{current.label}</h3><p>{current.copy}</p></div><button className="guide-next" onClick={onAdvance}>{isLast ? "Finish route" : "Next observation"}<span aria-hidden="true">→</span></button></div>
      <ol className="guide-steps">{STEPS.map((item, index) => <li key={item.label} className={index === step ? "is-current" : index < step ? "is-complete" : ""}><i>{index < step ? "✓" : `0${index + 1}`}</i><span>{item.label}</span></li>)}</ol>
    </aside>
  );
}
