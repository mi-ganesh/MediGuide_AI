export default function Question({ question, asked, max, busy, canUndo, onAnswer, onUndo, onShowNow }) {
  return (
    <section className="panel question" aria-live="polite">
      <p className="muted small">Question {asked + 1} of up to {max}</p>
      <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={max} aria-valuenow={asked}>
        <span style={{ width: `${(asked / max) * 100}%` }} />
      </div>

      <h2>Do you have {question.label.toLowerCase()}?</h2>

      <div className="answers">
        <button className="btn yes" disabled={busy} onClick={() => onAnswer("yes")}>Yes</button>
        <button className="btn no" disabled={busy} onClick={() => onAnswer("no")}>No</button>
        <button className="btn ghost" disabled={busy} onClick={() => onAnswer("skip")}>Not sure</button>
      </div>

      <div className="actions">
        {canUndo && <button className="btn ghost" disabled={busy} onClick={onUndo}>Undo last answer</button>}
        <button className="btn ghost" disabled={busy} onClick={onShowNow}>Show results now</button>
      </div>
    </section>
  );
}
