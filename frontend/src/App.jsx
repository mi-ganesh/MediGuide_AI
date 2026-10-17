import { useEffect, useMemo, useState } from "react";
import { fetchSymptoms, assess } from "./api";
import SymptomPicker from "./components/SymptomPicker.jsx";
import Question from "./components/Question.jsx";
import Matches from "./components/Matches.jsx";

const MAX_QUESTIONS = 7;

const buildAnswers = (initial, history) => ({
  confirmed: [...initial, ...history.filter((h) => h.kind === "yes").map((h) => h.symptom)],
  denied: history.filter((h) => h.kind === "no").map((h) => h.symptom),
  skipped: history.filter((h) => h.kind === "skip").map((h) => h.symptom),
});

export default function App() {
  const [symptoms, setSymptoms] = useState([]);
  const [loadError, setLoadError] = useState("");
  const [phase, setPhase] = useState("pick"); // pick | ask | done
  const [initial, setInitial] = useState([]); // symptoms the person starts with
  const [history, setHistory] = useState([]); // [{symptom, kind: yes|no|skip}]
  const [assessment, setAssessment] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const loadSymptoms = () => {
    setLoadError("");
    fetchSymptoms().then(setSymptoms).catch((e) => setLoadError(e.message));
  };
  useEffect(loadSymptoms, []);

  const labels = useMemo(() => Object.fromEntries(symptoms.map((s) => [s.id, s.label])), [symptoms]);
  const label = (id) => labels[id] ?? id;

  async function run(nextInitial, nextHistory) {
    setBusy(true);
    setError("");
    try {
      const res = await assess({
        ...buildAnswers(nextInitial, nextHistory),
        asked: nextHistory.length,
        max_questions: MAX_QUESTIONS,
      });
      setInitial(nextInitial);
      setHistory(nextHistory);
      setAssessment(res);
      setPhase(res.finished ? "done" : "ask");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const addSymptom = (id) => setInitial((cur) => (cur.includes(id) ? cur : [...cur, id]));
  const removeSymptom = (id) => setInitial((cur) => cur.filter((s) => s !== id));
  const start = () => run(initial, []);
  const answer = (kind) =>
    run(initial, [...history, { symptom: assessment.next_question.symptom, kind }]);
  const undo = () => (history.length ? run(initial, history.slice(0, -1)) : null);

  const restart = () => {
    setInitial([]);
    setHistory([]);
    setAssessment(null);
    setError("");
    setPhase("pick");
  };
  const editSymptoms = () => {
    setHistory([]);
    setAssessment(null);
    setError("");
    setPhase("pick");
  };

  const { confirmed, denied } = buildAnswers(initial, history);
  const top = assessment?.predictions?.[0];
  const strong = top && assessment && top.probability >= assessment.threshold * 100;

  return (
    <div className="page">
      <header className="masthead">
        <h1>MediGuide AI</h1>
        <p>
          Tell us what you feel. We ask a few follow-up questions, then show the conditions that fit
          best. It helps you decide what to ask a doctor, it does not replace one.
        </p>
      </header>

      {loadError ? (
        <div className="notice error" role="alert">
          <p>{loadError}</p>
          <button className="btn ghost" onClick={loadSymptoms}>Try again</button>
        </div>
      ) : (
        <main>
          {error && <p className="notice error" role="alert">{error}</p>}

          {assessment?.urgent && phase !== "pick" && (
            <div className="urgent" role="alert">
              <strong>{assessment.urgent.symptoms.join(", ")}</strong>
              <p>{assessment.urgent.message}</p>
            </div>
          )}

          {phase === "pick" && (
            <SymptomPicker
              symptoms={symptoms}
              selected={initial}
              labelOf={label}
              busy={busy}
              onAdd={addSymptom}
              onRemove={removeSymptom}
              onStart={start}
            />
          )}

          {phase === "ask" && assessment?.next_question && (
            <div className="layout">
              <Question
                question={assessment.next_question}
                asked={history.length}
                max={MAX_QUESTIONS}
                busy={busy}
                canUndo={history.length > 0}
                onAnswer={answer}
                onUndo={undo}
                onShowNow={() => setPhase("done")}
              />
              <aside className="panel">
                <h2>Best matches so far</h2>
                <Matches predictions={assessment.predictions} />
                <h3 className="sub">What you told us</h3>
                <ul className="tags">
                  {confirmed.map((s) => <li key={s} className="tag yes">{label(s)}</li>)}
                  {denied.map((s) => <li key={s} className="tag no">No {label(s).toLowerCase()}</li>)}
                </ul>
              </aside>
            </div>
          )}

          {phase === "done" && assessment && (
            <section className="panel done">
              <h2>
                {strong
                  ? `Strong match: ${top.disease}`
                  : "No single clear match yet. These are the closest conditions."}
              </h2>
              <p className="muted small">
                Based on {confirmed.length} symptom{confirmed.length === 1 ? "" : "s"} you have
                {denied.length ? ` and ${denied.length} you do not` : ""}.
              </p>
              <Matches predictions={assessment.predictions} detailed />
              <p className="notice">{assessment.disclaimer}</p>
              <div className="actions">
                <button className="btn primary" onClick={restart}>Start over</button>
                <button className="btn ghost" onClick={editSymptoms}>Change my symptoms</button>
                {assessment.next_question && (
                  <button className="btn ghost" onClick={() => setPhase("ask")}>Answer more questions</button>
                )}
              </div>
            </section>
          )}
        </main>
      )}

      <footer className="foot">
        Not a medical diagnosis. If you have severe pain, trouble breathing or feel unsafe, contact emergency services.
      </footer>
    </div>
  );
}
