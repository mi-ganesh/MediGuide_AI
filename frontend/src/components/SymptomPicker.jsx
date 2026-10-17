import { useMemo, useState } from "react";

export default function SymptomPicker({ symptoms, selected, labelOf, busy, onAdd, onRemove, onStart }) {
  const [query, setQuery] = useState("");

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return symptoms
      .filter((s) => !selected.includes(s.id) && s.label.toLowerCase().includes(q))
      .slice(0, 10);
  }, [symptoms, selected, query]);

  const popular = symptoms.filter((s) => s.popular && !selected.includes(s.id));

  const add = (id) => {
    onAdd(id);
    setQuery("");
  };

  return (
    <section className="panel picker" aria-labelledby="picker-title">
      <h2 id="picker-title">What are you feeling?</h2>
      <p className="muted small">Add one or more symptoms to start. More detail gives better results.</p>

      <input
        type="search"
        className="search"
        placeholder="Search symptoms, e.g. headache"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search symptoms"
      />

      {query.trim() ? (
        matches.length ? (
          <ul className="suggest">
            {matches.map((s) => (
              <li key={s.id}><button onClick={() => add(s.id)}>{s.label}</button></li>
            ))}
          </ul>
        ) : (
          <p className="muted">No symptom matches “{query}”. Try a shorter word.</p>
        )
      ) : (
        <>
          <h3 className="sub">Common symptoms</h3>
          {symptoms.length === 0 ? (
            <p className="muted">Loading symptoms…</p>
          ) : (
            <ul className="chips">
              {popular.map((s) => (
                <li key={s.id}><button className="chip" onClick={() => add(s.id)}>{s.label}</button></li>
              ))}
            </ul>
          )}
        </>
      )}

      <h3 className="sub">Your symptoms</h3>
      {selected.length === 0 ? (
        <p className="muted">None yet.</p>
      ) : (
        <ul className="tags">
          {selected.map((id) => (
            <li key={id}>
              <button className="tag yes" onClick={() => onRemove(id)} aria-label={`Remove ${labelOf(id)}`}>
                {labelOf(id)} <span aria-hidden="true">×</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="actions">
        <button className="btn primary" onClick={onStart} disabled={!selected.length || busy}>
          {busy ? "Checking…" : "Start assessment"}
        </button>
      </div>
    </section>
  );
}
