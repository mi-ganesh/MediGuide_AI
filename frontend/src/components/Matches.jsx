import { useState } from "react";

export default function Matches({ predictions, detailed = false }) {
  const [open, setOpen] = useState(0);

  return (
    <ol className="matches" aria-live="polite">
      {predictions.map((p, i) => {
        const isOpen = detailed && open === i;
        const Head = detailed ? "button" : "div";
        return (
          <li key={p.disease} className={`result ${isOpen ? "open" : ""}`}>
            <Head
              className="result-head"
              {...(detailed ? { "aria-expanded": isOpen, onClick: () => setOpen(isOpen ? -1 : i) } : {})}
            >
              <span className="name">{p.disease}</span>
              <span className="pct">{p.probability}%</span>
              <span className="bar" aria-hidden="true">
                <span style={{ width: `${Math.max(p.probability, 2)}%` }} />
              </span>
            </Head>

            {isOpen && (
              <div className="result-body">
                {p.description && <p>{p.description}</p>}
                <h3>What you can do</h3>
                <ul className="list">
                  {p.precautions.map((x) => <li key={x}>{x}</li>)}
                </ul>
                <p className="who"><strong>Who to see:</strong> {p.specialist}</p>
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
