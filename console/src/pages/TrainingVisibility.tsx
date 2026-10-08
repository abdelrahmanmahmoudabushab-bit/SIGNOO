import { useEffect, useState } from "react";
import { fetchModelRegistry, fetchTrainingRuns } from "../api/endpoints";
import { GATE_CRITERIA, type ModelRegistryEntry, type TrainingRun } from "../api/types";
import { SourceTag } from "../components/SourceTag";

// FR-W5: show every one of the 7 DR-6 gate criteria explicitly, never collapse to a single
// pass/fail badge — the whole point of the gate is that "the judgement is not automatic",
// and hiding the criteria behind a badge would undo that transparency.
function GateVerdict({ entry }: { entry: ModelRegistryEntry | undefined }) {
  if (!entry) return <p>No gate result recorded for this candidate yet.</p>;
  return (
    <div className="gate-verdict">
      <p className={entry.gate.passed ? "gate-pass" : "gate-fail"}>
        Overall: {entry.gate.passed ? "PASSED — eligible for promotion" : "FAILED — champion retained"}
      </p>
      <table className="data-table">
        <thead>
          <tr>
            <th>Criterion (DR-6)</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {GATE_CRITERIA.map((c) => (
            <tr key={c.key}>
              <td>{c.label}</td>
              <td className={entry.gate.criteria[c.key] === "pass" ? "gate-pass" : "gate-fail"}>
                {entry.gate.criteria[c.key] ?? "not evaluated"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TrainingVisibility() {
  const [runs, setRuns] = useState<TrainingRun[] | null>(null);
  const [registry, setRegistry] = useState<ModelRegistryEntry[] | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchTrainingRuns(), fetchModelRegistry()]).then(([r, reg]) => {
      if (cancelled) return;
      setRuns(r.data);
      setRegistry(reg.data);
      setSource(r.source === "server" && reg.source === "server" ? "server" : "mock");
      if (r.data.length > 0) setSelectedRun(r.data[0].run_id);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const run = runs?.find((r) => r.run_id === selectedRun) ?? null;
  const candidateEntry = registry?.find((m) => m.version === run?.candidate_version);
  const championEntry = registry?.find((m) => m.version === run?.champion_version);

  return (
    <section>
      <header className="view-header">
        <h1>Training visibility (FR-W5)</h1>
        <SourceTag source={source} />
      </header>
      {!runs && <p>Loading…</p>}
      {runs && (
        <div className="training-layout">
          <ul className="run-list">
            {runs.map((r) => (
              <li key={r.run_id}>
                <button className={r.run_id === selectedRun ? "active" : undefined} onClick={() => setSelectedRun(r.run_id)}>
                  {r.run_id} — {r.status}
                </button>
              </li>
            ))}
          </ul>
          {run && (
            <div className="run-detail">
              <h2 dir="ltr">{run.run_id}</h2>
              <p>
                Started {new Date(run.started_at).toLocaleString()}
                {run.finished_at && ` · finished ${new Date(run.finished_at).toLocaleString()}`}
              </p>
              <h3>Training curve</h3>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Train loss</th>
                    <th>Val accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {run.curves.map((c) => (
                    <tr key={c.epoch}>
                      <td>{c.epoch}</td>
                      <td>{c.train_loss.toFixed(2)}</td>
                      <td>{(c.val_accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <h3>Diff vs champion</h3>
              {candidateEntry && championEntry ? (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th></th>
                      <th>Champion</th>
                      <th>Candidate</th>
                      <th>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Top-1</td>
                      <td>{(championEntry.metrics.top1 * 100).toFixed(1)}%</td>
                      <td>{(candidateEntry.metrics.top1 * 100).toFixed(1)}%</td>
                      <td>
                        {((candidateEntry.metrics.top1 - championEntry.metrics.top1) * 100).toFixed(1)} pts
                      </td>
                    </tr>
                    <tr>
                      <td>Top-5</td>
                      <td>{(championEntry.metrics.top5 * 100).toFixed(1)}%</td>
                      <td>{(candidateEntry.metrics.top5 * 100).toFixed(1)}%</td>
                      <td>
                        {((candidateEntry.metrics.top5 - championEntry.metrics.top5) * 100).toFixed(1)} pts
                      </td>
                    </tr>
                  </tbody>
                </table>
              ) : (
                <p>No prior champion to diff against (first trained model).</p>
              )}

              <h3>Promotion gate verdict (DR-6)</h3>
              <GateVerdict entry={candidateEntry} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
