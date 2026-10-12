import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { can } from "../auth/permissions";
import { fetchModelRegistry, fetchTrainingRuns, triggerTraining } from "../api/endpoints";
import { GATE_CRITERIA, type ModelRegistryEntry, type TrainingRun } from "../api/types";
import { LanguageSelect } from "../components/LanguageSelect";
import { SourceTag } from "../components/SourceTag";
import { WriteStatus } from "../components/WriteStatus";
import { useLanguage } from "../hooks/useLanguage";
import { useWrite } from "../hooks/useWrite";

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
  const { user } = useAuth();
  const selection = useLanguage();
  const { language } = selection;
  // Runs and registry rows are kept with the language they were fetched for: a candidate is
  // only ever compared with a champion of its own language.
  const [loaded, setLoaded] = useState<{ language: string; runs: TrainingRun[]; registry: ModelRegistryEntry[] } | null>(null);
  const [runsSource, setRunsSource] = useState<"server" | "mock" | null>(null);
  const [registrySource, setRegistrySource] = useState<"server" | "mock" | null>(null);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [trained, setTrained] = useState<ModelRegistryEntry | null>(null);
  const write = useWrite();

  useEffect(() => {
    if (!language) return;
    let cancelled = false;
    Promise.all([fetchTrainingRuns(language), fetchModelRegistry(language)]).then(([r, reg]) => {
      if (cancelled) return;
      setLoaded({ language, runs: r.data, registry: reg.data });
      setRunsSource(r.source);
      setRegistrySource(reg.source);
      setSelectedRun(r.data.length > 0 ? r.data[0].run_id : null);
      setTrained(null);
    });
    return () => {
      cancelled = true;
    };
  }, [language]);

  const current = loaded && loaded.language === language ? loaded : null;
  const runs = current?.runs ?? null;
  const registry = current?.registry ?? null;
  const run = runs?.find((r) => r.run_id === selectedRun) ?? null;
  const candidateEntry = registry?.find((m) => m.version === run?.candidate_version);
  const championEntry = registry?.find((m) => m.version === run?.champion_version);

  // Training produces a candidate for this one language. It is never live by itself: it still
  // has to pass the gate and be promoted on the registry page. With no server reachable
  // nothing is trained and the page says so (data is null; see endpoints.ts::triggerTraining).
  function onTrain(lang: string) {
    setTrained(null);
    return write.run(`Train ${lang.toUpperCase()}`, () => triggerTraining(lang), (model) => {
      if (!model) return;
      setTrained(model);
      setLoaded((prev) => prev && prev.language === model.language ? { ...prev, registry: [...prev.registry, model] } : prev);
    });
  }

  return (
    <section>
      <header className="view-header">
        <h1>Training visibility (FR-W5)</h1>
        <LanguageSelect selection={selection} />
        <SourceTag source={runsSource} label="training runs" />
        <SourceTag source={registrySource} label="gate results" />
      </header>
      {language && can(user?.role, "training:trigger") && (
        <p>
          <button disabled={write.busy !== null} onClick={() => onTrain(language)}>
            {write.busy ? "Training…" : `Train a ${language.toUpperCase()} candidate now`}
          </button>
        </p>
      )}
      <WriteStatus write={write} />
      {trained && (
        <p role="status">
          Trained <span dir="ltr">{trained.version}</span> for {trained.language.toUpperCase()}. It is a {trained.status},
          not live: it goes through the gate on the Model registry page.
        </p>
      )}
      {language && !runs && <p>Loading…</p>}
      {runs && runs.length === 0 && <p>No training runs recorded for {language?.toUpperCase()}.</p>}
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
