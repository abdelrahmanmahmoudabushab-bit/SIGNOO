import { useEffect, useState } from "react";
import { fetchCoverage, fetchDatasetSessions, retireSession } from "../api/endpoints";
import type { DatasetSession, SignCoverage } from "../api/types";
import { SourceTag } from "../components/SourceTag";

export function DatasetManagement() {
  const [coverage, setCoverage] = useState<SignCoverage[] | null>(null);
  const [sessions, setSessions] = useState<DatasetSession[] | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchCoverage(), fetchDatasetSessions()]).then(([cov, sess]) => {
      if (cancelled) return;
      setCoverage(cov.data);
      setSessions(sess.data);
      setSource(cov.source === "server" && sess.source === "server" ? "server" : "mock");
    });
    return () => {
      cancelled = true;
    };
  }, []);

  async function onRetire(sessionId: string) {
    await retireSession(sessionId);
    setSessions((prev) => prev?.map((s) => (s.session_id === sessionId ? { ...s, status: "retired" } : s)) ?? null);
  }

  return (
    <section>
      <header className="view-header">
        <h1>Dataset management (FR-W4)</h1>
        <SourceTag source={source} />
      </header>

      <h2>Coverage by sign</h2>
      {!coverage && <p>Loading…</p>}
      {coverage && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Sign</th>
              <th>Language</th>
              <th>Signers</th>
              <th>Clips</th>
              <th>Gap to target</th>
            </tr>
          </thead>
          <tbody>
            {coverage.map((c) => {
              const gap = c.target_signers - c.signer_count;
              return (
                <tr key={c.sign_id} className={gap > 0 ? "coverage-gap" : undefined}>
                  <td>{c.sign_id}</td>
                  <td>{c.language.toUpperCase()}</td>
                  <td>
                    {c.signer_count} / {c.target_signers}
                  </td>
                  <td>
                    {c.clip_count} / {c.target_clips}
                  </td>
                  <td>{gap > 0 ? `${gap} more signers needed` : "target met"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      <h2>Recorded sessions</h2>
      {!sessions && <p>Loading…</p>}
      {sessions && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Session</th>
              <th>Language</th>
              <th>Signer</th>
              <th>Recorded</th>
              <th>Clips</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.session_id}>
                <td dir="ltr">{s.session_id}</td>
                <td>{s.language.toUpperCase()}</td>
                <td dir="ltr">{s.signer_id}</td>
                <td>{new Date(s.recorded_at).toLocaleDateString()}</td>
                <td>{s.clip_count}</td>
                <td>{s.status}</td>
                <td>
                  {s.status === "active" && <button onClick={() => onRetire(s.session_id)}>Retire</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
