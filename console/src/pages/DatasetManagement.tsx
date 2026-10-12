import { useEffect, useState } from "react";
import { fetchCoverage, fetchDatasetSessions, retireSession } from "../api/endpoints";
import type { DatasetSession, SignCoverage } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { can } from "../auth/permissions";
import { SourceTag } from "../components/SourceTag";
import { WriteStatus } from "../components/WriteStatus";
import { useWrite } from "../hooks/useWrite";

export function DatasetManagement() {
  const [coverage, setCoverage] = useState<SignCoverage[] | null>(null);
  const [sessions, setSessions] = useState<DatasetSession[] | null>(null);
  const [coverageSource, setCoverageSource] = useState<"server" | "mock" | null>(null);
  const [sessionsSource, setSessionsSource] = useState<"server" | "mock" | null>(null);
  const { user } = useAuth();
  const canManage = can(user?.role, "dataset:manage");
  const write = useWrite();

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchCoverage(), fetchDatasetSessions()]).then(([cov, sess]) => {
      if (cancelled) return;
      setCoverage(cov.data);
      setSessions(sess.data);
      setCoverageSource(cov.source);
      setSessionsSource(sess.source);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // A refused retire shows the server's reason and the session stays "active".
  const onRetire = (sessionId: string) =>
    write.run(sessionId, () => retireSession(sessionId), () =>
      setSessions((prev) => prev?.map((s) => (s.session_id === sessionId ? { ...s, status: "retired" } : s)) ?? null)
    );

  return (
    <section>
      <header className="view-header">
        <h1>Dataset management (FR-W4)</h1>
        <SourceTag source={coverageSource} label="coverage" />
        <SourceTag source={sessionsSource} label="sessions" />
      </header>
      <WriteStatus write={write} />

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
                  {canManage && s.status === "active" && (
                    <button disabled={write.busy === s.session_id} onClick={() => onRetire(s.session_id)}>
                      Retire
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
