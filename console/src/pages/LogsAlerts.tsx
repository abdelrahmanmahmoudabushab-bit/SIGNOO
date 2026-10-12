import { useEffect, useMemo, useState } from "react";
import { acknowledgeAlert, fetchAlerts, fetchLogs } from "../api/endpoints";
import type { AlertEntry, LogEntry } from "../api/types";
import { SourceTag } from "../components/SourceTag";
import { StatusPill } from "../components/StatusPill";
import { WriteStatus } from "../components/WriteStatus";
import { useWrite } from "../hooks/useWrite";

export function LogsAlerts() {
  const [logs, setLogs] = useState<LogEntry[] | null>(null);
  const [alerts, setAlerts] = useState<AlertEntry[] | null>(null);
  const [logsSource, setLogsSource] = useState<"server" | "mock" | null>(null);
  const [alertsSource, setAlertsSource] = useState<"server" | "mock" | null>(null);
  const write = useWrite();
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchLogs(), fetchAlerts()]).then(([l, a]) => {
      if (cancelled) return;
      setLogs(l.data);
      setAlerts(a.data);
      setLogsSource(l.source);
      setAlertsSource(a.source);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredLogs = useMemo(() => {
    if (!logs) return null;
    const q = query.trim().toLowerCase();
    if (!q) return logs;
    return logs.filter(
      (l) => l.message.toLowerCase().includes(q) || l.source.toLowerCase().includes(q) || (l.correlation_id ?? "").toLowerCase().includes(q)
    );
  }, [logs, query]);

  // A refused acknowledgement shows the server's reason and the alert stays unacknowledged.
  const onAck = (id: string) =>
    write.run(`alert ${id}`, () => acknowledgeAlert(id), () =>
      setAlerts((prev) => prev?.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)) ?? null)
    );

  return (
    <section>
      <header className="view-header">
        <h1>Logs and alerts (FR-W8)</h1>
        <SourceTag source={alertsSource} label="alerts" />
        <SourceTag source={logsSource} label="logs" />
      </header>
      <WriteStatus write={write} />

      <h2>Alerts</h2>
      {!alerts && <p>Loading…</p>}
      {alerts && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Severity</th>
              <th>Kind</th>
              <th>Kiosk</th>
              <th>Message</th>
              <th>Raised</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {alerts.map((a) => (
              <tr key={a.id} className={a.acknowledged ? "acknowledged" : undefined}>
                <td>
                  <StatusPill status={a.severity} />
                </td>
                <td>{a.kind.replace(/_/g, " ")}</td>
                <td dir="ltr">{a.kiosk_id ?? "—"}</td>
                <td>{a.message}</td>
                <td>{new Date(a.raised_at).toLocaleString()}</td>
                <td>
                  {!a.acknowledged && (
                    <button disabled={write.busy === `alert ${a.id}`} onClick={() => onAck(a.id)}>
                      Acknowledge
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Structured logs</h2>
      <input
        className="log-search"
        placeholder="Search message, source, or correlation id…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        aria-label="Search logs"
      />
      {!filteredLogs && <p>Loading…</p>}
      {filteredLogs && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Level</th>
              <th>Source</th>
              <th>Correlation ID</th>
              <th>Message</th>
            </tr>
          </thead>
          <tbody>
            {filteredLogs.map((l) => (
              <tr key={l.id}>
                <td>{new Date(l.timestamp).toLocaleString()}</td>
                <td>
                  <StatusPill status={l.level} />
                </td>
                <td dir="ltr">{l.source}</td>
                <td dir="ltr">{l.correlation_id ?? "—"}</td>
                <td>{l.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
