import { useEffect, useState } from "react";
import { fetchFleet } from "../api/endpoints";
import type { KioskSummary } from "../api/types";
import { SourceTag } from "../components/SourceTag";
import { StatusPill } from "../components/StatusPill";

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  if (days > 0) return `${days}d ${hours}h`;
  return `${hours}h`;
}

export function FleetView() {
  const [kiosks, setKiosks] = useState<KioskSummary[] | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchFleet().then(({ data, source: src }) => {
      if (!cancelled) {
        setKiosks(data);
        setSource(src);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section>
      <header className="view-header">
        <h1>Fleet (FR-W1)</h1>
        <SourceTag source={source} />
      </header>
      {!kiosks && <p>Loading…</p>}
      {kiosks && kiosks.length === 0 && <p>No kiosks registered.</p>}
      {kiosks && kiosks.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Kiosk</th>
              <th>Site</th>
              <th>Language</th>
              <th>Health</th>
              <th>Model version</th>
              <th>Uptime</th>
              <th>Temperature</th>
              <th>Last contact</th>
            </tr>
          </thead>
          <tbody>
            {kiosks.map((k) => (
              <tr key={k.kiosk_id}>
                <td dir="ltr">{k.kiosk_id}</td>
                <td>{k.site_id}</td>
                <td>{k.language.toUpperCase()}</td>
                <td>
                  <StatusPill status={k.health} />
                </td>
                <td dir="ltr">{k.model_version}</td>
                <td>{formatUptime(k.uptime_seconds)}</td>
                <td>{k.temperature_c === null ? "unavailable (non-Jetson)" : `${k.temperature_c}°C`}</td>
                <td>{new Date(k.last_contact).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
