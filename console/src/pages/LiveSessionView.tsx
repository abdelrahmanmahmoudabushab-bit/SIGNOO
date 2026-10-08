import { useEffect, useState } from "react";
import { fetchLiveSessions } from "../api/endpoints";
import type { LiveSession } from "../api/types";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { SourceTag } from "../components/SourceTag";

// FR-W2: camera imagery is explicitly out of scope for now (no video is ever retained per
// NFR-S2). This is rendered as a genuinely disabled control with an explanatory tooltip —
// not a toggle that silently does nothing — so an operator never wonders whether it's broken.
function CameraImageryControl() {
  return (
    <button
      disabled
      className="camera-control-disabled"
      title="Camera imagery requires elevated authorisation and is audited (NFR-S4). No video is retained by this system (NFR-S2) — this control has no live feed to show even for an authorised admin session."
    >
      View camera imagery (disabled)
    </button>
  );
}

export function LiveSessionView() {
  const [sessions, setSessions] = useState<LiveSession[] | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchLiveSessions().then(({ data, source: src }) => {
      if (!cancelled) {
        setSessions(data);
        setSource(src);
        if (data.length > 0) setSelected(data[0].session_id);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const current = sessions?.find((s) => s.session_id === selected) ?? null;

  return (
    <section>
      <header className="view-header">
        <h1>Live session view (FR-W2)</h1>
        <SourceTag source={source} />
      </header>
      {!sessions && <p>Loading…</p>}
      {sessions && sessions.length === 0 && <p>No sessions currently in progress.</p>}
      {sessions && sessions.length > 0 && (
        <div className="live-session-layout">
          <ul className="session-list">
            {sessions.map((s) => (
              <li key={s.session_id}>
                <button className={s.session_id === selected ? "active" : undefined} onClick={() => setSelected(s.session_id)}>
                  {s.kiosk_id} — {s.language.toUpperCase()} — {s.status}
                </button>
              </li>
            ))}
          </ul>
          {current && (
            <div className="session-detail">
              <div className="session-detail-header">
                <h2 dir="ltr">{current.session_id}</h2>
                <CameraImageryControl />
              </div>
              <p>
                Started {new Date(current.started_at).toLocaleString()} · status: {current.status}
              </p>
              <ol className="recognition-stream">
                {current.events.map((ev) => (
                  <li key={ev.seq}>
                    <span className="gloss">{ev.gloss}</span>
                    <ConfidenceBadge confidence={ev.confidence} />
                    <span className="timestamp">{new Date(ev.timestamp).toLocaleTimeString()}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
