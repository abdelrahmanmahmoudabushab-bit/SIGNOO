import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { can } from "../auth/permissions";
import { completeShadowRun, fetchFleet, fetchModelRegistry, pinModelToKiosk, promoteModel, rollbackModel } from "../api/endpoints";
import type { KioskSummary, ModelRegistryEntry } from "../api/types";
import { SourceTag } from "../components/SourceTag";
import { StatusPill } from "../components/StatusPill";

export function ModelRegistry() {
  const { user } = useAuth();
  const [entries, setEntries] = useState<ModelRegistryEntry[] | null>(null);
  const [kiosks, setKiosks] = useState<KioskSummary[]>([]);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchModelRegistry(), fetchFleet()]).then(([reg, fleet]) => {
      if (cancelled) return;
      setEntries(reg.data);
      setKiosks(fleet.data);
      setSource(reg.source);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Every action reports what the server said. A refusal (the promotion gate's 409, a 403)
  // is shown as an error and the row is left unchanged; it is never displayed as a success.
  async function act(version: string, run: () => Promise<void>) {
    setBusy(version);
    setError(null);
    try {
      await run();
    } catch (e) {
      setError(`${version}: ${e instanceof Error ? e.message : "action failed"}`);
    } finally {
      setBusy(null);
    }
  }

  function setStatus(version: string, status: string) {
    setEntries((prev) => prev?.map((e) => (e.version === version ? { ...e, status: status as ModelRegistryEntry["status"] } : e)) ?? null);
  }

  const onPromote = (version: string) => act(version, async () => setStatus(version, (await promoteModel(version)).data.status));
  const onCompleteShadow = (version: string) => act(version, async () => setStatus(version, (await completeShadowRun(version)).data.status));
  const onRollback = (version: string) => act(version, async () => setStatus(version, (await rollbackModel(version)).data.status));
  const onPin = (version: string, kioskId: string) => act(version, async () => void (await pinModelToKiosk(version, kioskId)));

  const canPromote = can(user?.role, "registry:promote");

  return (
    <section>
      <header className="view-header">
        <h1>Model registry (FR-W6)</h1>
        <SourceTag source={source} />
      </header>
      {error && <p role="alert" className="gate-fail">{error}</p>}
      {!entries && <p>Loading…</p>}
      {entries && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Version</th>
              <th>Commit</th>
              <th>Data snapshot</th>
              <th>Top-1</th>
              <th>Gate</th>
              <th>Status</th>
              {canPromote && <th>Actions</th>}
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.version}>
                <td dir="ltr">{e.version}</td>
                <td dir="ltr">{e.code_commit}</td>
                <td dir="ltr">{e.data_snapshot}</td>
                <td>{(e.metrics.top1 * 100).toFixed(1)}%</td>
                <td className={e.gate.passed ? "gate-pass" : "gate-fail"}>{e.gate.passed ? "pass" : "fail"}</td>
                <td>
                  <StatusPill status={e.status} />
                </td>
                {canPromote && (
                  <td className="registry-actions">
                    {e.status === "candidate" && (
                      <button disabled={!e.gate.passed || busy === e.version} onClick={() => onCompleteShadow(e.version)}>
                        Complete shadow run
                      </button>
                    )}
                    <button disabled={!e.gate.passed || e.status !== "shadow" || busy === e.version} onClick={() => onPromote(e.version)}>
                      Promote
                    </button>
                    <button disabled={e.status !== "promoted" || busy === e.version} onClick={() => onRollback(e.version)}>
                      Roll back
                    </button>
                    {kiosks.length > 0 && (
                      <select
                        defaultValue=""
                        disabled={busy === e.version}
                        onChange={(ev) => {
                          if (ev.target.value) onPin(e.version, ev.target.value);
                        }}
                      >
                        <option value="" disabled>
                          Pin to kiosk…
                        </option>
                        {kiosks.map((k) => (
                          <option key={k.kiosk_id} value={k.kiosk_id}>
                            {k.kiosk_id}
                          </option>
                        ))}
                      </select>
                    )}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
