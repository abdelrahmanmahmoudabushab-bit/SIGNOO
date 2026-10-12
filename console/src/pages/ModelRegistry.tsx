import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { can } from "../auth/permissions";
import { completeShadowRun, fetchFleet, fetchModelRegistry, pinModelToKiosk, promoteModel, rollbackModel } from "../api/endpoints";
import type { KioskSummary, ModelRegistryEntry } from "../api/types";
import { LanguageSelect } from "../components/LanguageSelect";
import { SourceTag } from "../components/SourceTag";
import { StatusPill } from "../components/StatusPill";
import { WriteStatus } from "../components/WriteStatus";
import { useLanguage } from "../hooks/useLanguage";
import { useWrite } from "../hooks/useWrite";

// Which model is live for ONE language. The server's rule (routers/registry.py) is "the
// promoted version of this language with the latest promoted_at", but its response carries no
// promoted_at, so with more than one promoted version the console cannot tell which is the
// champion and says exactly that instead of guessing.
function ChampionLine({ language, entries }: { language: string; entries: ModelRegistryEntry[] }) {
  const lang = language.toUpperCase();
  const promoted = entries.filter((e) => e.status === "promoted");
  if (promoted.length === 0) {
    return <p className="champion-line">No champion for {lang}: no model of this language has been promoted.</p>;
  }
  if (promoted.length > 1) {
    return (
      <p className="champion-line">
        {lang} has {promoted.length} promoted versions. The server does not report which was promoted last, so the
        champion for {lang} cannot be identified here.
      </p>
    );
  }
  return (
    <p className="champion-line">
      Champion for {lang}: <strong dir="ltr">{promoted[0].version}</strong>
    </p>
  );
}

export function ModelRegistry() {
  const { user } = useAuth();
  const selection = useLanguage();
  const { language } = selection;
  // Rows are kept with the language they were fetched for, so switching language can never
  // show one language's models (or champion) under another language's name, even briefly.
  const [loaded, setLoaded] = useState<{ language: string; entries: ModelRegistryEntry[] } | null>(null);
  const entries = loaded && loaded.language === language ? loaded.entries : null;
  const [kiosks, setKiosks] = useState<KioskSummary[]>([]);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [kioskSource, setKioskSource] = useState<"server" | "mock" | null>(null);
  // Every action reports what the server said. A refusal (the promotion gate's 409, a 403)
  // is shown as an error and the row is left unchanged; it is never displayed as a success.
  const write = useWrite();

  useEffect(() => {
    let cancelled = false;
    fetchFleet().then((fleet) => {
      if (cancelled) return;
      setKiosks(fleet.data);
      setKioskSource(fleet.source);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!language) return;
    let cancelled = false;
    fetchModelRegistry(language).then((reg) => {
      if (cancelled) return;
      setLoaded({ language, entries: reg.data });
      setSource(reg.source);
    });
    return () => {
      cancelled = true;
    };
  }, [language]);

  const setStatus = (version: string) => (result: { status: string }) =>
    setLoaded(
      (prev) =>
        prev && {
          ...prev,
          entries: prev.entries.map((e) => (e.version === version ? { ...e, status: result.status as ModelRegistryEntry["status"] } : e)),
        }
    );

  const onPromote = (version: string) => write.run(version, () => promoteModel(version), setStatus(version));
  const onCompleteShadow = (version: string) => write.run(version, () => completeShadowRun(version), setStatus(version));
  const onRollback = (version: string) => write.run(version, () => rollbackModel(version), setStatus(version));
  const onPin = (version: string, kioskId: string) => write.run(version, () => pinModelToKiosk(version, kioskId), () => {});

  const canPromote = can(user?.role, "registry:promote");
  const champion = entries?.filter((e) => e.status === "promoted") ?? [];

  return (
    <section>
      <header className="view-header">
        <h1>Model registry (FR-W6)</h1>
        <LanguageSelect selection={selection} />
        <SourceTag source={source} label="registry" />
        {canPromote && <SourceTag source={kioskSource} label="kiosk list" />}
      </header>
      <WriteStatus write={write} />
      {language && !entries && <p>Loading…</p>}
      {language && entries && (
        <>
          <ChampionLine language={language} entries={entries} />
          {entries.length === 0 && <p>No model versions for {language.toUpperCase()} yet.</p>}
        </>
      )}
      {language && entries && entries.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>Version</th>
              <th>Language</th>
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
                <td>{e.language.toUpperCase()}</td>
                <td dir="ltr">{e.code_commit}</td>
                <td dir="ltr">{e.data_snapshot}</td>
                <td>{(e.metrics.top1 * 100).toFixed(1)}%</td>
                <td className={e.gate.passed ? "gate-pass" : "gate-fail"}>{e.gate.passed ? "pass" : "fail"}</td>
                <td>
                  <StatusPill status={e.status} />
                  {champion.length === 1 && champion[0].version === e.version && (
                    <strong className="champion-mark"> champion for {e.language.toUpperCase()}</strong>
                  )}
                </td>
                {canPromote && (
                  <td className="registry-actions">
                    {e.status === "candidate" && (
                      <button disabled={!e.gate.passed || write.busy === e.version} onClick={() => onCompleteShadow(e.version)}>
                        Complete shadow run
                      </button>
                    )}
                    <button disabled={!e.gate.passed || e.status !== "shadow" || write.busy === e.version} onClick={() => onPromote(e.version)}>
                      Promote
                    </button>
                    <button disabled={e.status !== "promoted" || write.busy === e.version} onClick={() => onRollback(e.version)}>
                      Roll back
                    </button>
                    {kiosks.length > 0 && (
                      <select
                        defaultValue=""
                        aria-label={`Pin ${e.version} to a kiosk`}
                        disabled={write.busy === e.version}
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
