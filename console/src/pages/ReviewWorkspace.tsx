import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { fetchReviewQueue, submitReviewDecision } from "../api/endpoints";
import type { ReviewDecisionKind, ReviewQueueItem } from "../api/types";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { LanguageSelect } from "../components/LanguageSelect";
import { SourceTag } from "../components/SourceTag";
import { WriteStatus } from "../components/WriteStatus";
import { useLanguage } from "../hooks/useLanguage";
import { useWrite } from "../hooks/useWrite";

// FR-W3: THE workspace the whole learning loop depends on. Keyboard-driven, one decision per
// screen, auto-advance — reviewer time is the scarcest resource in the system (ROADMAP.md R3),
// so every decision must be reachable without touching the mouse.
//
// Keys: C = confirm · X = correct (opens a text field, Enter submits) · F = flag for linguist
// · Esc cancels an open correction field.
//
// The queue is one language at a time: a reviewer's decisions become that language's training
// data, so items of another language are never shown in the same queue.
export function ReviewWorkspace() {
  const { user } = useAuth();
  const selection = useLanguage();
  const { language } = selection;
  // Kept with the language it was fetched for, so a language switch never shows the previous
  // language's item under the new language, even briefly.
  const [loaded, setLoaded] = useState<{ language: string; queue: ReviewQueueItem[]; index: number } | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [correcting, setCorrecting] = useState(false);
  const [correctionValue, setCorrectionValue] = useState("");
  const [decidedCount, setDecidedCount] = useState(0);
  const correctionInputRef = useRef<HTMLInputElement>(null);
  const write = useWrite();
  const { busy, run } = write;

  useEffect(() => {
    if (!language) return;
    let cancelled = false;
    fetchReviewQueue(language).then(({ data, source: src }) => {
      if (!cancelled) {
        setLoaded({ language, queue: data, index: 0 });
        setSource(src);
        setCorrecting(false);
        setCorrectionValue("");
      }
    });
    return () => {
      cancelled = true;
    };
  }, [language]);

  const queue = loaded && loaded.language === language ? loaded.queue : null;
  const index = loaded?.index ?? 0;
  const current = queue && index < queue.length ? queue[index] : null;

  const advance = useCallback(() => {
    setLoaded((prev) => prev && { ...prev, index: prev.index + 1 });
    setCorrecting(false);
    setCorrectionValue("");
    setDecidedCount((c) => c + 1);
  }, []);

  // A decision the server refuses is not recorded: the reason is shown, the item stays on
  // screen and the count does not move. One decision at a time, so a held key cannot submit
  // the same item twice.
  const decide = useCallback(
    async (kind: ReviewDecisionKind, correctedGloss: string | null) => {
      if (!current || !user || busy) return;
      await run(current.queue_id, () => submitReviewDecision(current.queue_id, kind, correctedGloss, user.username), advance);
    },
    [current, user, busy, run, advance]
  );

  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const tag = (document.activeElement?.tagName ?? "").toLowerCase();
      if (correcting) {
        if (e.key === "Escape") {
          setCorrecting(false);
          setCorrectionValue("");
        }
        if (e.key === "Enter" && tag === "input") {
          e.preventDefault();
          void decide("correct", correctionValue.trim());
        }
        return;
      }
      if (tag === "input" || tag === "textarea" || tag === "select") return;
      if (!current) return;
      if (e.key.toLowerCase() === "c") {
        void decide("confirm", null);
      } else if (e.key.toLowerCase() === "x") {
        setCorrecting(true);
        setCorrectionValue(current.correction ?? "");
        setTimeout(() => correctionInputRef.current?.focus(), 0);
      } else if (e.key.toLowerCase() === "f") {
        void decide("flag_linguist", null);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [correcting, correctionValue, current, decide]);

  const remaining = useMemo(() => (queue ? Math.max(queue.length - index, 0) : 0), [queue, index]);

  return (
    <section className="review-workspace">
      <header className="view-header">
        <h1>Review workspace (FR-W3)</h1>
        <LanguageSelect selection={selection} />
        <SourceTag source={source} label="queue" />
      </header>
      <p className="review-progress">
        {decidedCount} decided this session · {remaining} remaining in {language ? `the ${language.toUpperCase()} queue` : "queue"}
      </p>
      <WriteStatus write={write} />
      {language && !queue && <p>Loading…</p>}
      {queue && !current && <p>Queue empty — nothing left to review for {language?.toUpperCase()}.</p>}
      {current && (
        <div className="review-card">
          <div className="review-meta">
            <span className="priority-reason">{current.priority_reason.replace(/_/g, " ")}</span>
            <span dir="ltr">{current.kiosk_id}</span>
            <span>{current.language.toUpperCase()}</span>
            {current.signer_id && <span dir="ltr">signer: {current.signer_id}</span>}
          </div>
          <div className="review-prediction">
            <span className="predicted-gloss">{current.predicted_gloss}</span>
            <ConfidenceBadge confidence={current.confidence} />
          </div>
          {current.correction && (
            <p className="weak-label-note">
              Visitor selected a different candidate on the kiosk: <strong>{current.correction}</strong> (weak label —
              not yet verified).
            </p>
          )}
          {correcting && (
            <div className="correction-field">
              <label>
                Corrected gloss
                <input
                  ref={correctionInputRef}
                  value={correctionValue}
                  onChange={(e) => setCorrectionValue(e.target.value)}
                  placeholder="type the correct sign id, then Enter"
                />
              </label>
            </div>
          )}
          <div className="review-actions">
            <button disabled={busy !== null} onClick={() => void decide("confirm", null)}>
              <kbd>C</kbd> Confirm
            </button>
            <button
              disabled={busy !== null}
              onClick={() => {
                setCorrecting(true);
                setTimeout(() => correctionInputRef.current?.focus(), 0);
              }}
            >
              <kbd>X</kbd> Correct
            </button>
            <button disabled={busy !== null} onClick={() => void decide("flag_linguist", null)}>
              <kbd>F</kbd> Flag for linguist
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
