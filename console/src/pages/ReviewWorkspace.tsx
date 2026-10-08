import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { fetchReviewQueue, submitReviewDecision } from "../api/endpoints";
import type { ReviewDecisionKind, ReviewQueueItem } from "../api/types";
import { ConfidenceBadge } from "../components/ConfidenceBadge";
import { SourceTag } from "../components/SourceTag";

// FR-W3: THE workspace the whole learning loop depends on. Keyboard-driven, one decision per
// screen, auto-advance — reviewer time is the scarcest resource in the system (ROADMAP.md R3),
// so every decision must be reachable without touching the mouse.
//
// Keys: C = confirm · X = correct (opens a text field, Enter submits) · F = flag for linguist
// · Esc cancels an open correction field.
export function ReviewWorkspace() {
  const { user } = useAuth();
  const [queue, setQueue] = useState<ReviewQueueItem[] | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [index, setIndex] = useState(0);
  const [correcting, setCorrecting] = useState(false);
  const [correctionValue, setCorrectionValue] = useState("");
  const [decidedCount, setDecidedCount] = useState(0);
  const correctionInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    fetchReviewQueue().then(({ data, source: src }) => {
      if (!cancelled) {
        setQueue(data);
        setSource(src);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const current = queue && index < queue.length ? queue[index] : null;

  const advance = useCallback(() => {
    setIndex((i) => i + 1);
    setCorrecting(false);
    setCorrectionValue("");
    setDecidedCount((c) => c + 1);
  }, []);

  const decide = useCallback(
    async (kind: ReviewDecisionKind, correctedGloss: string | null) => {
      if (!current || !user) return;
      await submitReviewDecision(current.queue_id, kind, correctedGloss, user.username);
      advance();
    },
    [current, user, advance]
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
      if (tag === "input" || tag === "textarea") return;
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
        <SourceTag source={source} />
      </header>
      <p className="review-progress">
        {decidedCount} decided this session · {remaining} remaining in queue
      </p>
      {!queue && <p>Loading…</p>}
      {queue && !current && <p>Queue empty — nothing left to review.</p>}
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
            <button onClick={() => void decide("confirm", null)}>
              <kbd>C</kbd> Confirm
            </button>
            <button
              onClick={() => {
                setCorrecting(true);
                setTimeout(() => correctionInputRef.current?.focus(), 0);
              }}
            >
              <kbd>X</kbd> Correct
            </button>
            <button onClick={() => void decide("flag_linguist", null)}>
              <kbd>F</kbd> Flag for linguist
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
