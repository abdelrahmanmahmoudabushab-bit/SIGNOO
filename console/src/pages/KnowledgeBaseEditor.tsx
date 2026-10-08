import { useEffect, useMemo, useState } from "react";
import { fetchClipLibrary, fetchKB, saveKBEntry } from "../api/endpoints";
import type { ClipLibrary, KBEntry } from "../api/types";
import { can } from "../auth/permissions";
import { useAuth } from "../auth/AuthContext";
import { SourceTag } from "../components/SourceTag";

// Splits an answer into rough "words" for the unrenderable-word check. Good enough for a
// preview warning; the real word-boundary/lemma matching against core/vocab.json belongs on
// the server (B14) and can replace this once that endpoint exists.
function wordsOf(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

/**
 * FR-W7: preview the signed rendering, with a warning for any word the clip library doesn't
 * have yet. Per core/clip_library.json today, that's EVERY word — so this must read as the
 * normal, informative state (a calm list of "not yet recorded" placeholders), not an alarming
 * wall of red. Compare against how the kiosk's own B8 treats this same fact (see
 * docs/SYSTEM_DESIGN.md B8): a clearly-marked placeholder per word, never a crash.
 */
function SignedPreview({ text, clipLibrary }: { text: string; clipLibrary: ClipLibrary }) {
  const words = wordsOf(text);
  const unrecorded = words.filter((w) => !clipLibrary[w] || clipLibrary[w].clip_path === null);
  return (
    <div className="signed-preview">
      <div className="clip-strip">
        {words.map((w, i) => {
          const known = clipLibrary[w];
          const recorded = known && known.clip_path !== null;
          return (
            <span key={`${w}-${i}`} className={recorded ? "clip-known" : "clip-placeholder"}>
              {w}
              {!recorded && <em> (clip not yet recorded)</em>}
            </span>
          );
        })}
        {words.length === 0 && <span className="clip-empty">Nothing to preview yet.</span>}
      </div>
      {unrecorded.length > 0 && (
        <p className="clip-library-note">
          {unrecorded.length} of {words.length} word(s) have no recorded clip yet — this is expected before
          Phase 6's recording pass (docs/ROADMAP.md), not an error. They will be fingerspelled or flagged on the
          kiosk (FR-A3).
        </p>
      )}
    </div>
  );
}

export function KnowledgeBaseEditor() {
  const { user } = useAuth();
  const [entries, setEntries] = useState<KBEntry[] | null>(null);
  const [clipLibrary, setClipLibrary] = useState<ClipLibrary>({});
  const [source, setSource] = useState<"server" | "mock" | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<KBEntry | null>(null);
  const [saved, setSaved] = useState(false);

  const canEdit = can(user?.role, "kb:edit");

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchKB(), fetchClipLibrary()]).then(([kb, clips]) => {
      if (cancelled) return;
      setEntries(kb.data);
      setClipLibrary(clips.data);
      setSource(kb.source === "server" && clips.source === "server" ? "server" : "mock");
      if (kb.data.length > 0) setSelectedId(kb.data[0].id);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const found = entries?.find((e) => e.id === selectedId) ?? null;
    setDraft(found ? { ...found } : null);
    setSaved(false);
  }, [selectedId, entries]);

  const previewText = useMemo(() => (draft ? `${draft.answer_ar} ${draft.answer_en}` : ""), [draft]);

  async function onSave() {
    if (!draft) return;
    const updated = { ...draft, updated_at: new Date().toISOString() };
    await saveKBEntry(updated);
    setEntries((prev) => prev?.map((e) => (e.id === updated.id ? updated : e)) ?? null);
    setSaved(true);
  }

  return (
    <section>
      <header className="view-header">
        <h1>Knowledge base (FR-W7)</h1>
        <SourceTag source={source} />
      </header>
      {!entries && <p>Loading…</p>}
      {entries && (
        <div className="kb-layout">
          <ul className="kb-list">
            {entries.map((e) => (
              <li key={e.id}>
                <button className={e.id === selectedId ? "active" : undefined} onClick={() => setSelectedId(e.id)}>
                  {e.answer_en.slice(0, 40)}
                </button>
              </li>
            ))}
          </ul>
          {draft && (
            <div className="kb-editor">
              <label>
                Keywords (comma-separated)
                <input
                  disabled={!canEdit}
                  value={draft.keywords.join(", ")}
                  onChange={(e) => setDraft({ ...draft, keywords: e.target.value.split(",").map((k) => k.trim()).filter(Boolean) })}
                />
              </label>
              <label>
                Answer (Arabic)
                <textarea
                  dir="rtl"
                  disabled={!canEdit}
                  value={draft.answer_ar}
                  onChange={(e) => setDraft({ ...draft, answer_ar: e.target.value })}
                />
              </label>
              <label>
                Answer (English)
                <textarea
                  disabled={!canEdit}
                  value={draft.answer_en}
                  onChange={(e) => setDraft({ ...draft, answer_en: e.target.value })}
                />
              </label>
              {canEdit ? (
                <button onClick={onSave}>Save</button>
              ) : (
                <p className="hint">Your role ({user?.role}) can view but not edit knowledge-base entries.</p>
              )}
              {saved && <p className="saved-note">Saved.</p>}

              <h3>Signed rendering preview</h3>
              <SignedPreview text={previewText} clipLibrary={clipLibrary} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
