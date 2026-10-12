// One function per operation the console needs. Every function tries the real server first
// (if VITE_API_BASE_URL is set and mocks aren't forced) and falls back to seeded mock data on
// any failure, tagging the result with `source` so views can disclose which one answered.
// This is the ONLY module (besides client.ts) that should change when server/'s real routes
// land — no view component talks to fetch() directly.
import { ApiError, apiFetch, type Sourced } from "./client";
import * as mock from "./mock";
import type {
  AlertEntry,
  AuthUser,
  DatasetSession,
  KBEntry,
  KioskSummary,
  Language,
  LiveSession,
  LogEntry,
  LoginResponse,
  ModelRegistryEntry,
  ReviewDecision,
  ReviewDecisionKind,
  ReviewQueueItem,
  ServerKioskOut,
  ServerReviewItemOut,
  ServerTokenOut,
  SignCoverage,
  TrainingRun,
} from "./types";

// A real server answered and said no (as opposed to no server being configured or reachable,
// which apiFetch reports as status 0 or a network error).
function isServerRefusal(err: unknown): boolean {
  return err instanceof ApiError && err.status >= 400;
}

async function withMockFallback<T>(path: string, init: RequestInit | undefined, fallback: () => T): Promise<Sourced<T>> {
  try {
    const data = await apiFetch<T>(path, init);
    return { data, source: "server" };
  } catch (err) {
    // A real server that answered and refused a write (the promotion gate's 409, a 403 for
    // a role) must surface as a failure. Falling back to the mock's success here would tell
    // an operator a model was promoted when the server said no. Reads, and any failure to
    // reach a server at all, still fall back to mock data.
    const isWrite = (init?.method ?? "GET").toUpperCase() !== "GET";
    if (isWrite && isServerRefusal(err)) throw err;
    return { data: fallback(), source: "mock" };
  }
}

// Every language is its own engine: its own models, review queue and training runs. These two
// keep a per-language request per-language even if a server ignored the query parameter.
function languageQuery(language?: Language): string {
  return language ? `?language=${encodeURIComponent(language)}` : "";
}

function inLanguage<T extends { language: Language }>(rows: T[], language?: Language): T[] {
  return language ? rows.filter((r) => r.language === language) : rows;
}

// The real server has no offline-window concept exposed to the client (it already computed
// `online` server-side, per an ONLINE_WINDOW in routers/fleet.py) — "degraded" isn't a state
// the server reports at all yet, so the best honest mapping is online/offline; a kiosk that
// heartbeats but reports `throttled: true` is shown as degraded, since that IS visible signal.
function adaptKioskOut(k: ServerKioskOut): KioskSummary {
  return {
    kiosk_id: k.kiosk_id,
    site_id: k.site_id,
    language: k.language,
    health: !k.online ? "offline" : k.throttled ? "degraded" : "online",
    model_version: k.current_model_version ?? "(none deployed)",
    model_versions: k.model_versions ?? {},
    uptime_seconds: k.uptime_s ?? 0,
    temperature_c: k.temperature_c,
    last_contact: k.last_heartbeat_at ?? "",
  };
}

export async function login(username: string, password: string): Promise<Sourced<LoginResponse>> {
  try {
    // server/routers/auth.py expects `email`, not `username`, and returns a bare token + role
    // (no user object — it has no username to give back). The console still wants an
    // `AuthUser` for display, so it's built from what was actually entered and confirmed.
    const token = await apiFetch<ServerTokenOut>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email: username, password }),
    });
    const user: AuthUser = { username, role: token.role };
    return { data: { token: token.access_token, user }, source: "server" };
  } catch (err) {
    // A server that rejected these credentials must not be answered by a demo account of the
    // same name: that would sign someone in with a mock token the server never issued.
    if (isServerRefusal(err)) throw err;
    const user = mock.findMockUser(username, password);
    if (!user) throw new Error("invalid credentials");
    return { data: { token: `mock-token-${user.username}`, user }, source: "mock" };
  }
}

export async function fetchFleet(): Promise<Sourced<KioskSummary[]>> {
  try {
    const raw = await apiFetch<ServerKioskOut[]>("/fleet/kiosks");
    return { data: raw.map(adaptKioskOut), source: "server" };
  } catch {
    return { data: mock.mockKiosks, source: "mock" };
  }
}

export function fetchLiveSession(sessionId: string): Promise<Sourced<LiveSession | null>> {
  return withMockFallback(`/sessions/${sessionId}/live`, undefined, () => mock.mockLiveSessions.find((s) => s.session_id === sessionId) ?? null);
}

export function fetchLiveSessions(): Promise<Sourced<LiveSession[]>> {
  return withMockFallback("/sessions/live", undefined, () => mock.mockLiveSessions);
}

function adaptReviewItem(item: ServerReviewItemOut): ReviewQueueItem {
  return {
    queue_id: item.id,
    session_id: item.session_id,
    utterance_id: item.utterance_id,
    kiosk_id: item.kiosk_id,
    language: item.language,
    predicted_gloss: item.predicted_gloss,
    confidence: item.confidence,
    correction: item.correction,
    signer_id: item.signer_id,
    captured_at: item.captured_at,
    priority_reason: item.priority_reason,
  };
}

export async function fetchReviewQueue(language?: Language): Promise<Sourced<ReviewQueueItem[]>> {
  try {
    const raw = await apiFetch<ServerReviewItemOut[]>(`/review/queue${languageQuery(language)}`);
    return { data: inLanguage(raw.map(adaptReviewItem), language), source: "server" };
  } catch {
    return { data: inLanguage(mock.mockReviewQueue, language), source: "mock" };
  }
}

// server/schemas.py ReviewDecisionIn only accepts "confirm" | "correct" | "flag" (no
// "_linguist" suffix) and has no reviewer/decided_at/queue_id fields — those are set
// server-side from the auth token and the URL path.
const DECISION_TO_SERVER: Record<ReviewDecisionKind, "confirm" | "correct" | "flag"> = {
  confirm: "confirm",
  correct: "correct",
  flag_linguist: "flag",
};

export async function submitReviewDecision(
  queueId: string,
  decision: ReviewDecisionKind,
  correctedGloss: string | null,
  reviewer: string
): Promise<Sourced<ReviewDecision>> {
  const body: ReviewDecision = {
    queue_id: queueId,
    decision,
    corrected_gloss: correctedGloss,
    reviewer,
    decided_at: new Date().toISOString(),
  };
  try {
    await apiFetch<ServerReviewItemOut>(`/review/${queueId}/decision`, {
      method: "POST",
      body: JSON.stringify({
        decision: DECISION_TO_SERVER[decision],
        validated_label: correctedGloss,
      }),
    });
    return { data: body, source: "server" };
  } catch (err) {
    // A refused decision (403 for the role, 422 for a correction with no label, 404 for an
    // item that left the queue) was not recorded; reporting it as decided would lose it.
    if (isServerRefusal(err)) throw err;
    return { data: body, source: "mock" };
  }
}

export function fetchCoverage(): Promise<Sourced<SignCoverage[]>> {
  return withMockFallback("/dataset/coverage", undefined, () => mock.mockCoverage);
}

export function fetchDatasetSessions(): Promise<Sourced<DatasetSession[]>> {
  return withMockFallback("/dataset/sessions", undefined, () => mock.mockDatasetSessions);
}

export function retireSession(sessionId: string): Promise<Sourced<{ session_id: string; status: "retired" }>> {
  return withMockFallback(
    `/dataset/sessions/${sessionId}/retire`,
    { method: "POST" },
    () => ({ session_id: sessionId, status: "retired" as const })
  );
}

// server/ has no GET /training/runs yet (only POST /training/trigger), so against a real
// server this is always the mock list; the page's source tag says so.
export async function fetchTrainingRuns(language?: Language): Promise<Sourced<TrainingRun[]>> {
  const res = await withMockFallback(`/training/runs${languageQuery(language)}`, undefined, () => mock.mockTrainingRuns);
  return { ...res, data: inLanguage(res.data, language) };
}

// server/routers/training.py POST /training/trigger trains a candidate for ONE language and
// returns its registry row. There is deliberately no mock outcome: with no server, nothing was
// trained, and inventing a model row would be a model nobody can load.
export function triggerTraining(language: Language): Promise<Sourced<ModelRegistryEntry | null>> {
  return withMockFallback<ModelRegistryEntry | null>(
    "/training/trigger",
    { method: "POST", body: JSON.stringify({ language }) },
    () => null
  );
}

// server/routers/registry.py GET "" (optional ?language=)
export async function fetchModelRegistry(language?: Language): Promise<Sourced<ModelRegistryEntry[]>> {
  const res = await withMockFallback(`/registry${languageQuery(language)}`, undefined, () => mock.mockRegistry);
  return { ...res, data: inLanguage(res.data, language) };
}

// The languages this console can work on. The server has no "list languages" endpoint, so the
// list is the distinct `language` values in data it does serve: registered kiosks (their
// default language and every language engine they report a model for), registry rows, and
// pending review items. A language with none of those yet is not listed. Mock languages are
// used only when no server answered at all, never mixed into a real server's list.
export async function fetchLanguages(): Promise<Sourced<Language[]>> {
  type Row = { language: Language; model_versions?: Record<Language, string> };
  const distinct = (lists: Row[][]): Language[] => {
    const seen = new Set<Language>();
    for (const row of lists.flat()) {
      if (row.language) seen.add(row.language);
      for (const code of Object.keys(row.model_versions ?? {})) seen.add(code);
    }
    return [...seen];
  };
  const answers = await Promise.allSettled([
    apiFetch<Row[]>("/fleet/kiosks"),
    apiFetch<Row[]>("/registry"),
    apiFetch<Row[]>("/review/queue"),
  ]);
  const fromServer = answers.flatMap((a) => (a.status === "fulfilled" ? [a.value] : []));
  if (fromServer.length > 0) return { data: distinct(fromServer), source: "server" };
  return { data: distinct([mock.mockKiosks, mock.mockRegistry, mock.mockReviewQueue]), source: "mock" };
}

export function promoteModel(version: string): Promise<Sourced<{ version: string; status: string }>> {
  // server/routers/registry.py requires status "shadow" + a passed gate before promoting
  // (409 otherwise). That refusal is rethrown by withMockFallback and shown by
  // ModelRegistry.tsx; it never falls back to the mock outcome below.
  return withMockFallback(
    `/registry/${version}/promote`,
    { method: "POST" },
    () => ({ version, status: "promoted" as const })
  );
}

export function rollbackModel(version: string): Promise<Sourced<{ version: string; status: string }>> {
  return withMockFallback(
    `/registry/${version}/rollback`,
    { method: "POST" },
    () => ({ version, status: "rolled_back" as const })
  );
}

export function completeShadowRun(version: string): Promise<Sourced<{ version: string; status: string }>> {
  return withMockFallback(
    `/registry/${version}/shadow-complete`,
    { method: "POST" },
    () => ({ version, status: "shadow" as const })
  );
}

export function pinModelToKiosk(version: string, kioskId: string): Promise<Sourced<{ version: string; kiosk_id: string }>> {
  // There is no per-kiosk "pin" endpoint — server/routers/fleet.py's rollout targeting
  // (POST /fleet/rollouts) is the real mechanism that moves a kiosk onto a given model
  // version, so "pin to one kiosk" is a rollout whose target list has exactly one id.
  return withMockFallback(
    "/fleet/rollouts",
    { method: "POST", body: JSON.stringify({ model_version: version, target_kiosk_ids: [kioskId] }) },
    () => ({ version, kiosk_id: kioskId })
  );
}

export function fetchKB(): Promise<Sourced<KBEntry[]>> {
  return withMockFallback("/kb", undefined, () => mock.mockKB);
}

export function saveKBEntry(entry: KBEntry): Promise<Sourced<KBEntry>> {
  // server/routers/kb.py's PUT /kb/{id} expects KBEntryIn (site_id, keywords, answer_ar,
  // answer_en — no id/updated_at/unrenderable_words, which are server-computed), so the
  // full KBEntry can't be sent verbatim.
  const { site_id, keywords, answer_ar, answer_en } = entry;
  return withMockFallback(
    `/kb/${entry.id}`,
    { method: "PUT", body: JSON.stringify({ site_id, keywords, answer_ar, answer_en }) },
    () => entry
  );
}

export function fetchLogs(): Promise<Sourced<LogEntry[]>> {
  return withMockFallback("/logs", undefined, () => mock.mockLogs);
}

export function fetchAlerts(): Promise<Sourced<AlertEntry[]>> {
  return withMockFallback("/alerts", undefined, () => mock.mockAlerts);
}

export function acknowledgeAlert(id: string): Promise<Sourced<{ id: string; acknowledged: true }>> {
  return withMockFallback(`/alerts/${id}/ack`, { method: "POST" }, () => ({ id, acknowledged: true as const }));
}

export function fetchClipLibrary() {
  return withMockFallback("/kb/clip-library", undefined, () => mock.mockClipLibrary); // server/routers/kb.py's GET /kb/clip-library
}

export type { AuthUser };
