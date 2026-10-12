// Seeded mock data, used whenever the real server/ API is unreachable (or VITE_USE_MOCKS=1).
// Shapes match docs/API_CONTRACTS.md exactly, so swapping to the real server is a data-source
// change only — no consumer of `endpoints.ts` needs to know which one it's talking to.
import type {
  AlertEntry,
  AuthUser,
  ClipLibrary,
  DatasetSession,
  KBEntry,
  KioskSummary,
  LiveSession,
  LogEntry,
  ModelRegistryEntry,
  ReviewQueueItem,
  SignCoverage,
  TrainingRun,
  VocabEntry,
} from "./types";

export const MOCK_USERS: Record<string, { password: string; user: AuthUser }> = {
  operator: { password: "operator", user: { username: "operator", role: "operator" } },
  reviewer: { password: "reviewer", user: { username: "reviewer", role: "reviewer" } },
  engineer: { password: "engineer", user: { username: "engineer", role: "engineer" } },
  admin: { password: "admin", user: { username: "admin", role: "admin" } },
};

export const mockVocab: VocabEntry[] = [
  { id: "marhaba", ar: "مرحبا", en: "hello" },
  { id: "shukran", ar: "شكرا", en: "thank you" },
  { id: "musaada", ar: "مساعدة", en: "help" },
  { id: "ma_fahimt", ar: "لم أفهم", en: "I don't understand" },
];

// core/clip_library.json is empty of real clips by design (Phase 6 not reached yet) — every
// entry has clip_path: null. That is the normal state (FR-W7), not an error.
export const mockClipLibrary: ClipLibrary = Object.fromEntries(
  mockVocab.map((v) => [v.id, { clip_path: null, recorded_by: null, recorded_at: null }])
);

export const mockKiosks: KioskSummary[] = [
  {
    kiosk_id: "kiosk-dev-01",
    site_id: "default",
    language: "liu",
    health: "online",
    model_version: "2026-09-01T00:00:00Z-a1b2c3d",
    model_versions: { liu: "2026-09-01T00:00:00Z-a1b2c3d" },
    uptime_seconds: 1234567,
    temperature_c: null,
    last_contact: "2026-09-04T09:58:00Z",
  },
  {
    kiosk_id: "kiosk-dev-02",
    site_id: "default",
    language: "liu",
    health: "degraded",
    model_version: "2026-08-29T00:00:00Z-9f8e7d6",
    model_versions: { liu: "2026-08-29T00:00:00Z-9f8e7d6" },
    uptime_seconds: 45210,
    temperature_c: null,
    last_contact: "2026-09-04T08:40:00Z",
  },
  {
    kiosk_id: "kiosk-dev-03",
    site_id: "site-b",
    language: "esl",
    health: "offline",
    model_version: "2026-08-20T00:00:00Z-1234abc",
    model_versions: { esl: "2026-08-20T00:00:00Z-1234abc" },
    uptime_seconds: 0,
    temperature_c: null,
    last_contact: "2026-09-03T14:02:00Z",
  },
];

export const mockLiveSessions: LiveSession[] = [
  {
    session_id: "sess-live-001",
    kiosk_id: "kiosk-dev-01",
    language: "liu",
    started_at: "2026-09-04T09:55:00Z",
    status: "active",
    events: [
      { seq: 1, gloss: "marhaba", confidence: 0.91, timestamp: "2026-09-04T09:55:03Z" },
      { seq: 2, gloss: "musaada", confidence: 0.62, timestamp: "2026-09-04T09:55:05Z" },
      { seq: 3, gloss: "ma_fahimt", confidence: 0.34, timestamp: "2026-09-04T09:55:08Z" },
    ],
  },
];

export const mockReviewQueue: ReviewQueueItem[] = [
  {
    queue_id: "q-001",
    session_id: "sess-001",
    utterance_id: "utt-001",
    kiosk_id: "kiosk-dev-01",
    language: "liu",
    predicted_gloss: "shukran",
    confidence: 0.41,
    correction: null,
    signer_id: "signer-07",
    captured_at: "2026-09-04T08:00:00Z",
    priority_reason: "low_confidence",
  },
  {
    queue_id: "q-002",
    session_id: "sess-002",
    utterance_id: "utt-002",
    kiosk_id: "kiosk-dev-01",
    language: "liu",
    predicted_gloss: "naam",
    confidence: 0.77,
    correction: "la",
    signer_id: "signer-03",
    captured_at: "2026-09-04T08:05:00Z",
    priority_reason: "corrected",
  },
  {
    queue_id: "q-003",
    session_id: "sess-003",
    utterance_id: "utt-003",
    kiosk_id: "kiosk-dev-02",
    language: "liu",
    predicted_gloss: "ayna",
    confidence: 0.55,
    correction: null,
    signer_id: "signer-11",
    captured_at: "2026-09-04T08:10:00Z",
    priority_reason: "under_represented_sign",
  },
  // A second language, so the review workspace can be seen keeping the two queues apart.
  {
    queue_id: "q-004",
    session_id: "sess-004",
    utterance_id: "utt-004",
    kiosk_id: "kiosk-dev-01",
    language: "asl",
    predicted_gloss: "thank_you",
    confidence: 0.48,
    correction: null,
    signer_id: "signer-21",
    captured_at: "2026-09-04T08:15:00Z",
    priority_reason: "low_confidence",
  },
];

export const mockCoverage: SignCoverage[] = mockVocab.map((v, i) => ({
  sign_id: v.id,
  language: "liu",
  signer_count: [2, 8, 1, 4][i % 4],
  clip_count: [6, 32, 3, 15][i % 4],
  target_signers: 25,
  target_clips: 125,
}));

export const mockDatasetSessions: DatasetSession[] = [
  { session_id: "sess-001", language: "liu", signer_id: "signer-07", recorded_at: "2026-09-01T10:00:00Z", clip_count: 12, status: "active" },
  { session_id: "sess-002", language: "liu", signer_id: "signer-03", recorded_at: "2026-09-02T10:00:00Z", clip_count: 8, status: "active" },
  { session_id: "sess-old", language: "liu", signer_id: "signer-01", recorded_at: "2026-06-01T10:00:00Z", clip_count: 4, status: "retired" },
];

export const mockTrainingRuns: TrainingRun[] = [
  {
    run_id: "run-002",
    language: "liu",
    started_at: "2026-09-04T02:00:00Z",
    finished_at: "2026-09-04T03:10:00Z",
    status: "completed",
    candidate_version: "2026-09-04T03:10:00Z-cand02",
    champion_version: "2026-09-01T00:00:00Z-a1b2c3d",
    curves: [
      { epoch: 1, train_loss: 1.8, val_accuracy: 0.42 },
      { epoch: 2, train_loss: 1.1, val_accuracy: 0.58 },
      { epoch: 3, train_loss: 0.7, val_accuracy: 0.67 },
    ],
  },
  {
    run_id: "run-001",
    language: "liu",
    started_at: "2026-09-01T02:00:00Z",
    finished_at: "2026-09-01T03:05:00Z",
    status: "completed",
    candidate_version: "2026-09-01T00:00:00Z-a1b2c3d",
    champion_version: null,
    curves: [
      { epoch: 1, train_loss: 2.0, val_accuracy: 0.35 },
      { epoch: 2, train_loss: 1.3, val_accuracy: 0.5 },
      { epoch: 3, train_loss: 0.9, val_accuracy: 0.6 },
    ],
  },
];

export const mockRegistry: ModelRegistryEntry[] = [
  {
    version: "2026-09-01T00:00:00Z-a1b2c3d",
    language: "liu",
    code_commit: "a1b2c3d",
    data_snapshot: "manifest-sha256:aaaa",
    metrics: { top1: 0.6, top5: 0.85, per_signer: { "signer-01": 0.55 }, per_sign: { marhaba: 0.7 } },
    gate: {
      passed: true,
      criteria: {
        top1_improves_or_flat: "pass",
        no_signer_regression: "pass",
        no_sign_regression: "pass",
        no_zero_recall_class: "pass",
        false_accept_rate: "pass",
        latency_budget: "pass",
        shadow_run_clean: "pass",
      },
    },
    status: "promoted",
  },
  {
    version: "2026-09-04T03:10:00Z-cand02",
    language: "liu",
    code_commit: "cand02",
    data_snapshot: "manifest-sha256:bbbb",
    metrics: { top1: 0.58, top5: 0.83, per_signer: { "signer-01": 0.4 }, per_sign: { marhaba: 0.62 } },
    gate: {
      passed: false,
      criteria: {
        top1_improves_or_flat: "fail",
        no_signer_regression: "fail",
        no_sign_regression: "pass",
        no_zero_recall_class: "pass",
        false_accept_rate: "pass",
        latency_budget: "pass",
        shadow_run_clean: "fail",
      },
    },
    status: "rejected",
  },
  // A second language with its own champion, so the registry can be seen keeping them apart.
  {
    version: "2026-09-02T00:00:00Z-asl0001",
    language: "asl",
    code_commit: "asl0001",
    data_snapshot: "manifest-sha256:cccc",
    metrics: { top1: 0.71, top5: 0.9, per_signer: { "signer-21": 0.68 }, per_sign: { thank_you: 0.74 } },
    gate: {
      passed: true,
      criteria: {
        top1_improves_or_flat: "pass",
        no_signer_regression: "pass",
        no_sign_regression: "pass",
        no_zero_recall_class: "pass",
        false_accept_rate: "pass",
        latency_budget: "pass",
        shadow_run_clean: "pass",
      },
    },
    status: "promoted",
  },
];

export const mockKB: KBEntry[] = [
  {
    id: "kb_001",
    site_id: "default",
    keywords: ["افتتاح", "hours", "opening"],
    answer_ar: "نفتح من التاسعة صباحًا حتى الخامسة مساءً.",
    answer_en: "We are open from 9am to 5pm.",
    updated_at: "2026-09-01T00:00:00Z",
  },
  {
    id: "kb_002",
    site_id: "default",
    keywords: ["مساعدة", "help", "musaada"],
    answer_ar: "يمكنك سؤال أحد الموظفين عند مكتب الاستقبال.",
    answer_en: "You can ask a staff member at the front desk.",
    updated_at: "2026-09-02T00:00:00Z",
  },
];

export const mockLogs: LogEntry[] = [
  { id: "l1", timestamp: "2026-09-04T09:00:00Z", level: "info", correlation_id: "corr-1", source: "kiosk-dev-01", message: "session started" },
  { id: "l2", timestamp: "2026-09-04T09:01:00Z", level: "warn", correlation_id: "corr-1", source: "kiosk-dev-01", message: "low confidence recognition (0.34)" },
  { id: "l3", timestamp: "2026-09-04T08:00:00Z", level: "error", correlation_id: "corr-2", source: "kiosk-dev-03", message: "heartbeat missed" },
];

export const mockAlerts: AlertEntry[] = [
  { id: "a1", kind: "kiosk_offline", kiosk_id: "kiosk-dev-03", severity: "critical", message: "kiosk-dev-03 has not reported in 18h", raised_at: "2026-09-04T08:10:00Z", acknowledged: false },
  { id: "a2", kind: "review_queue_backlog", kiosk_id: null, severity: "warning", message: "Review queue exceeds 200 items", raised_at: "2026-09-04T07:00:00Z", acknowledged: false },
  { id: "a3", kind: "thermal_throttling", kiosk_id: "kiosk-dev-02", severity: "warning", message: "kiosk-dev-02 reported degraded state", raised_at: "2026-09-04T06:30:00Z", acknowledged: true },
];

export function findMockUser(username: string, password: string): AuthUser | null {
  const entry = MOCK_USERS[username];
  if (!entry || entry.password !== password) return null;
  return entry.user;
}
