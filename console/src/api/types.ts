// Types mirroring docs/API_CONTRACTS.md exactly. Keep this file in sync with that document —
// it is the single integration point shared with server/ (built by a different agent against
// the same shapes).

export type Language = "liu" | "esl" | "asl";

export type Role = "operator" | "reviewer" | "engineer" | "admin";

// --- API_CONTRACTS.md §1: vocabulary -------------------------------------------------------
export interface VocabEntry {
  id: string;
  ar: string;
  en: string;
}

// --- API_CONTRACTS.md §2: clip library manifest --------------------------------------------
export interface ClipLibraryEntry {
  clip_path: string | null;
  recorded_by: string | null;
  recorded_at: string | null;
}
export type ClipLibrary = Record<string, ClipLibraryEntry>;

// --- API_CONTRACTS.md §3: knowledge base entry ----------------------------------------------
export interface KBEntry {
  id: string;
  site_id: string;
  keywords: string[];
  answer_ar: string;
  answer_en: string;
  updated_at: string;
}

// --- API_CONTRACTS.md §4: session upload (kiosk B10 -> server B11) -------------------------
export interface SessionItem {
  utterance_id: string;
  landmarks: number[][];
  predicted_gloss: string;
  confidence: number;
  correction: string | null;
}

export interface SessionUpload {
  session_id: string;
  kiosk_id: string;
  language: Language;
  consent_retention: boolean;
  started_at: string;
  ended_at: string | null;
  items: SessionItem[];
}

// --- API_CONTRACTS.md §6: model registry entry ----------------------------------------------
export type GateCriterionVerdict = "pass" | "fail";

export interface ModelRegistryEntry {
  version: string;
  code_commit: string;
  data_snapshot: string;
  metrics: {
    top1: number;
    top5: number;
    per_signer: Record<string, number>;
    per_sign: Record<string, number>;
  };
  gate: {
    passed: boolean;
    criteria: Record<string, GateCriterionVerdict>;
  };
  status: "candidate" | "shadow" | "promoted" | "rejected" | "rolled_back";
}

// The 7 DR-6 gate criteria, in the fixed order the promotion gate evaluates them.
// Used to render every criterion explicitly (FR-W5), not collapse to a pass/fail badge.
export const GATE_CRITERIA: { key: string; label: string }[] = [
  { key: "top1_improves_or_flat", label: "Top-1 accuracy improves, or within 0.5pt while another criterion improves" },
  { key: "no_signer_regression", label: "No signer regresses by more than 3 points" },
  { key: "no_sign_regression", label: "No sign regresses by more than 5 points" },
  { key: "no_zero_recall_class", label: "No class falls to zero recall" },
  { key: "false_accept_rate", label: "False-accept rate does not increase" },
  { key: "latency_budget", label: "Latency stays within NFR-P1 on device" },
  { key: "shadow_run_clean", label: "Completed shadow-run without live-metric degradation" },
];

// --- Fleet (FR-W1) — shape not fixed by API_CONTRACTS.md yet; kept small and obvious so
// wiring the real server/B15 endpoint later is a type-compatible drop-in. -------------------
export type KioskHealth = "online" | "degraded" | "offline";

export interface KioskSummary {
  kiosk_id: string;
  site_id: string;
  language: Language;
  health: KioskHealth;
  model_version: string;
  uptime_seconds: number;
  temperature_c: number | null; // null = "unavailable" (non-Jetson dev box), never a faked number
  last_contact: string;
}

// The real server/routers/fleet.py `KioskOut` wire shape (server/schemas.py) — field names
// differ from the KioskSummary view-model above (current_model_version vs model_version,
// last_heartbeat_at vs last_contact, uptime_s vs uptime_seconds, plus an explicit `online`
// flag instead of a derived health enum). `endpoints.ts`'s `adaptKioskOut` is the one place
// that reconciles this; nothing else should read this type directly.
export interface ServerKioskOut {
  kiosk_id: string;
  site_id: string;
  language: Language;
  current_model_version: string | null;
  last_heartbeat_at: string | null;
  temperature_c: number | null;
  throttled: boolean;
  uptime_s: number | null;
  online: boolean;
}

// The real server/routers/auth.py `TokenOut` wire shape — no `user` object, since the server
// only knows the role from the JWT, not a username. `endpoints.ts`'s `login()` reconciles this
// into the `LoginResponse` shape the rest of the console expects.
export interface ServerTokenOut {
  access_token: string;
  token_type: string;
  role: Role;
}

// --- Live session (FR-W2) -------------------------------------------------------------------
export interface LiveRecognitionEvent {
  seq: number;
  gloss: string;
  confidence: number;
  timestamp: string;
}

export interface LiveSession {
  session_id: string;
  kiosk_id: string;
  language: Language;
  started_at: string;
  status: "active" | "ended";
  events: LiveRecognitionEvent[];
}

// --- Review queue (FR-W3 / B11) --------------------------------------------------------------
export type ReviewDecisionKind = "confirm" | "correct" | "flag_linguist";

export interface ReviewQueueItem {
  queue_id: string;
  session_id: string;
  utterance_id: string;
  kiosk_id: string;
  language: Language;
  predicted_gloss: string;
  confidence: number;
  correction: string | null; // weak label from the kiosk uncertainty UI, if any
  signer_id: string | null;
  captured_at: string;
  priority_reason: "low_confidence" | "corrected" | "under_represented_sign" | "routine";
}

export interface ReviewDecision {
  queue_id: string;
  decision: ReviewDecisionKind;
  corrected_gloss: string | null;
  reviewer: string;
  decided_at: string;
}

// server/schemas.py ReviewItemOut — the wire shape from GET /review/queue and
// POST /review/{id}/decision. `endpoints.ts` adapts this into ReviewQueueItem.
export interface ServerReviewItemOut {
  id: string;
  session_id: string;
  kiosk_id: string;
  utterance_id: string;
  language: Language;
  signer_id: string;
  predicted_gloss: string;
  confidence: number;
  correction: string | null;
  status: string;
  validated_label: string | null;
  captured_at: string;
  priority_score: number;
  priority_reason: "low_confidence" | "corrected" | "under_represented_sign" | "routine";
}

// --- Dataset management (FR-W4) --------------------------------------------------------------
export interface SignCoverage {
  sign_id: string;
  language: Language;
  signer_count: number;
  clip_count: number;
  target_signers: number;
  target_clips: number;
}

export interface DatasetSession {
  session_id: string;
  language: Language;
  signer_id: string;
  recorded_at: string;
  clip_count: number;
  status: "active" | "retired";
}

// --- Training visibility (FR-W5) -------------------------------------------------------------
export interface TrainingRun {
  run_id: string;
  started_at: string;
  finished_at: string | null;
  status: "running" | "completed" | "failed";
  candidate_version: string | null;
  champion_version: string | null;
  curves: { epoch: number; train_loss: number; val_accuracy: number }[];
}

// --- Logs / alerts (FR-W8) --------------------------------------------------------------------
export interface LogEntry {
  id: string;
  timestamp: string;
  level: "debug" | "info" | "warn" | "error";
  correlation_id: string | null;
  source: string;
  message: string;
}

export type AlertKind = "kiosk_offline" | "thermal_throttling" | "accuracy_drop" | "review_queue_backlog";

export interface AlertEntry {
  id: string;
  kind: AlertKind;
  kiosk_id: string | null;
  severity: "info" | "warning" | "critical";
  message: string;
  raised_at: string;
  acknowledged: boolean;
}

// --- Auth --------------------------------------------------------------------------------------
export interface AuthUser {
  username: string;
  role: Role;
}

export interface LoginResponse {
  token: string;
  user: AuthUser;
}
