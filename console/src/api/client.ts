// Thin API client. Pointing the console at the real server/ (B16 Console API) later is meant
// to be a one-line change: set VITE_API_BASE_URL. Until that endpoint answers, every function
// in endpoints.ts falls back to the seeded mock data in mock.ts, and callers get told which
// source answered via the `source` field on results where that matters for the UI to disclose.
const RAW_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
export const API_BASE_URL = RAW_BASE.replace(/\/$/, "");

// Force mock mode regardless of server reachability (useful for demos/tests).
export const FORCE_MOCKS = import.meta.env.VITE_USE_MOCKS === "1";

export function getToken(): string | null {
  try {
    return localStorage.getItem("signoo_console_token");
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem("signoo_console_token", token);
    else localStorage.removeItem("signoo_console_token");
  } catch {
    // localStorage unavailable (e.g. private mode) — session simply won't persist across reload
  }
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_BASE_URL || FORCE_MOCKS) {
    throw new ApiError("no API base URL configured / mocks forced", 0);
  }
  const token = getToken();
  const headers = new Headers(init?.headers);
  headers.set("Content-Type", "application/json");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });
  if (!res.ok) {
    throw new ApiError(`${path} -> ${res.status}`, res.status);
  }
  return (await res.json()) as T;
}

/** True once we've confirmed the real server answered rather than a mock fallback. */
export type Sourced<T> = { data: T; source: "server" | "mock" };
