import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";

// Auth state lives in localStorage and jsdom's localStorage persists across tests within the
// same file — clear it so one test's login doesn't leak into the next.
afterEach(() => {
  localStorage.clear();
});
