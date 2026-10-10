import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// client.ts reads VITE_API_BASE_URL once at import, so each test sets the env and
// re-imports the modules to get a client that believes a real server is configured.
async function withServer(fetchImpl: typeof fetch) {
  vi.resetModules();
  vi.stubEnv("VITE_API_BASE_URL", "http://server.test");
  vi.stubEnv("VITE_USE_MOCKS", "0");
  vi.stubGlobal("fetch", fetchImpl);
  return import("./endpoints");
}

const reply = (status: number, body: unknown) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));

describe("a real server's answer is never replaced by mock data", () => {
  beforeEach(() => vi.unstubAllEnvs());
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it("surfaces the promotion gate's refusal instead of showing 'promoted'", async () => {
    const { promoteModel } = await withServer(() => reply(409, { detail: "gate did not pass; champion unchanged" }));
    await expect(promoteModel("v1")).rejects.toThrow("gate did not pass; champion unchanged");
  });

  it("still falls back to mock data when the server cannot be reached", async () => {
    const { promoteModel } = await withServer(() => Promise.reject(new TypeError("network down")));
    const result = await promoteModel("v1");
    expect(result.source).toBe("mock");
  });

  it("still falls back for reads, where the page discloses the mock source", async () => {
    const { fetchModelRegistry } = await withServer(() => reply(500, { detail: "boom" }));
    const result = await fetchModelRegistry();
    expect(result.source).toBe("mock");
  });
});
