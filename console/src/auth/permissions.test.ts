import { describe, expect, it } from "vitest";
import { can } from "./permissions";

describe("permissions matrix (FR-W9 least privilege)", () => {
  it("lets an operator view the fleet but not decide reviews", () => {
    expect(can("operator", "fleet:view")).toBe(true);
    expect(can("operator", "review:decide")).toBe(false);
  });

  it("lets a reviewer decide reviews but not promote models", () => {
    expect(can("reviewer", "review:decide")).toBe(true);
    expect(can("reviewer", "registry:promote")).toBe(false);
  });

  it("lets an engineer promote models and view training, but not decide reviews", () => {
    expect(can("engineer", "registry:promote")).toBe(true);
    expect(can("engineer", "training:view")).toBe(true);
    expect(can("engineer", "review:decide")).toBe(false);
  });

  it("gives admin every permission checked here", () => {
    expect(can("admin", "fleet:view")).toBe(true);
    expect(can("admin", "review:decide")).toBe(true);
    expect(can("admin", "registry:promote")).toBe(true);
    expect(can("admin", "kb:edit")).toBe(true);
  });

  it("denies an unauthenticated (null) role everything", () => {
    expect(can(null, "fleet:view")).toBe(false);
    expect(can(undefined, "logs:view")).toBe(false);
  });
});
