import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { ModelRegistry } from "./ModelRegistry";
import { renderWithProviders } from "../test/utils";

describe("ModelRegistry (FR-W6)", () => {
  it("lists registry entries with their gate result and status", async () => {
    renderWithProviders(<ModelRegistry />);
    expect(await screen.findByText("promoted")).toBeInTheDocument();
    expect(await screen.findByText("rejected")).toBeInTheDocument();
  });
});
