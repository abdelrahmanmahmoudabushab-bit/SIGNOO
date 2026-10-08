import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { TrainingVisibility } from "./TrainingVisibility";
import { renderWithProviders } from "../test/utils";
import { GATE_CRITERIA } from "../api/types";

describe("TrainingVisibility (FR-W5)", () => {
  it("renders every one of the 7 DR-6 gate criteria explicitly, not a single pass/fail badge", async () => {
    renderWithProviders(<TrainingVisibility />);
    for (const c of GATE_CRITERIA) {
      expect(await screen.findByText(c.label)).toBeInTheDocument();
    }
  });
});
