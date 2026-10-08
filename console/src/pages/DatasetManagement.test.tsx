import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { DatasetManagement } from "./DatasetManagement";
import { renderWithProviders } from "../test/utils";

describe("DatasetManagement (FR-W4)", () => {
  it("renders coverage gaps and recorded sessions", async () => {
    renderWithProviders(<DatasetManagement />);
    expect((await screen.findAllByText(/more signers needed|target met/i)).length).toBeGreaterThan(0);
    expect((await screen.findAllByText("Retire")).length).toBeGreaterThan(0);
  });
});
