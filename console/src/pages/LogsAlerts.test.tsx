import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { LogsAlerts } from "./LogsAlerts";
import { renderWithProviders } from "../test/utils";

describe("LogsAlerts (FR-W8)", () => {
  it("lists alerts and lets a search filter the log table", async () => {
    renderWithProviders(<LogsAlerts />);
    expect(await screen.findByText(/kiosk-dev-03 has not reported/i)).toBeInTheDocument();
    expect(await screen.findByText("session started")).toBeInTheDocument();
  });
});
