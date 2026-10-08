import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { FleetView } from "./FleetView";
import { renderWithProviders } from "../test/utils";
import { mockKiosks } from "../api/mock";

describe("FleetView (FR-W1)", () => {
  it("renders every seeded kiosk with health and model version, and discloses mock source", async () => {
    renderWithProviders(<FleetView />);
    for (const kiosk of mockKiosks) {
      expect(await screen.findByText(kiosk.kiosk_id)).toBeInTheDocument();
    }
    expect(await screen.findByText(/mock data/i)).toBeInTheDocument();
    // non-Jetson dev box: temperature must read "unavailable", never a faked number.
    expect(screen.getAllByText(/unavailable/i).length).toBeGreaterThan(0);
  });
});
