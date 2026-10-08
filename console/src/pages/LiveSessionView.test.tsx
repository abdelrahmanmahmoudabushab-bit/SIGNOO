import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { LiveSessionView } from "./LiveSessionView";
import { renderWithProviders } from "../test/utils";

describe("LiveSessionView (FR-W2)", () => {
  it("shows the recognition stream and a disabled, explained camera-imagery control", async () => {
    renderWithProviders(<LiveSessionView />);
    expect(await screen.findByText("marhaba")).toBeInTheDocument();
    const cameraButton = await screen.findByRole("button", { name: /camera imagery/i });
    expect(cameraButton).toBeDisabled();
    expect(cameraButton.title).toMatch(/elevated authorisation/i);
  });
});
