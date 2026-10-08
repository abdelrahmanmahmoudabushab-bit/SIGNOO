import { beforeEach, describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReviewWorkspace } from "./ReviewWorkspace";
import { renderWithProviders } from "../test/utils";
import { mockReviewQueue } from "../api/mock";

// Decisions require a signed-in reviewer (the decision records who made it), so seed a stored
// session the same way AuthContext reads one back after a real login.
beforeEach(() => {
  localStorage.setItem("signoo_console_token", "mock-token-reviewer");
  localStorage.setItem("signoo_console_user", JSON.stringify({ username: "reviewer", role: "reviewer" }));
});

describe("ReviewWorkspace (FR-W3)", () => {
  it("shows the first queue item and advances to the next on the 'c' (confirm) keystroke", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewWorkspace />);

    expect(await screen.findByText(mockReviewQueue[0].predicted_gloss)).toBeInTheDocument();

    await user.keyboard("c");

    expect(await screen.findByText(mockReviewQueue[1].predicted_gloss)).toBeInTheDocument();
    expect(screen.getByText(/1 decided this session/i)).toBeInTheDocument();
  });

  it("opens a correction field on 'x' without needing the mouse", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ReviewWorkspace />);
    await screen.findByText(mockReviewQueue[0].predicted_gloss);

    await user.keyboard("x");

    expect(screen.getByLabelText(/corrected gloss/i)).toBeInTheDocument();
  });
});
