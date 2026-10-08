import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import { KnowledgeBaseEditor } from "./KnowledgeBaseEditor";
import { renderWithProviders } from "../test/utils";

describe("KnowledgeBaseEditor (FR-W7)", () => {
  it("renders the unrenderable-word note as informative, not alarming, since every clip is currently unrecorded", async () => {
    renderWithProviders(<KnowledgeBaseEditor />);
    expect((await screen.findAllByText(/clip not yet recorded/i)).length).toBeGreaterThan(0);
    expect(await screen.findByText(/expected before phase 6/i)).toBeInTheDocument();
  });
});
