import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import App from "./App";

describe("App", () => {
  it("redirects an unauthenticated visitor to the login screen", async () => {
    render(
      <MemoryRouter initialEntries={["/fleet"]}>
        <App />
      </MemoryRouter>
    );
    expect(await screen.findByRole("heading", { name: /signoo console/i })).toBeInTheDocument();
  });
});
