import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { LoginPage } from "./LoginPage";
import { renderWithProviders } from "../test/utils";

beforeEach(() => {
  localStorage.clear();
});

describe("LoginPage", () => {
  it("logs a seeded mock user in and the login form is replaced by a redirect", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoginPage />);

    await user.type(screen.getByLabelText(/username/i), "reviewer");
    await user.type(screen.getByLabelText(/password/i), "reviewer");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    // On success LoginPage itself renders <Navigate> instead of the form (see component: it
    // checks `if (user) return <Navigate .../>`), so the form fields disappear.
    await waitFor(() => expect(screen.queryByLabelText(/username/i)).not.toBeInTheDocument());
  });

  it("shows an error for bad credentials instead of silently failing", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoginPage />);

    await user.type(screen.getByLabelText(/username/i), "nobody");
    await user.type(screen.getByLabelText(/password/i), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/invalid credentials/i);
  });
});
