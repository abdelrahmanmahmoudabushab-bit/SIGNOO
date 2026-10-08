import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export function LoginPage() {
  const { user, login, loading, error } = useAuth();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  if (user) {
    const from = (location.state as { from?: string } | null)?.from ?? "/fleet";
    return <Navigate to={from} replace />;
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    try {
      await login(username, password);
    } catch {
      // error already surfaced via auth context
    }
  }

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={onSubmit}>
        <h1>SIGNOO console</h1>
        <p className="hint">
          Demo accounts (mock mode, if the server is unreachable): operator/operator, reviewer/reviewer,
          engineer/engineer, admin/admin. Against the real server, sign in with the email/password from
          `python -m server.seed &lt;email&gt; &lt;password&gt;` (see server/README.md) — it takes an email, not
          a username.
        </p>
        <label>
          Email (or demo username, in mock mode)
          <input value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <button type="submit" disabled={loading}>
          {loading ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
