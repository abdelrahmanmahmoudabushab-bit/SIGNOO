import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { getToken, setToken as persistToken } from "../api/client";
import { login as apiLogin } from "../api/endpoints";
import type { AuthUser } from "../api/types";

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  loading: boolean;
  error: string | null;
  source: "server" | "mock" | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | undefined>(undefined);

const USER_KEY = "signoo_console_user";

function loadStoredUser(): AuthUser | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as AuthUser) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(() => loadStoredUser());
  const [token, setTokenState] = useState<string | null>(() => getToken());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<"server" | "mock" | null>(null);

  const login = useCallback(async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const { data, source: src } = await apiLogin(username, password);
      persistToken(data.token);
      setTokenState(data.token);
      setUser(data.user);
      setSource(src);
      try {
        localStorage.setItem(USER_KEY, JSON.stringify(data.user));
      } catch {
        // best-effort persistence only
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "login failed");
      throw e;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    persistToken(null);
    try {
      localStorage.removeItem(USER_KEY);
    } catch {
      // ignore
    }
    setTokenState(null);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, token, loading, error, source, login, logout }),
    [user, token, loading, error, source, login, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
