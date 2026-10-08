import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { can } from "../auth/permissions";

const NAV_ITEMS: { to: string; label: string; permission: Parameters<typeof can>[1] }[] = [
  { to: "/fleet", label: "Fleet", permission: "fleet:view" },
  { to: "/sessions", label: "Live sessions", permission: "session:view" },
  { to: "/review", label: "Review workspace", permission: "review:decide" },
  { to: "/dataset", label: "Dataset", permission: "dataset:view" },
  { to: "/training", label: "Training", permission: "training:view" },
  { to: "/registry", label: "Model registry", permission: "registry:view" },
  { to: "/knowledge-base", label: "Knowledge base", permission: "kb:view" },
  { to: "/logs", label: "Logs & alerts", permission: "logs:view" },
];

export function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="app-shell">
      <nav className="sidebar" aria-label="Main navigation">
        <div className="brand">SIGNOO console</div>
        <ul>
          {NAV_ITEMS.filter((item) => can(user?.role, item.permission)).map((item) => (
            <li key={item.to}>
              <NavLink to={item.to} className={({ isActive }) => (isActive ? "active" : undefined)}>
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
        <div className="user-box">
          <div>
            <strong>{user?.username}</strong>
            <div className="role-tag">{user?.role}</div>
          </div>
          <button onClick={logout}>Log out</button>
        </div>
      </nav>
      <main className="content">
        <Outlet />
      </main>
    </div>
  );
}
