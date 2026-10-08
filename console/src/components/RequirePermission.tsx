import type { ReactNode } from "react";
import { useAuth } from "../auth/AuthContext";
import { can, type Permission } from "../auth/permissions";

/**
 * Enforces FR-W9 (least privilege) at the routing layer, consistently, rather than each
 * screen re-implementing its own check. A role lacking the permission sees a plain
 * "not permitted" message, never a broken or partially-rendered screen.
 */
export function RequirePermission({ permission, children }: { permission: Permission; children: ReactNode }) {
  const { user } = useAuth();
  if (!can(user?.role, permission)) {
    return (
      <div className="not-permitted" role="alert">
        <h2>Not permitted</h2>
        <p>
          Your role (<strong>{user?.role}</strong>) does not have the <code>{permission}</code> permission.
          Ask an admin if you believe this is wrong.
        </p>
      </div>
    );
  }
  return <>{children}</>;
}
