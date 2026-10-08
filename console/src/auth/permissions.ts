// The single source of truth for FR-W9 (access control). Every screen and every action that
// should be hidden or disabled for a role checks THIS table, so enforcement is consistent
// rather than ad hoc per screen. Camera imagery is deliberately absent from this table: no
// endpoint for it exists yet (no video is ever retained per NFR-S2); when one is added it is
// reserved for `admin` and must be audited with a reason (API_CONTRACTS.md §5, NFR-S4).
import type { Role } from "../api/types";

export type Permission =
  | "fleet:view"
  | "session:view"
  | "review:decide"
  | "dataset:view"
  | "dataset:manage"
  | "training:view"
  | "registry:view"
  | "registry:promote"
  | "kb:view"
  | "kb:edit"
  | "logs:view"
  | "alerts:view"
  | "alerts:ack";

const MATRIX: Record<Permission, Role[]> = {
  "fleet:view": ["operator", "reviewer", "engineer", "admin"],
  "session:view": ["operator", "engineer", "admin"],
  "review:decide": ["reviewer", "admin"],
  "dataset:view": ["reviewer", "engineer", "admin"],
  "dataset:manage": ["engineer", "admin"],
  "training:view": ["engineer", "admin"],
  "registry:view": ["engineer", "admin"],
  "registry:promote": ["engineer", "admin"],
  "kb:view": ["reviewer", "engineer", "admin"],
  "kb:edit": ["engineer", "admin"],
  "logs:view": ["operator", "engineer", "admin"],
  "alerts:view": ["operator", "engineer", "admin"],
  "alerts:ack": ["operator", "engineer", "admin"],
};

export function can(role: Role | undefined | null, permission: Permission): boolean {
  if (!role) return false;
  return MATRIX[permission].includes(role);
}

export function rolesFor(permission: Permission): Role[] {
  return MATRIX[permission];
}
