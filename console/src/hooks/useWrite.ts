import { useCallback, useState } from "react";
import type { Sourced } from "../api/client";

export interface WriteState {
  /** What is being written right now (a version, a queue id, ...), or null. */
  busy: string | null;
  /** The server's reason for refusing the last write, or null. */
  error: string | null;
  /** Set when the last write reached no server and only changed what this page displays. */
  unsent: string | null;
}

// One place for what every write in the console must do: report what the server said. A
// refusal (the promotion gate's 409, a 403 for the role, a 404) is shown with the server's
// reason and `apply` is not called, so the displayed data stays as it was; the busy state is
// always cleared. A write that reached no server at all is applied to the page but flagged in
// `unsent`, so it is never mistaken for a real change.
export function useWrite(): WriteState & {
  run: <T>(subject: string, write: () => Promise<Sourced<T>>, apply: (data: T, source: Sourced<T>["source"]) => void) => Promise<void>;
} {
  const [state, setState] = useState<WriteState>({ busy: null, error: null, unsent: null });

  const run = useCallback(async function run<T>(
    subject: string,
    write: () => Promise<Sourced<T>>,
    apply: (data: T, source: Sourced<T>["source"]) => void
  ) {
    setState({ busy: subject, error: null, unsent: null });
    try {
      const { data, source } = await write();
      apply(data, source);
      setState({ busy: null, error: null, unsent: source === "mock" ? subject : null });
    } catch (e) {
      setState({ busy: null, error: `${subject}: ${e instanceof Error ? e.message : "action failed"}`, unsent: null });
    }
  }, []);

  return { ...state, run };
}
