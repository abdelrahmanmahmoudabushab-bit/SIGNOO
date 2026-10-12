import type { WriteState } from "../hooks/useWrite";

// The visible outcome of the last write on a page (see hooks/useWrite.ts): the server's reason
// when it refused, or a plain statement that nothing real changed when no server was reached.
export function WriteStatus({ write }: { write: WriteState }) {
  return (
    <>
      {write.error && (
        <p role="alert" className="gate-fail">
          {write.error}
        </p>
      )}
      {write.unsent && (
        <p role="status" className="mock-write-note">
          {write.unsent}: not sent to a server. This change is shown on this page only (mock data); nothing real was
          changed.
        </p>
      )}
    </>
  );
}
