export function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  return <span className={`status-pill status-${normalized}`}>{status}</span>;
}
