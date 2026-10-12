// Small, honest disclosure of whether a screen is showing real server data or the seeded
// mock fallback — the task explicitly requires this distinction to stay visible, not just
// documented in a README. A page showing more than one dataset renders one tag per dataset,
// each with a `label` saying which data it describes, since one can be real while another is
// mock (the server may serve the registry but not, say, training runs).
export function SourceTag({ source, label }: { source: "server" | "mock" | null; label?: string }) {
  if (!source) return null;
  return (
    <span className={`source-tag source-${source}`}>
      {label && `${label}: `}
      {source === "server" ? "live server data" : "mock data (not from the server)"}
    </span>
  );
}
