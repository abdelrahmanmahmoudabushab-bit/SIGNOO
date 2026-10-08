// Small, honest disclosure of whether a screen is showing real server data or the seeded
// mock fallback — the task explicitly requires this distinction to stay visible, not just
// documented in a README.
export function SourceTag({ source }: { source: "server" | "mock" | null }) {
  if (!source) return null;
  return (
    <span className={`source-tag source-${source}`}>
      {source === "server" ? "live server data" : "mock data (server unreachable)"}
    </span>
  );
}
