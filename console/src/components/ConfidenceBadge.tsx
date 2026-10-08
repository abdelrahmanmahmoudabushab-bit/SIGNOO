// A solid word vs a faded one — deaf-kiosk-ux skill's rule for showing confidence without
// requiring the reader to parse a number, reused here since operator tooling benefits from
// the same instantly-readable convention as the kiosk itself.
export function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const tier = confidence >= 0.75 ? "high" : confidence >= 0.5 ? "medium" : "low";
  return (
    <span className={`confidence-badge confidence-${tier}`} title={`${pct}% confidence`}>
      {pct}%
    </span>
  );
}
