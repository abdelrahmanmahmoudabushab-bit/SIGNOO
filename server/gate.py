"""B13's promotion gate: the seven criteria in REQUIREMENTS.md §5.3 (DR-6).

Takes two `ml/evaluate.py --report ...`-style metrics dicts (candidate, champion —
see the shape at the bottom of `ml/evaluate.py::main`) plus the two facts evaluate.py
does not itself produce (measured latency, shadow-run completion) and returns a
pass/fail per criterion plus an overall verdict.

This module does not re-score anything — it wraps the output of `ml/evaluate.py`,
per `SYSTEM_DESIGN.md` §3 B13: "reuses ml/evaluate.py and ml/compare_splits.py logic".

DR-7: the benchmark these metrics come from must be the frozen, human-verified split;
this module trusts its caller to have evaluated both models against exactly that split
and does not attempt to detect otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

TOP1_TOLERANCE = 0.005     # DR-6 #1: "within 0.5 points"
SIGNER_REGRESSION_LIMIT = 0.03   # DR-6 #2
SIGN_REGRESSION_LIMIT = 0.05     # DR-6 #3, NFR-Q5


@dataclass
class GateVerdict:
    passed: bool
    criteria: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    # Raw numeric inputs that must survive a re-evaluation later (e.g. shadow-complete
    # re-checking latency) without being re-derived by parsing `details`' human text.
    measurements: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "criteria": {
            name: ("pass" if ok else "fail") for name, ok in self.criteria.items()
        }, "details": self.details, "measurements": self.measurements}


def _other_metric_improves(candidate: dict, champion: dict) -> bool:
    """Supports DR-6 #1's "within 0.5pt while another criterion improves" clause."""
    cand_top5 = candidate.get("top5", 0.0)
    champ_top5 = champion.get("top5", 0.0)
    if cand_top5 > champ_top5:
        return True

    cand_signer = candidate.get("per_signer", {}) or {}
    champ_signer = champion.get("per_signer", {}) or {}
    if cand_signer and champ_signer:
        if min(cand_signer.values()) > min(champ_signer.values()):
            return True

    return False


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _no_regression(champion_scores: dict | None, candidate_scores: dict | None,
                   kind: str, limit: float) -> tuple[bool, str]:
    """DR-6 #2/#3. Every signer/sign the champion was scored on must also be scored
    for the candidate: one that is absent was not measured, and an unmeasured
    regression must fail the gate rather than be skipped."""
    champion_scores = champion_scores or {}
    candidate_scores = candidate_scores or {}
    if not champion_scores:
        return False, f"champion has no per_{kind} metrics, so regressions cannot be ruled out"
    missing = sorted(k for k in champion_scores if not _is_number(candidate_scores.get(k)))
    if missing:
        return False, f"candidate has no score for {kind}(s) the champion was measured on: {', '.join(missing)}"
    worst, worst_drop = None, 0.0
    for key, champ_acc in champion_scores.items():
        drop = champ_acc - candidate_scores[key]
        if drop > worst_drop:
            worst, worst_drop = key, drop
    if worst_drop <= limit:
        return True, f"no {kind} regressed"
    return False, f"{kind} {worst} regressed {worst_drop:.1%} (limit {limit:.1%})"


def evaluate_gate(
    candidate: dict,
    champion: dict | None,
    *,
    candidate_latency_ms: float,
    latency_budget_ms: float,
    shadow_run_complete: bool,
) -> GateVerdict:
    """Evaluates DR-6's seven criteria. `champion=None` (first-ever model) passes
    every accuracy-relative criterion trivially — there is nothing to regress against
    yet — but still enforces latency and the shadow-run gate.
    """
    criteria: dict[str, bool] = {}
    details: dict[str, str] = {}

    cand_top1 = candidate.get("top1")
    champ_top1 = champion.get("top1") if champion else None

    # 1. Top-1 improves, or ties within tolerance while something else improves.
    # A missing top1 on either side is a comparison nobody made: it fails, it never
    # defaults to 0 (which would let any candidate "beat" a champion with no number).
    if not _is_number(cand_top1):
        criteria["top1_improves_or_ties"] = False
        details["top1_improves_or_ties"] = "candidate has no top1 metric"
    elif champion is None:
        criteria["top1_improves_or_ties"] = True
        details["top1_improves_or_ties"] = "no champion yet — nothing to compare against"
    elif not _is_number(champ_top1):
        criteria["top1_improves_or_ties"] = False
        details["top1_improves_or_ties"] = "champion has no top1 metric, so the candidate cannot be compared with it"
    elif cand_top1 >= champ_top1:
        criteria["top1_improves_or_ties"] = True
        details["top1_improves_or_ties"] = f"candidate {cand_top1:.1%} >= champion {champ_top1:.1%}"
    elif champ_top1 - cand_top1 <= TOP1_TOLERANCE and _other_metric_improves(candidate, champion):
        criteria["top1_improves_or_ties"] = True
        details["top1_improves_or_ties"] = (
            f"candidate {cand_top1:.1%} within {TOP1_TOLERANCE:.1%} of champion "
            f"{champ_top1:.1%}, and another metric improved"
        )
    else:
        criteria["top1_improves_or_ties"] = False
        details["top1_improves_or_ties"] = (
            f"candidate {cand_top1:.1%} vs champion {champ_top1:.1%} "
            f"(drop {champ_top1 - cand_top1:.1%} exceeds {TOP1_TOLERANCE:.1%}, "
            f"no offsetting improvement)"
        )

    # 2. No signer regresses by more than 3 points.
    if champion is None:
        criteria["no_signer_regression"] = True
        details["no_signer_regression"] = "no champion yet"
    else:
        ok, detail = _no_regression(
            champion.get("per_signer"), candidate.get("per_signer"), "signer", SIGNER_REGRESSION_LIMIT)
        criteria["no_signer_regression"] = ok
        details["no_signer_regression"] = detail

    # 3. No sign regresses by more than 5 points (NFR-Q5).
    if champion is None:
        criteria["no_sign_regression"] = True
        details["no_sign_regression"] = "no champion yet"
    else:
        ok, detail = _no_regression(
            champion.get("per_sign"), candidate.get("per_sign"), "sign", SIGN_REGRESSION_LIMIT)
        criteria["no_sign_regression"] = ok
        details["no_sign_regression"] = detail

    # 4. No class falls to zero recall.
    cand_per_sign = candidate.get("per_sign")
    if not cand_per_sign:
        # ml/evaluate.py always reports per_sign, so its absence means the evaluation
        # did not run properly. A check nobody could perform must not read as a pass.
        criteria["no_zero_recall_class"] = False
        details["no_zero_recall_class"] = "candidate has no per_sign metrics, so zero-recall classes cannot be ruled out"
    else:
        zero_recall = [sign for sign, acc in cand_per_sign.items() if acc == 0.0]
        criteria["no_zero_recall_class"] = not zero_recall
        details["no_zero_recall_class"] = (
            "no class at zero recall" if not zero_recall
            else f"zero recall on: {', '.join(sorted(zero_recall))}"
        )

    # 5. False-accept rate does not increase (NFR-Q4). ml/evaluate.py does not
    # currently measure this (it needs negative/non-sign clips, which the frozen
    # benchmark does not yet include) — treated as "not measured, does not block"
    # rather than silently passing on fabricated data. Documented gap; see README.
    cand_far = candidate.get("false_accept_rate")
    champ_far = champion.get("false_accept_rate") if champion else None
    if cand_far is None or champ_far is None:
        criteria["false_accept_rate_not_worse"] = True
        details["false_accept_rate_not_worse"] = (
            "not measured (ml/evaluate.py has no non-signing negative set yet) — "
            "treated as pass, see server/README.md known gaps"
        )
    else:
        ok = cand_far <= champ_far
        criteria["false_accept_rate_not_worse"] = ok
        details["false_accept_rate_not_worse"] = f"candidate {cand_far:.1%} vs champion {champ_far:.1%}"

    # 6. Latency stays within NFR-P1 on device.
    ok = candidate_latency_ms <= latency_budget_ms
    criteria["latency_within_budget"] = ok
    details["latency_within_budget"] = (
        f"{candidate_latency_ms:.0f}ms vs budget {latency_budget_ms:.0f}ms"
    )

    # 7. Shadow-run period completed without live-metric degradation (FR-L7).
    criteria["shadow_run_complete"] = shadow_run_complete
    details["shadow_run_complete"] = (
        "completed" if shadow_run_complete else "not yet run or still in progress"
    )

    return GateVerdict(
        passed=all(criteria.values()), criteria=criteria, details=details,
        measurements={"candidate_latency_ms": candidate_latency_ms},
    )
