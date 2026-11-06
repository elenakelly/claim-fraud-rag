"""Evaluate the screener on a labelled set of claims it has never seen.

Measures two things:
1. Decision quality: accuracy, precision and recall for "fraud" (= refer to investigator).
2. Explanation quality:
   - groundedness: every [citation] in the explanation is an indicator or case the system actually found
   - evidence faithfulness: every quoted piece of evidence really appears in the claim text

Usage: python -m eval.run_eval [--min-recall 0.8] [--min-groundedness 1.0]
Exits with code 1 if a threshold is not met, so CI fails on regressions.
"""

import argparse
import json
import sys
from pathlib import Path

from app.explainer import check_grounding
from app.schemas import Claim
from app.screener import Screener

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "eval_claims.jsonl"


def evidence_is_faithful(result, claim: Claim) -> bool:
    for ind in result.indicators:
        if ind.evidence.startswith('"'):
            quoted = ind.evidence.split('"')[1]
            if quoted.lower() not in claim.description.lower():
                return False
    return True


def evaluate(path: Path = EVAL_FILE, screener: Screener | None = None) -> dict:
    screener = screener or Screener()
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    tp = fp = tn = fn = grounded = faithful = 0
    errors = []

    for row in rows:
        label = row.pop("label")
        claim = Claim(**row)
        result = screener.screen(claim)
        predicted_fraud = result.decision == "refer_to_investigator"

        if predicted_fraud and label == "fraud":
            tp += 1
        elif predicted_fraud:
            fp += 1
            errors.append((claim.claim_id, "false positive", result.risk_score))
        elif label == "fraud":
            fn += 1
            errors.append((claim.claim_id, "false negative", result.risk_score))
        else:
            tn += 1

        ok, _ = check_grounding(result.explanation, result.indicators, result.similar_cases)
        grounded += ok
        faithful += evidence_is_faithful(result, claim)

    n = len(rows)
    return {
        "n": n,
        "accuracy": (tp + tn) / n,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "groundedness": grounded / n,
        "evidence_faithfulness": faithful / n,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-groundedness", type=float, default=1.0)
    args = parser.parse_args()

    report = evaluate()
    for key in ("n", "accuracy", "precision", "recall", "groundedness", "evidence_faithfulness"):
        value = report[key]
        print(f"{key:>22}: {value:.2f}" if isinstance(value, float) else f"{key:>22}: {value}")
    for claim_id, kind, risk in report["errors"]:
        print(f"  {kind}: {claim_id} (risk {risk:.2f})")

    if report["recall"] < args.min_recall or report["groundedness"] < args.min_groundedness:
        print("FAILED: below threshold")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
