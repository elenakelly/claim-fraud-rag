"""Rule-based fraud indicators.

Each indicator returns the exact evidence that triggered it, so every flag can be
traced back to the claim. This is what keeps the final explanation grounded.
"""

import re

from app.schemas import Claim, Indicator

# Text patterns: (code, weight, regex)
TEXT_RULES = [
    ("NO_RECEIPT_OR_CASH", 0.20, r"no receipts?|lost the receipts?|paid in cash|for cash"),
    ("VAGUE_DETAILS", 0.20, r"don'?t remember|not sure|can'?t recall|unclear"),
    ("URGENCY_PRESSURE", 0.15, r"urgent(ly)?|as soon as possible|immediately|need the (money|payout)"),
    ("NO_WITNESSES", 0.10, r"no witness(es)?"),
    ("NO_EMERGENCY_SERVICES", 0.15, r"(did not|didn'?t) go to the police|fire brigade was not called"),
    ("DAMAGE_INCONSISTENT", 0.25, r"older than the accident|damage seems older|looks older"),
]

THEFT_WORDS = re.compile(r"stolen|theft|burglary|robbed", re.IGNORECASE)


def detect(claim: Claim) -> list[Indicator]:
    text = claim.description
    found: list[Indicator] = []

    for code, weight, pattern in TEXT_RULES:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found.append(Indicator(code=code, weight=weight, evidence=f'"{match.group(0)}"'))

    if claim.policy_age_days < 60:
        found.append(Indicator(code="NEW_POLICY", weight=0.20,
                               evidence=f"policy_age_days={claim.policy_age_days}"))
    if claim.prior_claims >= 3:
        found.append(Indicator(code="MANY_PRIOR_CLAIMS", weight=0.20,
                               evidence=f"prior_claims={claim.prior_claims}"))
    if claim.days_to_report > 14:
        found.append(Indicator(code="LATE_REPORTING", weight=0.15,
                               evidence=f"days_to_report={claim.days_to_report}"))
    if claim.amount > 10_000:
        found.append(Indicator(code="HIGH_AMOUNT", weight=0.10,
                               evidence=f"amount={claim.amount:.0f}"))
    theft = THEFT_WORDS.search(text)
    if theft and not claim.police_report:
        found.append(Indicator(code="THEFT_WITHOUT_POLICE_REPORT", weight=0.15,
                               evidence=f'"{theft.group(0)}" and police_report=false'))

    return found


def indicator_score(indicators: list[Indicator]) -> float:
    return min(1.0, sum(i.weight for i in indicators))
