"""Combines indicators, retrieval and explanation into one screening decision."""

from app import indicators as ind
from app.explainer import CITATION, TemplateExplainer
from app.retriever import CaseRetriever, load_cases
from app.schemas import Claim, ScreeningResult, SimilarCase

INDICATOR_WEIGHT = 0.6  # share of the score from red flags
NEIGHBOUR_WEIGHT = 0.4  # share of the score from similar past cases
THRESHOLD = 0.45


class Screener:
    def __init__(self, retriever: CaseRetriever | None = None, explainer=None, k: int = 3):
        self.retriever = retriever or CaseRetriever(load_cases())
        self.explainer = explainer or TemplateExplainer()
        self.k = k

    def neighbour_score(self, matches) -> float:
        """Similarity-weighted share of fraud among the nearest past cases."""
        total = sum(sim for _, sim in matches)
        if total == 0:
            return 0.0
        return sum(sim for case, sim in matches if case.label == "fraud") / total

    def screen(self, claim: Claim) -> ScreeningResult:
        flags = ind.detect(claim)
        matches = self.retriever.search(claim.description, k=self.k)

        risk = INDICATOR_WEIGHT * ind.indicator_score(flags) + NEIGHBOUR_WEIGHT * self.neighbour_score(matches)
        risk = round(min(1.0, risk), 3)
        decision = "refer_to_investigator" if risk >= THRESHOLD else "fast_track"

        similar = [
            SimilarCase(case_id=c.case_id, label=c.label, similarity=round(s, 3), excerpt=c.description[:120])
            for c, s in matches
        ]
        explanation = self.explainer.explain(risk, decision, flags, similar)

        return ScreeningResult(
            claim_id=claim.claim_id,
            risk_score=risk,
            decision=decision,
            indicators=flags,
            similar_cases=similar,
            explanation=explanation,
            citations=sorted(set(CITATION.findall(explanation))),
            explainer=getattr(self.explainer, "name", "template"),
        )
