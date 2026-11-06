"""Explanations for investigators (the "G" in RAG).

Two explainers share one contract: the explanation may only cite indicator codes
and case ids that were actually given to it, in square brackets, e.g. [NEW_POLICY]
or [C-007]. `check_grounding` enforces that contract.

- TemplateExplainer: deterministic, offline, always grounded.
- LLMExplainer: uses OpenAI or Azure OpenAI when credentials are set. If the model
  cites something it was not given, the result is rejected and the template is used.
"""

import json
import os
import re

from app.schemas import Indicator, SimilarCase

CITATION = re.compile(r"\[([A-Z][A-Z0-9_\-]+)\]")


def allowed_citations(indicators: list[Indicator], cases: list[SimilarCase]) -> set[str]:
    return {i.code for i in indicators} | {c.case_id for c in cases}


def check_grounding(explanation: str, indicators: list[Indicator], cases: list[SimilarCase]) -> tuple[bool, list[str]]:
    """Return (is_grounded, unsupported_citations)."""
    cited = set(CITATION.findall(explanation))
    unsupported = sorted(cited - allowed_citations(indicators, cases))
    return (not unsupported and bool(cited or not indicators)), unsupported


class TemplateExplainer:
    name = "template"

    def explain(self, risk: float, decision: str, indicators: list[Indicator], cases: list[SimilarCase]) -> str:
        verdict = "Refer to an investigator" if decision == "refer_to_investigator" else "Fast-track"
        parts = [f"{verdict} (risk {risk:.2f})."]
        if indicators:
            flags = "; ".join(f"[{i.code}] {i.evidence}" for i in sorted(indicators, key=lambda i: -i.weight))
            parts.append(f"Red flags: {flags}.")
        else:
            parts.append("No red flags found in the claim.")
        if cases:
            sims = ", ".join(f"[{c.case_id}] ({c.label}, similarity {c.similarity:.2f})" for c in cases)
            parts.append(f"Most similar past cases: {sims}.")
        return " ".join(parts)


SYSTEM_PROMPT = """You help insurance fraud investigators triage claims.
Write 2-4 sentences explaining the decision you are given. Rules:
- Cite evidence ONLY with the codes and case ids provided, in square brackets, e.g. [NEW_POLICY], [C-007].
- Do not invent facts, amounts or cases. Do not change the decision.
- Be neutral: a red flag is a reason to look closer, not proof of fraud.
Return JSON: {"explanation": "..."}"""


class LLMExplainer:
    name = "llm"

    def __init__(self, client=None, model: str | None = None):
        self.client = client or self._client_from_env()
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.fallback = TemplateExplainer()

    @staticmethod
    def _client_from_env():
        from openai import AzureOpenAI, OpenAI  # optional dependency

        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            return AzureOpenAI(api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"))
        return OpenAI()

    def explain(self, risk, decision, indicators, cases) -> str:
        context = {
            "decision": decision,
            "risk_score": round(risk, 2),
            "indicators": [i.model_dump() for i in indicators],
            "similar_cases": [c.model_dump() for c in cases],
        }
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context)},
                ],
            )
            text = json.loads(response.choices[0].message.content)["explanation"]
        except Exception:  # network error, bad JSON, missing key
            self.name = "template"
            return self.fallback.explain(risk, decision, indicators, cases)

        grounded, _ = check_grounding(text, indicators, cases)
        if not grounded:
            self.name = "template"
            return self.fallback.explain(risk, decision, indicators, cases)
        self.name = "llm"
        return text


def get_explainer():
    if os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_ENDPOINT"):
        try:
            return LLMExplainer()
        except ImportError:
            pass
    return TemplateExplainer()
