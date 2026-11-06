import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import indicators
from app.explainer import LLMExplainer, TemplateExplainer, check_grounding
from app.main import app
from app.retriever import CaseRetriever, load_cases
from app.schemas import Claim, Indicator, SimilarCase
from app.screener import Screener
from eval.run_eval import evaluate

SUSPICIOUS = Claim(
    claim_id="T-1",
    description="Laptop stolen from my car. No receipt, I paid in cash. I need the money urgently.",
    amount=3000, policy_age_days=10, prior_claims=0, days_to_report=1, police_report=False,
)
ROUTINE = Claim(
    claim_id="T-2",
    description="Another car hit me at a junction, police attended and the driver accepted fault. Garage quote attached.",
    amount=1800, policy_age_days=900, prior_claims=0, days_to_report=1, police_report=True,
)


@pytest.fixture(scope="module")
def screener():
    return Screener()


# --- indicators -------------------------------------------------------------

def test_detects_expected_red_flags():
    codes = {i.code for i in indicators.detect(SUSPICIOUS)}
    assert {"NO_RECEIPT_OR_CASH", "URGENCY_PRESSURE", "NEW_POLICY", "THEFT_WITHOUT_POLICE_REPORT"} <= codes


def test_routine_claim_has_no_red_flags():
    assert indicators.detect(ROUTINE) == []


def test_indicator_evidence_is_quoted_from_the_claim():
    for ind in indicators.detect(SUSPICIOUS):
        if ind.evidence.startswith('"'):
            assert ind.evidence.split('"')[1].lower() in SUSPICIOUS.description.lower()


# --- retrieval --------------------------------------------------------------

def test_retriever_finds_a_similar_theft_case():
    retriever = CaseRetriever(load_cases())
    top_case, score = retriever.search("bike stolen at the station, no witnesses", k=1)[0]
    assert top_case.case_id == "C-005"
    assert score > 0


def test_retriever_rejects_empty_case_base():
    with pytest.raises(ValueError):
        CaseRetriever([])


# --- screening decisions ----------------------------------------------------

def test_suspicious_claim_is_referred(screener):
    result = screener.screen(SUSPICIOUS)
    assert result.decision == "refer_to_investigator"
    assert 0 <= result.risk_score <= 1


def test_routine_claim_is_fast_tracked(screener):
    assert screener.screen(ROUTINE).decision == "fast_track"


# --- grounding --------------------------------------------------------------

def test_template_explanation_is_grounded(screener):
    result = screener.screen(SUSPICIOUS)
    grounded, unsupported = check_grounding(result.explanation, result.indicators, result.similar_cases)
    assert grounded and unsupported == []


def test_grounding_check_catches_invented_citations():
    flags = [Indicator(code="NEW_POLICY", weight=0.2, evidence="policy_age_days=10")]
    cases = [SimilarCase(case_id="C-001", label="fraud", similarity=0.4, excerpt="...")]
    grounded, unsupported = check_grounding("Flagged because of [NEW_POLICY] and [C-999].", flags, cases)
    assert not grounded
    assert unsupported == ["C-999"]


def _fake_llm(text):
    """A stand-in for the OpenAI client that always returns `text`."""
    message = SimpleNamespace(content=json.dumps({"explanation": text}))
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    create = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_explanation_is_used_when_grounded():
    explainer = LLMExplainer(client=_fake_llm("Look closer: the policy is brand new [NEW_POLICY]."))
    result = Screener(explainer=explainer).screen(SUSPICIOUS)
    assert result.explainer == "llm"
    assert "[NEW_POLICY]" in result.explanation


def test_hallucinated_llm_explanation_falls_back_to_template():
    explainer = LLMExplainer(client=_fake_llm("Same fraud ring as case [C-999]."))
    result = Screener(explainer=explainer).screen(SUSPICIOUS)
    assert result.explainer == "template"
    assert "C-999" not in result.explanation


# --- API --------------------------------------------------------------------

def test_api_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["cases_indexed"] > 0


def test_api_screen_returns_explanation_with_citations():
    response = TestClient(app).post("/screen", json=SUSPICIOUS.model_dump())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "refer_to_investigator"
    assert body["citations"]


def test_api_rejects_invalid_claim():
    response = TestClient(app).post("/screen", json={"claim_id": "X", "description": "short", "amount": -5,
                                                     "policy_age_days": 1})
    assert response.status_code == 422


# --- evaluation as a regression test ----------------------------------------

def test_eval_set_meets_quality_bar():
    report = evaluate()
    assert report["recall"] >= 0.8
    assert report["groundedness"] == 1.0
    assert report["evidence_faithfulness"] == 1.0
