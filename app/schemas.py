"""Request and response models for the claim screening API."""

from pydantic import BaseModel, Field


class Claim(BaseModel):
    claim_id: str = Field(..., examples=["CLM-1042"])
    description: str = Field(..., min_length=10, description="Free-text claim description")
    amount: float = Field(..., ge=0, description="Claimed amount in EUR")
    policy_age_days: int = Field(..., ge=0, description="Days between policy start and the incident")
    prior_claims: int = Field(0, ge=0)
    days_to_report: int = Field(0, ge=0, description="Days between the incident and the report")
    police_report: bool = False


class Indicator(BaseModel):
    code: str
    weight: float
    evidence: str  # the exact text or field value that triggered the indicator


class SimilarCase(BaseModel):
    case_id: str
    label: str
    similarity: float
    excerpt: str


class ScreeningResult(BaseModel):
    claim_id: str
    risk_score: float = Field(..., ge=0, le=1)
    decision: str  # "refer_to_investigator" or "fast_track"
    indicators: list[Indicator]
    similar_cases: list[SimilarCase]
    explanation: str
    citations: list[str]
    explainer: str  # which explainer produced the text: "template" or "llm"
