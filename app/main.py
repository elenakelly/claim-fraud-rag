"""FastAPI service: POST /screen returns a risk score, a decision and a grounded explanation."""

import logging
import time

from fastapi import FastAPI, Request

from app.explainer import get_explainer
from app.schemas import Claim, ScreeningResult
from app.screener import Screener

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("claim-fraud-rag")

app = FastAPI(
    title="Claim Fraud Screener",
    description="Screens insurance claims for fraud indicators and explains why, citing its evidence.",
    version="0.1.0",
)
screener = Screener(explainer=get_explainer())


@app.middleware("http")
async def log_latency(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cases_indexed": len(screener.retriever.cases)}


@app.post("/screen", response_model=ScreeningResult)
def screen(claim: Claim) -> ScreeningResult:
    result = screener.screen(claim)
    # Log the decision but not the claim text (GDPR: keep personal data out of logs)
    log.info("claim=%s risk=%.2f decision=%s flags=%s explainer=%s", result.claim_id, result.risk_score,
             result.decision, [i.code for i in result.indicators], result.explainer)
    return result
