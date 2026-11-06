# Claim Fraud Screener

A small retrieval-augmented service that screens insurance claims for fraud indicators and tells an investigator **why** a claim was flagged, citing only evidence it can point to.

```
claim ──► red-flag rules ─────────┐
   │                              ├──► risk score + decision ──► explanation with [citations]
   └────► retrieve similar cases ─┘                                  │
                                                                     └── grounding check
```

## Why

A fraud model is only useful if an investigator can trust it. Every flag in this service carries the exact text or field that triggered it, every explanation cites only indicators and past cases the system really found, and a grounding check rejects any explanation that cites something else (for example, a case id an LLM made up).

## How it works

| Step | File | What it does |
|---|---|---|
| Red flags | `app/indicators.py` | Rules for signals such as a brand-new policy, cash and no receipt, vague details, late reporting, theft without a police report. Each returns its evidence. |
| Retrieval | `app/retriever.py` | Finds the most similar past cases (TF-IDF, runs offline). Swap in dense embeddings by implementing `search`. |
| Scoring | `app/screener.py` | 60% red flags + 40% fraud share among similar cases. Score ≥ 0.45 → refer to an investigator, otherwise fast-track. |
| Explanation | `app/explainer.py` | Template explainer by default. With `OPENAI_API_KEY` or `AZURE_OPENAI_ENDPOINT` set, an LLM writes the explanation; if it cites anything it was not given, the template is used instead. |
| API | `app/main.py` | FastAPI: `POST /screen`, `GET /health`. Logs latency and decisions, never the claim text. |

## Run it

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
# open http://localhost:8000/docs
```

Or with Docker:

```bash
docker build -t claim-fraud-rag .
docker run -p 8000:8000 claim-fraud-rag
```

Example request:

```bash
curl -X POST localhost:8000/screen -H "Content-Type: application/json" -d '{
  "claim_id": "CLM-1042",
  "description": "Laptop and camera stolen from my car. No receipt, I paid in cash. I need the money urgently.",
  "amount": 3400, "policy_age_days": 12, "prior_claims": 0, "days_to_report": 1, "police_report": false
}'
```

Response (shortened):

```json
{
  "risk_score": 0.82,
  "decision": "refer_to_investigator",
  "explanation": "Refer to an investigator (risk 0.82). Red flags: [NO_RECEIPT_OR_CASH] \"No receipt\"; [NEW_POLICY] policy_age_days=12; [URGENCY_PRESSURE] \"need the money\"; [THEFT_WITHOUT_POLICE_REPORT] \"stolen\" and police_report=false. Most similar past cases: [C-001] (fraud, similarity 0.60), ...",
  "explainer": "template"
}
```

## Tests and evaluation

```bash
python -m pytest -q        # 15 tests: rules, retrieval, decisions, grounding, API, LLM fallback
python -m eval.run_eval    # decision and explanation quality on unseen claims
```

The evaluation reports accuracy, precision, recall, **groundedness** (all citations are real) and **evidence faithfulness** (all quoted evidence appears in the claim). CI fails if recall drops below 0.8 or groundedness below 1.0.

The LLM path is tested with a stub client, including a test where the "LLM" invents a case id and the service falls back to the template.

## CI

GitHub Actions runs the tests and the evaluation, builds the Docker image and smoke-tests the running API.

## Limitations and next steps

- The data is a small synthetic set (24 past cases, 14 evaluation claims), so the scores are optimistic. A real system needs real, labelled claims and a held-out test set.
- The rules and score weights are hand-set. Next step: learn them with a classifier (e.g. logistic regression or gradient boosting) and calibrate the threshold on investigator capacity.
- Replace TF-IDF with dense embeddings and a vector index (e.g. Azure AI Search) for better retrieval on paraphrased claims.
- Add an LLM-as-judge evaluation for explanation clarity, and tracing (latency, token cost) for the LLM path.
- Red flags are reasons to look closer, not proof of fraud: decisions should stay with human investigators.
