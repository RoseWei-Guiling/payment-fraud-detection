"""
Step 8: a small fraud-scoring API.

  uvicorn api.app:app --reload        then open http://127.0.0.1:8000/docs

POST /score with {"payments": [{"TransactionAmt": 120.0, "ProductCD": "W", ...}, ...]}
Any feature left out is treated as blank, exactly as the model handled blanks in training.
"""
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scoring import Scorer  # noqa: E402

app = FastAPI(title="Payment fraud scoring", version="1.0",
              description="LightGBM fraud score with an amount-aware block / allow decision and reasons.")
scorer = None


def get_scorer():
    global scorer
    if scorer is None:
        scorer = Scorer()
    return scorer


class ScoreRequest(BaseModel):
    payments: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000,
                                           examples=[[{"TransactionAmt": 250.0, "ProductCD": "C",
                                                       "uid_n_24h": 4, "P_emaildomain": "gmail.com"}]])


@app.get("/health")
def health():
    s = get_scorer()
    return {"status": "ok", "features": len(s.config["features"]), "cost_per_blocked_genuine": s.cost}


@app.post("/score")
def score(request: ScoreRequest):
    return {"results": get_scorer().score(request.payments)}
