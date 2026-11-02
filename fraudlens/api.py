import os
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraudlens.schema import Transaction
from fraudlens.service import FraudService


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=500)


def create_app(artifacts=None, api_key=None):
    key = os.getenv("FRAUDLENS_API_KEY", "") if api_key is None else api_key

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = FraudService() if artifacts is None else FraudService(artifacts)
        yield

    app = FastAPI(title="FraudLens", version="1.0.0", lifespan=lifespan)

    def authorize(x_api_key: str | None = Header(default=None)):
        if key and not secrets.compare_digest(key, x_api_key or ""):
            raise HTTPException(401, "Invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "synthetic": True}

    @app.post("/v1/score", dependencies=[Depends(authorize)])
    def score(transaction: Transaction):
        return app.state.service.score([transaction])[0]

    @app.post("/v1/batch", dependencies=[Depends(authorize)])
    def batch(request: Batch):
        return {"scores": app.state.service.score(request.transactions, explain=False)}

    @app.post("/v1/drift", dependencies=[Depends(authorize)])
    def drift(request: Batch):
        if len(request.transactions) < 100:
            raise HTTPException(422, "At least 100 rows are needed")
        return {"features": app.state.service.drift(request.transactions)}

    return app


app = create_app()
