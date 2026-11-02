from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "amount_rub",
    "account_age_days",
    "device_age_days",
    "distance_km",
    "transactions_last_hour",
    "hour",
    "new_payee",
    "channel",
    "merchant_category",
]
CATEGORICAL = ["channel", "merchant_category"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    amount_rub: float = Field(gt=0, le=5_000_000)
    account_age_days: int = Field(ge=0, le=20_000)
    device_age_days: int = Field(ge=0, le=20_000)
    distance_km: float = Field(ge=0, le=20_000)
    transactions_last_hour: int = Field(ge=0, le=200)
    hour: int = Field(ge=0, le=23)
    new_payee: Literal[0, 1]
    channel: Literal["mobile", "web", "pos"]
    merchant_category: Literal["retail", "travel", "digital", "services"]


EXAMPLES = {
    "Routine purchase": dict(
        amount_rub=1200,
        account_age_days=950,
        device_age_days=300,
        distance_km=2,
        transactions_last_hour=1,
        hour=14,
        new_payee=0,
        channel="pos",
        merchant_category="retail",
    ),
    "Unusual transfer": dict(
        amount_rub=85000,
        account_age_days=12,
        device_age_days=0,
        distance_km=1800,
        transactions_last_hour=12,
        hour=3,
        new_payee=1,
        channel="web",
        merchant_category="digital",
    ),
}
