"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

ContractType = Literal["month_to_month", "one_year", "two_year"]
RiskBucket = Literal["low", "medium", "high"]


class CustomerFeatures(BaseModel):
    """Features for one customer, as exported nightly by the billing system."""

    customer_id: str = Field(min_length=1, description="Billing-system customer identifier")
    tenure_months: int = Field(ge=0, le=120, description="Months since the account was opened")
    monthly_charges: float = Field(ge=0, description="Current monthly bill")
    total_charges: float = Field(ge=0, description="Lifetime charges to date")
    num_support_tickets: int = Field(ge=0, description="Support tickets in the last 90 days")
    contract_type: ContractType
    has_tech_support: bool


class PredictionResponse(BaseModel):
    """One churn prediction. ``customer_id`` is echoed so callers can join results."""

    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    churn_risk: RiskBucket
    model_version: str
