import enum
from typing import Optional
from typing import Any
from pydantic import BaseModel


class TransactionSchema(BaseModel):
    id: int
    price: float
    description: str
    user_id: int
    tax: Optional[str] = None
    email: Optional[str] = None
    paymentMethod: Optional[str] = None
    incCurrLabel: Optional[str] = None
    datetime: Optional[str] = None
    status: bool
    finished: bool


class TransactionCreateSchema(BaseModel):
    price: float
    name: str
    description: str


class TransactionReceiveSchema(BaseModel):
    OutSum: float
    InvId: int
    SignatureValue: Optional[str] = None
    Fee: Optional[str] = None
    EMail: Optional[str] = None
    PaymentMethod: Optional[str] = None
    IncCurrLabel: Optional[str] = None


class CreateTransactionResponseSchema(BaseModel):
    message: str
