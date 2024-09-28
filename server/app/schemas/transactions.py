import enum
from typing import Optional
from typing import Any
from pydantic import BaseModel


class TransactionSchema(BaseModel):
    id: int
    price: float
    description: str
    user_id: int
    tax: str
    email: str
    paymentMethod: str
    incCurrLabel: str
    datetime: str
    status: bool
    finished: bool


class TransactionCreateSchema(BaseModel):
    price: float
    name: str
    description: str
    user_id: int
    datetime: str


class TransactionReceiveSchema(BaseModel):
    OutSum: float
    InvId: int
    Fee: float
    EMail: str
    SignatureValue: str
    PaymentMethod: str
    IncCurrLabel: str


class CreateTransactionResponseSchema(BaseModel):
    message: str
