import requests
import base64
import json
import hashlib
import hmac
import decimal

from datetime import datetime, timedelta

from sqlalchemy.sql import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from fastapi.encoders import jsonable_encoder

from app.dependencies.jwt import jwt_verify
from app.models import Transactions, User
from app.core.database import get_db
from app.schemas import ErrorResponseSchema
from app.schemas.transactions import TransactionCreateSchema, CreateTransactionResponseSchema, TransactionSchema, TransactionReceiveSchema
from app.config import MerchantLogin, password1, password2
from urllib import parse
from urllib.parse import urlparse


router = APIRouter(prefix="/transactions", tags=["Транзакции"])


def date_month(m):
    current_date = datetime.now().date()
    # Получение даты через месяц
    year = current_date.year
    month = current_date.month + m
    if month > 12:
        month = month % 12
        year += 1
    # Определяем последний день текущего месяца
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = current_date.day
    elif month in [4, 6, 9, 11]:
        day = min(current_date.day, 30)  # 30 дней
    else:  # Февраль
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            day = min(current_date.day, 29)  # Високосный год
        else:
            day = min(current_date.day, 28)  # Невисокосный год
    next_month_date = datetime(year, month, day).date()
    return next_month_date


def generate_payment_link(
    merchant_login: str,  # Merchant login
    merchant_password_1: str,  # Merchant password
    cost: decimal,  # Cost of goods, RU
    number: int,  # Invoice number
    description: str,  # Description of the purchase
    robokassa_payment_url='https://auth.robokassa.ru/Merchant/Index.aspx',
) -> str:
    """URL for redirection of the customer to the service.
    """
    signature = calculate_signature(
        merchant_login,
        cost,
        number,
        {"items": [{"name": description, "quantity": 1, "sum": cost, "tax": "none"}]},
        merchant_password_1
    )

    data = {
        'MerchantLogin': merchant_login,
        'OutSum': cost,
        'invoiceID': number,
        "Receipt": {"items": [{"name": description, "quantity": 1, "sum": cost, "tax": "none"}]},
        'SignatureValue': signature,
    }
    return f'{robokassa_payment_url}?{parse.urlencode(data)}'


def calculate_signature(*args) -> str:
    """Create signature MD5.
    """
    return hashlib.md5(':'.join(str(arg) for arg in args).encode()).hexdigest()


@router.post(
    "",  # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Создание ссылки для оплаты",
)
async def create_payment_url(
    data: TransactionCreateSchema,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify)
):
    if data.name == "1 month":
        price = 1950
    elif data.name == "3 month":
        price = 3900
    elif data.name == "6 month":
        price = 5850
    elif data.name == "12 month":
        price = 7850
    elif data.name == "1 day":
        price = 1
    else:
        price = 7850
    new_transaction = Transactions(
        price=price,
        name=data.name,
        description=data.description,
        user_id=user.id,
        datetime=datetime.now()
    )

    db.add(new_transaction)
    await db.commit()
    await db.refresh(new_transaction)
    query = select(Transactions).where(Transactions.user_id == user.id).order_by(Transactions.id.desc())
    result = await db.execute(query)
    transaction_id = result.scalars().first().id

    d = {
        "MerchantLogin": MerchantLogin,
        "InvId": transaction_id,
        "OutSum": price,
        "Description": data.description,
    }

    sign = f"{d['MerchantLogin']}:{d['OutSum']}:{d['InvId']}:{password1}"  # тут password1 для тестовых платежей

    d["SignatureValue"] = hashlib.md5(sign.encode()).hexdigest()

    url = "https://auth.robokassa.ru/Merchant/Indexjson.aspx?"
    response = requests.post(url=url, data=d)
    pay_url = generate_payment_link(MerchantLogin, password1, price, transaction_id, data.description)
    return Response(content=f"{pay_url}", media_type='text/plain')


@router.post(
    "/accept",  # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Принятие ответа от платежной системы",
)
async def accept_payment(
    data: TransactionReceiveSchema,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify)
):
    query = select(Transactions).where(Transactions.id == data.invId)
    print(data)
    result = await db.execute(query)
    print(result)
    transaction = result.scalar().first()
    sign = ':'.join(data.outSum, data.invId, password2)
    sign_encode = hashlib.md5(sign.encode()).hexdigest().upper()
    if data.SignatureValue.upper() == sign_encode:
        transaction.price = data.outSum
        transaction.tax = data.Fee
        transaction.eMail = data.EMail
        transaction.paymentMethod = data.PaymentMethod
        transaction.incCurrLabel = data.IncCurrLabel
        transaction.status = 'Оплачено'
        transaction.finished = True
        transaction.datetime = datetime.now()
        user.subscribed = True
        if transaction.name == '1 month':
            user.end_subscribe = date_month(1)
        elif transaction.name == '3 month':
            user.end_subscribe = date_month(3)
        elif transaction.name == '6 month':
            user.end_subscribe = date_month(6)
        elif transaction.name == '12 month':
            user.end_subscribe = date_month(12)
        elif transaction.name == "1 day":
            user.end_subscribe = datetime.now() + timedelta(days=1)
        print(user.end_subscribe, data.SignatureValue, sign_encode, "Информация")
        await db.commit()
        await db.refresh(transaction)
        await db.refresh(user)
        return Response(content=f"OK{transaction.id}", media_type='text/plain')

    return Response(content="error: bad signature", media_type='text/plain')