import requests
import base64
import json
import hashlib
import hmac

from datetime import datetime

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
    new_transaction = Transactions(
        price=data.price,
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
    url = "https://services.robokassa.ru/InvoiceServiceWebApi/api/CreateInvoice"
    header = {"typ": "JWT", "alg": "MD5"}
    payload = {
        "MerchantLogin": MerchantLogin,
        "InvoiceType": "Reusable",
        "Culture": "ru", 
        "InvId": transaction_id,
        "OutSum": data.pirce,
        "Description": data.description,
        "MerchantComments": "no comment",
        "InvoiceItems": [
            {
              "Name": "Подписка",
              "Quantity": 1,
              "Cost": data.price,
              "Tax": "none",
              "PaymentMethod": "full_payment",
              "PaymentObject": "commodity"
            }	
        ]
    }

    header_encode = base64.urlsafe_b64encode(json.dumps(header).encode('utf-8')).decode()
    payload_encode = base64.urlsafe_b64encode(json.dumps(payload).encode('utf-8')).decode()
    key = f"{MerchantLogin}:{password1}"
    secret_encode = key.encode()
    body = header_encode + '.' + payload_encode
    body_encode = body.encode()
    signing = body + '.' + base64.b64encode(
        hmac.new(secret_encode, body_encode, hashlib.md5).hexdigest().encode()).decode()

    response = requests.post(
        url=url,
        json=json.dumps({'Header': header, 'Payload': payload, 'Signature': signing}),
        headers={'Content-type': 'application/json', 'Accept': 'application/json'}
    )
    return response.json()


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
    result = await db.execute(query)
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
            user.end_subsctibe = date_month(1)
        elif transaction.name == '3 month':
            user.end_subsctibe = date_month(3)
        elif transaction.name == '6 month':
            user.end_subsctibe = date_month(6)
        elif transaction.name == '12 month':
            user.end_subsctibe = date_month(12)
        await db.commit()
        await db.refresh(transaction)
        await db.refresh(user)
        return Response(content=f"OK{transaction.id}", media_type='text/plain')

    return Response(content="error: bad signature", media_type='text/plain')