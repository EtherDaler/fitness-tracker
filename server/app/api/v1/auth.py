import uuid
from datetime import datetime, timedelta, timezone

from aiosmtplib.errors import SMTPException
from smtplib import SMTPException as SyncSMTPException

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.models import User, ResetPasswordToken
from app.core.database import get_db
from app.dependencies.jwt import jwt_verify
from app.config import FORGOT_PASSWORD_TOKEN_EXPIRATION
from app.schemas import ErrorResponseSchema
from app.schemas.users import (
    UserLoginSchema,
    UserRegisterSchema,
    UserRegisterResponseSchema,
    UserLoginResponseSchema,
    ForgotPasswordSchema,
    ForgotPasswordResponseSchema,
    ResetUserPasswordSchema,
    ResetUserPasswordResponseSchema,
)
from app.core.utils import insert_generated_workouts
from app.core.emails import (
    read_email_template,
    send_mail_sync,
    # send_mail_async,
)
from app.core.security import verify_password, get_password_hash
from app.config import access_security, BASE_URL

router = APIRouter(prefix="/auth", tags=["Авторизация"])


@router.post(
    "/login",
    response_description="Войдите в систему с помощью электронной почты и пароля",
    responses={
        200: {
            "model": UserLoginResponseSchema,
            "description": "Успешный вход пользователя в систему",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Неверный адрес электронной почты или пароль",
        },
    },
)
async def login(data: UserLoginSchema, db: AsyncSession = Depends(get_db)):
    # noinspection PyTypeChecker
    query = select(User).where(User.email == data.email)
    result = await db.execute(query)
    user = result.scalars().first()

    if not user or not verify_password(data.password, user.hashed_password):
        return JSONResponse(
            status_code=401,
            content=jsonable_encoder(
                {"detail": "Неверный адрес электронной почты или пароль"}
            ),
        )

    subject = {"email": user.email, "user_id": user.id}

    access_token = access_security.create_access_token(
        subject=subject,
        expires_delta=timedelta(days=365 * 100),
    )

    return UserLoginResponseSchema(
        name=user.name, email=user.email, access_token=access_token
    )


@router.post(
    "/register",
    response_description="Зарегистрируйте пользователя по электронной почте и отправьте временный пароль на это "
    "электронное письмо",
    responses={
        200: {
            "model": UserRegisterResponseSchema,
            "description": "Регистрация прошла успешно, пароль отправлен на электронную почту",
        },
        409: {
            "model": ErrorResponseSchema,
            "description": "Электронная почта уже существует",
        },
    },
)
async def register(data: UserRegisterSchema, db: AsyncSession = Depends(get_db)):
    # noinspection PyTypeChecker
    query = select(User).where(User.email == data.email)
    result = await db.execute(query)
    user = result.scalars().first()

    if user:
        return JSONResponse(
            status_code=409,
            content=jsonable_encoder({"detail": "Электронная почта уже существует"}),
        )

    try:
        hashed_password = get_password_hash(data.password)
        user = User(email=data.email, name=data.name, hashed_password=hashed_password)
        db.add(user)
        await db.commit()
        await db.refresh(user)

        await insert_generated_workouts(user)

        subject = {"email": user.email, "user_id": user.id}

        access_token = access_security.create_access_token(
            subject=subject,
            expires_delta=timedelta(days=365 * 100),
        )

        # add some default exercises

        return UserRegisterResponseSchema(
            name=user.name,
            email=user.email,
            access_token=access_token,
            message="User registered successfully",
        )

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {
                    "detail": "Сообщение об ошибке при отправке на этот адрес электронной почты!"
                }
            ),
        )


@router.post(
    "/forget-password",
    response_description="Send email with temporary token to reset password",
    responses={
        200: {
            "model": UserRegisterResponseSchema,
            "description": "Регистрация прошла успешно, пароль отправлен на электронную почту",
        },
        409: {
            "model": ErrorResponseSchema,
            "description": "Электронная почта уже существует",
        },
    },
)
async def forgot_password(
    data: ForgotPasswordSchema, db: AsyncSession = Depends(get_db)
):
    query = select(User).where(User.email == data.email)
    result = await db.execute(query)
    user = result.scalars().first()

    if not user:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "User with provided email not found."}),
        )

    reset_token = str(uuid.uuid4())
    expiration = datetime.now(timezone.utc) + timedelta(
        hours=FORGOT_PASSWORD_TOKEN_EXPIRATION
    )

    html = read_email_template("reset-password.html")
    html = html.replace("{{link}}", f"{BASE_URL}/reset-password?token={reset_token}")

    try:
        send_mail_sync(user.email, "Password Reset", html)

    except SyncSMTPException as e:
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder(
                {"detail": "Error sending email to this email address!"}
            ),
        )

    token = ResetPasswordToken(
        email=user.email, token=reset_token, expiration=expiration
    )
    db.add(token)
    await db.commit()

    return ForgotPasswordResponseSchema(
        message="Password reset token sent successfully", token=reset_token
    )
    # Send email with reset token


@router.post(
    "/reset-password",
    response_description="Resets the password of the user using the temporary token",
    responses={
        200: {
            "model": UserRegisterResponseSchema,
            "description": "Регистрация прошла успешно, пароль отправлен на электронную почту",
        },
        409: {
            "model": ErrorResponseSchema,
            "description": "Электронная почта уже существует",
        },
    },
)
async def reset_password(
    data: ResetUserPasswordSchema, db: AsyncSession = Depends(get_db)
):
    query = select(ResetPasswordToken).where(
        ResetPasswordToken.token == data.reset_token
    )
    result = await db.execute(query)
    token = result.scalars().first()

    if not token:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "Reset token not found"}),
        )

    if token.expiration < datetime.now(timezone.utc):
        return JSONResponse(
            status_code=409,
            content=jsonable_encoder({"detail": "Reset token expired"}),
        )

    query = select(User).where(User.email == token.email)
    result = await db.execute(query)
    user = result.scalars().first()

    if not user:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder(
                {"detail": "User assaigned with that token not found"}
            ),
        )

    user.hashed_password = get_password_hash(data.new_password)
    await db.commit()
    await db.refresh(user)

    await db.delete(token)
    await db.commit()

    return ResetUserPasswordResponseSchema(message="Password reset successfully")
