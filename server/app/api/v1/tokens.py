from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Security, Depends
from fastapi_jwt import JwtAuthorizationCredentials

from app.models.users import User
from app.dependencies.jwt import jwt_verify
from app.schemas import ErrorResponseSchema
from app.schemas.tokens import TokenVerifyResponse
from app.config import access_security

router = APIRouter(prefix="/tokens", tags=["Жетоны"])


@router.get(
    "/validate",
    description="Проверяет только подпись токена, а не содержимое внутри него.",
    responses={
        200: {"model": TokenVerifyResponse, "description": "Токен действителен"},
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
    },
)
async def validate_token(_: JwtAuthorizationCredentials = Security(access_security)):
    return JSONResponse(
        status_code=200, content=jsonable_encoder({"message": "Токен действителен"})
    )


@router.get(
    "/verify",
    description="Полностью проверьте токен, включая содержимое внутри него.",
    responses={
        200: {"model": TokenVerifyResponse, "description": "Token is valid"},
        401: {
            "model": ErrorResponseSchema,
            "description": "Token is invalid",
        },
        404: {"model": ErrorResponseSchema, "description": "User not found"},
    },
)
async def verify_token(_: User = Depends(jwt_verify)):
    return JSONResponse(
        status_code=200, content=jsonable_encoder({"message": "Токен действителен"})
    )
