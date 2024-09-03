import enum
import os
import uuid

from fastapi.encoders import jsonable_encoder
from sqlalchemy.sql import select
from sqlalchemy.ext.asyncio import AsyncSession

from starlette.responses import JSONResponse, FileResponse
from fastapi import APIRouter, Depends, UploadFile, File

from app.core.utils import compress_and_save_image
from app.dependencies.jwt import jwt_verify
from app.models.users import User
from app.core.database import get_db
from app.schemas import ErrorResponseSchema
from app.schemas.users import UserDataUpdateSchema, UserSchema, FileUploadResponseSchema

router = APIRouter(prefix="/users", tags=["Пользователи"])
UPLOAD_DIRECTORY = "app/static/uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@router.get(
    "/me",
    response_description="Получает данные пользователя",
    responses={
        200: {"model": UserSchema, "description": "Все данные пользователя"},
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        404: {"model": ErrorResponseSchema, "description": "Пользователь не найден"},
    },
)
async def get_user_data(user: User = Depends(jwt_verify)):
    return UserSchema(
        email=user.email,
        name=user.name,
        gender=user.gender,
        height=user.height,
        weight=user.weight,
        activity_level=user.activity_level,
        profile_picture_url=user.profile_picture_url or "",
        age=user.age,
        desired_weight=user.desired_weight,
    )


@router.put(
    "",
    # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Обновляет данные пользователя",
    responses={
        200: {
            "model": UserSchema,
            "description": "Все обновленные данные пользователя",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        409: {
            "model": ErrorResponseSchema,
            "description": "Номер телефона уже существует!",
        },
        404: {"model": ErrorResponseSchema, "description": "Пользователь не найден"},
    },
)
async def change_user_data(
    data: UserDataUpdateSchema,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
):
    # noinspection PyTypeChecker
    query = select(User).where(User.id == user.id)
    result = await db.execute(query)
    new_user = result.scalar()

    update_data = data.model_dump(exclude_none=True)
    for key, value in update_data.items():
        if isinstance(value, enum.Enum):
            value = value.value
        setattr(new_user, key, value)

    await db.commit()
    await db.refresh(new_user)

    return UserSchema(
        email=new_user.email,
        name=new_user.name,
        gender=new_user.gender,
        height=new_user.height,
        weight=new_user.weight,
        activity_level=new_user.activity_level,
        age=new_user.age,
        desired_weight=new_user.desired_weight,
        profile_picture_url=new_user.profile_picture_url,
    )


@router.get(
    "/photo",
    response_description="Получает фото пользователя",
    response_class=FileResponse,
    responses={
        200: {"description": "Successfully uploaded and saved a file"},
        404: {
            "model": ErrorResponseSchema,
            "description": "User does not have profile picture or User does not exists",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
    },
)
async def get_user_picture(user: User = Depends(jwt_verify)):
    if not user.profile_picture_url or not os.path.exists(
        os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url)
    ):
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder(
                {"detail": "User profile picture does not exists!"}
            ),
        )

    file_path = os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url)

    content_type = "application/octet-stream"
    if user.profile_picture_url.endswith(".jpg") or user.profile_picture_url.endswith(
        ".jpeg"
    ):
        content_type = "image/jpeg"
    elif user.profile_picture_url.endswith(".png"):
        content_type = "image/png"
    elif user.profile_picture_url.endswith(".gif"):
        content_type = "image/gif"
    elif user.profile_picture_url.endswith(".HEIC"):
        content_type = "image/HEIC"

    return FileResponse(file_path, media_type=content_type)


@router.put(
    "/photo",
    response_description="Обновляет фото пользователя",
    responses={
        200: {
            "model": FileUploadResponseSchema,
            "description": "Successfully uploaded and saved a file",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        409: {"model": ErrorResponseSchema, "description": "Invalid file type"},
    },
)
async def change_user_picture(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
    photo: UploadFile = File(...),
):
    extension = photo.filename.split(".")[-1]
    if extension not in ["jpg", "jpeg", "png"]:
        return JSONResponse(
            status_code=409,
            content=jsonable_encoder(
                {"detail": "Valid extensions are jpg, jpeg, png only!"}
            ),
        )

    filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)

    await compress_and_save_image(photo, file_path)

    if user.profile_picture_url and os.path.exists(
        os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url)
    ):
        os.remove(os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url))

    user.profile_picture_url = filename
    await db.commit()
    await db.refresh(user)

    return FileUploadResponseSchema(file_id=filename)
