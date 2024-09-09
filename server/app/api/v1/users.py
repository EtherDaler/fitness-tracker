import enum
import os
import uuid
import pyheif

from fastapi.encoders import jsonable_encoder
from sqlalchemy.sql import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi.responses import JSONResponse, FileResponse
from fastapi import APIRouter, Depends, UploadFile, File

from app.core.utils import compress_and_save_image
from app.dependencies.jwt import jwt_verify
from app.models.users import User
from app.core.database import get_db
from app.schemas import ErrorResponseSchema
from app.schemas.users import UserDataUpdateSchema, UserSchema, FileUploadResponseSchema, UserProfileResponse

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
    file_path = os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url) if user.profile_picture_url else None

    # Проверка существования файла
    if not file_path or not os.path.isfile(file_path):
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "Изображение профиля пользователя не найдено!"})
        )

    # Определение типа контента
    extension = os.path.splitext(user.profile_picture_url)[1].lower()
    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    content_type = content_types.get(extension, "application/octet-stream")

    return FileResponse(file_path, media_type=content_type)


@router.put(
    "/photo",
    response_description="Обновление фотографии пользователя",
    responses={
        200: {
            "model": FileUploadResponseSchema,
            "description": "Файл успешно загружен и сохранён"
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен"
        },
        409: {"model": ErrorResponseSchema, "description": "Недопустимый формат файла"}
    }
)
async def change_user_picture(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
    photo: UploadFile = File(...)
):
    # Проверка расширения файла
    extension = photo.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png", "heic", "dng", "heif"]:
        return JSONResponse(
            status_code=409,
            content=jsonable_encoder(
                {"detail": "Допустимые форматы: jpg, jpeg, png, heic, dng, heif!"}
            ),
        )

    # Удаление предыдущего изображения, если оно существует
    if user.profile_picture_url:
        old_file_path = os.path.join(UPLOAD_DIRECTORY, user.profile_picture_url)
        if os.path.isfile(old_file_path):
            os.remove(old_file_path)

    if extension == "heic" or extension == "heif":
        heif_file = pyheif.read(await photo.read())
        image = Image.frombytes(
            heif_file.mode,
            heif_file.size,
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride
        )
        image = image.resize((400, 400))
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        with open(file_path, "wb") as f:
            f.write(image)

    else:
        filename = f"{uuid.uuid4()}.{extension}"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)

        with open(file_path, "wb") as f:
            f.write(await photo.read())  # Чтение и запись содержимого загружаемого файла

    # Обновление данных пользователя
    user.profile_picture_url = filename
    query = update(User).where(User.id == user.id).values(profile_picture_url=filename)
    await db.execute(query)
    await db.commit()
    print(f"Фотография профиля обновлена для пользователя: {user.id}, файл: {filename}")

    return FileUploadResponseSchema(file_id=filename)
@router.post(
    "/photo",
    response_description="Загрузка новой фотографии пользователя",
    responses={
        201: {
            "model": FileUploadResponseSchema,
            "description": "Файл успешно загружен и сохранён"
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен"
        },
        409: {"model": ErrorResponseSchema, "description": "Недопустимый формат файла"},
        400: {"model": ErrorResponseSchema, "description": "У пользователя уже есть аватар"}
    }
)
async def upload_user_picture(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
    photo: UploadFile = File(...)
):
    # Проверяем, есть ли уже аватар
    if user.profile_picture_url:
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"detail": "У пользователя уже есть аватар!"})
        )

    # Проверка расширения файла
    extension = photo.filename.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png", "heic"]:
        return JSONResponse(
            status_code=409,
            content=jsonable_encoder(
                {"detail": "Допустимые форматы: jpg, jpeg, png, heic!"}
            ),
        )

    # Генерация уникального имени файла
    filename = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(UPLOAD_DIRECTORY, filename)

    # Обычное сохранение файла без сжатия
    with open(file_path, "wb") as f:
        f.write(await photo.read())

    # Обновление данных пользователя
    user.profile_picture_url = filename
    query = update(User).where(User.id == user.id).values(profile_picture_url=filename)
    await db.execute(query)
    await db.commit()
    print(f"Фотография профиля загружена для пользователя: {user.id}, файл: {filename}")

    return FileUploadResponseSchema(file_id=filename)


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(user: User = Depends(jwt_verify)):
    # Передаем информацию о профиле пользователя, включая наличие аватара
    return {
        "name": user.name,
        "profile_picture_url": user.profile_picture_url,
        "has_avatar": bool(user.profile_picture_url)  # True, если есть аватар
    }

