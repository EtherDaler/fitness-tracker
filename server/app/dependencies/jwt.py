from sqlalchemy.sql import select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import HTTPException, Depends, Request
from fastapi_jwt import JwtAuthorizationCredentials

from app.models.users import User
from app.core.database import get_db
from app.config import access_security


async def jwt_verify(
        _: Request,
        credentials: JwtAuthorizationCredentials = Depends(access_security),
        db: AsyncSession = Depends(get_db)
) -> User:
    user_id = credentials.subject.get("user_id")
    if not user_id or user_id == "":
        raise HTTPException(status_code=401, detail="Недопустимый токен: в теме отсутствует user_id")

    # Check if user_id is integer
    try:
        user_id = int(user_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Недопустимый токен: user_id не является целым числом")

    # noinspection PyTypeChecker
    query = select(User).where(User.id == user_id)
    result = await db.execute(query)
    user = result.scalar()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return user

