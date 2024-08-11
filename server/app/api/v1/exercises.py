from sqlalchemy.sql import select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends

from app.dependencies.jwt import jwt_verify
from app.models.users import User
from app.models.exercises import Exercise
from app.core.database import get_db
from app.schemas import ErrorResponseSchema
from app.schemas.exercises import ExerciseSchema, AllExercisesResponse

router = APIRouter(prefix="/exercises", tags=["Упражнения"])


@router.get(
    "",  # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Выполните все упражнения",
    responses={
        200: {"model": AllExercisesResponse, "description": "Все упражнения"},
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не "
            "предоставлен",
        },
        404: {"model": ErrorResponseSchema, "description": "Пользователь не найден"},
    },
)
async def get_all_exercises(
    db: AsyncSession = Depends(get_db), _: User = Depends(jwt_verify)
):
    exercises: list[ExerciseSchema] = []

    query = select(Exercise)
    result = await db.execute(query)
    results = result.scalars().all()
    for exercise in results:
        exercises.append(
            ExerciseSchema(
                id=exercise.id,
                name=exercise.name,
                description=exercise.description,
                video_link=exercise.video_link,
                gif_link=exercise.gif_link,
                created_at=exercise.created_at,
            )
        )

    return AllExercisesResponse(exercises=exercises)
