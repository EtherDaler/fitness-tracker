from sqlalchemy.sql import select
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import HTTPException, Depends, Request

from app.models.workouts import Workout
from app.core.database import get_db

async def workout_verify( 
    request: Request,
    db: AsyncSession = Depends(get_db)
):
    workout_id = request.path_params.get("workout_id")
    if id is None:
        raise HTTPException(status_code=404, detail="Тренировка, не пройденная в пути")
      
    try:
        workout_id = int(workout_id)
    except ValueError:
        raise HTTPException(status_code=401, detail="Недопустимый токен: workout_id не является целым числом")

    # noinspection PyTypeChecker
    query = select(Workout).where(Workout.id == workout_id)
    result = await db.execute(query)
    workout = result.scalar()
    if not workout:
        raise HTTPException(status_code=404, detail="Тренировка не найдена")
      
    return workout
