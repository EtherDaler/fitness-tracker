from sqlalchemy.sql import select
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.dependencies.jwt import jwt_verify
from app.models.users import User
from app.models import Exercise, Workout, WorkoutExercise, WorkoutSession
from app.core.database import get_db
from app.schemas import ErrorResponseSchema
from app.schemas.workouts import (
    AllWorkoutsSchema,
    WorkoutSchema,
    CreateWorkoutSchema,
    CreateWorkoutResponseSchema,
    CreateWorkoutSessionSchema,
    CreateWorkoutSessionResponseSchema,
    workout_to_schema
)

router = APIRouter(prefix="/workouts", tags=["Тренировки"])


@router.get(
    "",
    # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Получает все тренировки, выполненные пользователями вместе с каждой сессией",
    responses={
        200: {"model": AllWorkoutsSchema, "description": "Все данные пользователя"},
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        404: {
            "model": ErrorResponseSchema,
            "description": "Пользователь или упражнение не найдены",
        },
    },
)
async def get_all_users_workouts(
    user: User = Depends(jwt_verify),
    db: AsyncSession = Depends(get_db),
):
    all_workouts: list[WorkoutSchema] = []

    # noinspection PyTypeChecker
    query = (
        select(Workout)
        .where(Workout.user_id == user.id)
        .options(
            joinedload(Workout.user),
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
    )
    default_query = (
        select(Workout)
        .where(Workout.efficiency == user.activity_level, Workout.user_id.is_(None))
        .options(
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.user),
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        ),
    )
    result = await db.execute(query)
    default_result = await db.execute(default_query)
    workouts = result.unique().scalars().all()
    default_workouts = default_result.unique().scalars().all()
    print(default_workouts)
    if not workouts and not default_workouts:
        return AllWorkoutsSchema(workouts=[])
    if workouts:
        for workout in workouts:
            workout.workout_exercises.sort(key=lambda we: we.id)
            all_workouts.append(workout_to_schema(workout))
    if default_workouts:
        for workout in default_workouts:
            workout.workout_exercises.sort(key=lambda we: we.id)
            all_workouts.append(workout_to_schema(workout))

    return AllWorkoutsSchema(workouts=all_workouts)


@router.get(
    "/{workout_id}",
    response_description="Получите одну тренировку по ее идентификатору",
    responses={
        200: {
            "model": WorkoutSchema,
            "description": "Подробная схема тренировки",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        404: {
            "model": ErrorResponseSchema,
            "description": "Пользователь или упражнение не найдены",
        },
    },
)
async def get_single_workout(
    workout_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
):
    # noinspection PyTypeChecker
    query = (
        select(Workout)
        .where(Workout.id == workout_id, Workout.user_id.is_(None))
        .options(
            joinedload(Workout.user),
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
    )
    default_query = query = (
        select(Workout)
        .where(Workout.id == workout_id, Workout.user_id.is_(None))
        .options(
            joinedload(Workout.user),
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
    )
    result = await db.execute(query)
    workout = result.scalars().first()

    default_result = await db.execute(default_query)
    default_workout = default_result.scalars().first()

    if not workout and not default_workout:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "Тренировка не найдена"}),
        )

    if workout:
        workout.workout_exercises.sort(key=lambda we: we.id)
    elif default_workout:
        workout = default_workout.workout_exercises.sort(key=lambda we: we.id)
    return workout_to_schema(workout)


@router.post(
    "",  # You might be temped to add a '/' here, but don't do it because it redirects traffics. Apparently FastAPI and
    # NGINX redrections are clashing with each other.
    response_description="Создайте новую тренировку для пользователя",
    responses={
        200: {
            "model": CreateWorkoutResponseSchema,
            "description": "Создана новая тренировка",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        404: {
            "model": ErrorResponseSchema,
            "description": "Пользователь или упражнение не найдены",
        },
    },
)
async def create_new_workout(
    data: CreateWorkoutSchema,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
):
    for ex in data.exercises:
        query = select(Exercise).where(Exercise.id == ex.exercise_id)
        result = await db.execute(query)
        exercise = result.scalars().first()

        if not exercise:
            return JSONResponse(
                status_code=404,
                content=jsonable_encoder(
                    {"detail": f"Exercise with id '{ex.exercise_id}' not found"}
                ),
            )

    new_workout = Workout(user_id=user.id, name=data.name, description=data.description)
    db.add(new_workout)
    await db.commit()
    await db.refresh(new_workout)

    for ex in data.exercises:
        new_workout_exercise = WorkoutExercise(
            workout_id=new_workout.id,
            exercise_id=ex.exercise_id,
            repetitions=ex.repetitions,
            rest_time=ex.rest_time,
        )
        db.add(new_workout_exercise)

    await db.commit()

    return CreateWorkoutResponseSchema(
        message=f"Created workout with id {new_workout.id}"
    )


@router.post(
    "/sessions",
    response_description="Добавьте новый тренировочный сеанс к тренировке",
    responses={
        200: {
            "model": CreateWorkoutSessionResponseSchema,
            "description": "Тренировка успешно добавлена",
        },
        401: {
            "model": ErrorResponseSchema,
            "description": "Токен недействителен, срок действия истек или не предоставлен",
        },
        404: {
            "model": ErrorResponseSchema,
            "description": "Пользователь или тренировка не найдены",
        },
    },
)
async def add_new_workout_session(
    data: CreateWorkoutSessionSchema,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(jwt_verify),
):
    query = select(Workout).where(
        Workout.id == data.workout_id, Workout.user_id == user.id
    )
    result = await db.execute(query)
    workout = result.scalars().first()
    default_query = select(Workout).where(
        Workout.id == data.workout_id, Workout.user_id.is_(None)
    )
    default_result = await db.execute(default_query)
    default_workout = default_result.scalars().first()

    if not workout:
        workout = default_workout
    if not workout:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "Workout not found"}),
        )
    query = select(WorkoutExercise).where(
        WorkoutExercise.id == data.workout_exercise_id
    )
    result = await db.execute(query)
    workout_exercise = result.scalars().first()

    if not workout_exercise:
        return JSONResponse(
            status_code=404,
            content=jsonable_encoder({"detail": "Workout Exercise not found"}),
        )

    new_workout_session = WorkoutSession(
        user_id=user.id,
        workout_exercise_id=data.workout_exercise_id,
        repetitions=data.repetitions,
        started_at=data.started_at,
        finished_at=data.finished_at,
    )

    db.add(new_workout_session)
    await db.commit()
    await db.refresh(new_workout_session)

    return CreateWorkoutSessionResponseSchema(message="Тренировка успешно завершена.")
