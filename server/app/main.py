import json
import asyncio
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select
from sqlalchemy.orm import joinedload

# from app.core.emails import init_smtp
from app.schemas.workouts import workout_to_schema
from app.core.utils import insert_default_data
from app.core.security import is_valid_jwt
from app.models import Workout, WorkoutExercise, User, Exercise, ResetPasswordToken

from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.tokens import router as tokens_router
from app.api.v1.exercises import router as exercises_router
from app.api.v1.websockets import router as websockets_router
from app.api.v1.workouts import router as workouts_router

from app.core.database import create_tables, drop_tables, get_db
from app.schemas.users import UserSchema
from app.schemas.exercises import ExerciseSchema
from app.schemas.workouts import WorkoutSchema

logging.getLogger("passlib").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warning: This will drop all tables, so use only in dev
    # asyncio.create_task(init_smtp())
    # asyncio.create_task(drop_tables())
    asyncio.create_task(create_tables())
    asyncio.create_task(insert_default_data())

    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index(request: Request):
    # return templates.TemplateResponse("index.html", {"request": request})
    return RedirectResponse("/login")


@app.get("/register")
def register_page(request: Request):
    access_token = request.cookies.get("access_token")
    if access_token and is_valid_jwt(access_token):
        return RedirectResponse(url="/dashboard")

    response = templates.TemplateResponse(
        "register.html",
        {
            "request": request,
        },
    )
    if access_token:
        response.delete_cookie("access_token")

    return response


@app.get("/login")
def login_page(request: Request):
    access_token = request.cookies.get("access_token")

    if access_token and is_valid_jwt(access_token):
        return RedirectResponse(url="/dashboard")

    response = templates.TemplateResponse(
        "login.html",
        {
            "request": request,
        },
    )
    if access_token:
        response.delete_cookie("access_token")

    return response


@app.get("/forget-password")
def forget_password_page(request: Request):
    access_token = request.cookies.get("access_token")

    if access_token and is_valid_jwt(access_token):
        return RedirectResponse(url="/dashboard")

    response = templates.TemplateResponse(
        "forget-password.html",
        {
            "request": request,
        },
    )
    if access_token:
        response.delete_cookie("access_token")

    return response


@app.get("/reset-password")
async def reset_password_page(
        request: Request, token: str, db: AsyncSession = Depends(get_db)
):
    access_token = request.cookies.get("access_token")

    if access_token and is_valid_jwt(access_token):
        return RedirectResponse(url="/dashboard")

    query = select(ResetPasswordToken).where(ResetPasswordToken.token == token)
    result = await db.execute(query)
    tkn = result.scalars().first()

    if not tkn:
        response = templates.TemplateResponse(
            "reset-password.html",
            {
                "request": request,
                "valid": False,
                "message": "Reset token not found",
            },
        )
        if access_token:
            response.delete_cookie("access_token")
        return response

    if tkn.expiration < datetime.now(timezone.utc):
        response = templates.TemplateResponse(
            "reset-password.html",
            {
                "request": request,
                "valid": False,
                "message": "Reset token expired",
            },
        )

    query = select(User).where(User.email == tkn.email)
    result = await db.execute(query)
    user = result.scalars().first()

    if not user:
        response = templates.TemplateResponse(
            "reset-password.html",
            {
                "request": request,
                "valid": False,
                "message": "User assaigned with that token not found",
            },
        )

    response = templates.TemplateResponse(
        "reset-password.html",
        {
            "request": request,
            "valid": True,
            "message": "",
        },
    )
    if access_token:
        response.delete_cookie("access_token")

    return response


@app.get("/dashboard")
async def dashboard_page(request: Request, db: AsyncSession = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    query = select(User).where(User.id == token["subject"]["user_id"])
    result = await db.execute(query)
    user = result.scalar()

    query = (
        select(Workout)
        .where(Workout.user_id == token["subject"]["user_id"])
        .options(
            joinedload(Workout.user),
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
        .order_by(Workout.created_at.desc())
    )

    default_query = (
        select(Workout)
        .where(Workout.user_id is None, Workout.efficiency == user.activity_level)
        .options(
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
        .order_by(Workout.created_at.desc())
    )

    workouts = []
    result = await db.execute(query)
    data = result.unique().scalars().all()
    default_result = await db.execute(default_query)
    default_data = default_result.unique().scalars().all()
    data += default_data

    for workout in data:
        workout.workout_exercises.sort(key=lambda we: we.id)
        workouts.append(json.loads(workout_to_schema(workout).model_dump_json()))

    query = select(User).where(User.id == token["subject"]["user_id"])
    result = await db.execute(query)
    user = result.scalar()

    if not user:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "access_token": access_token,
            "workouts": workouts,
            "user": UserSchema(
                email=user.email,
                name=user.name,
                gender=user.gender,
                height=user.height,
                weight=user.weight,
                activity_level=user.activity_level,
                profile_picture_url=user.profile_picture_url or "",
                age=user.age,
                desired_weight=user.desired_weight,
            ),
        },
    )


@app.get("/profile")
async def profile_page(request: Request, db: AsyncSession = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    query = select(User).where(User.id == token["subject"]["user_id"])
    result = await db.execute(query)
    user = result.scalar()

    if not user:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    return templates.TemplateResponse(
        "profile.html",
        {
            "request": request,
            "access_token": access_token,
            "user": UserSchema(
                email=user.email,
                name=user.name,
                gender=user.gender,
                height=user.height,
                weight=user.weight,
                activity_level=user.activity_level,
                profile_picture_url=user.profile_picture_url or "",
                age=user.age,
                desired_weight=user.desired_weight,
            ),
        },
    )


# @app.get("/workouts/")
# def new_workout(request: Request):
#     access_token = request.cookies.get("access_token", None)
#     return templates.TemplateResponse(
#         "new-workout.html", {"request": request, "access_token": access_token}
#     )


@app.get("/workouts/new")
async def new_workout(request: Request, db: AsyncSession = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    exercises: list[ExerciseSchema] = []

    query = select(Exercise)
    result = await db.execute(query)
    results = result.scalars().all()
    for exercise in results:
        exercises.append(
            json.loads(
                ExerciseSchema(
                    id=exercise.id,
                    name=exercise.name,
                    description=exercise.description,
                    video_link=exercise.video_link,
                    gif_link=exercise.gif_link,
                    created_at=exercise.created_at,
                ).model_dump_json()
            )
        )

    return templates.TemplateResponse(
        "add-workout.html",
        {
            "request": request,
            "access_token": access_token,
            "exercises": exercises,
        },
    )


@app.get("/workouts/edit")
async def new_workout(request: Request, db: AsyncSession = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    query = select(User).where(User.id == token["subject"]["user_id"])
    result = await db.execute(query)
    user = result.scalar()

    query = (
        select(Workout)
        .where(Workout.user_id == token["subject"]["user_id"])
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
        .where(Workout.user_id is None, Workout.efficiency == user.activity_level)
        .options(
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
    )

    workouts = []
    result = await db.execute(query)
    data = result.unique().scalars().all()
    default_result = await db.execute(default_query)
    default_data = default_result.unique().scalars().all()
    data += default_data
    for workout in data:
        workout.workout_exercises.sort(key=lambda we: we.id)
        workouts.append(json.loads(workout_to_schema(workout).model_dump_json()))

    return templates.TemplateResponse(
        "edit-workouts.html",
        {
            "request": request,
            "access_token": access_token,
            "workouts": workouts,
        },
    )


@app.get("/workouts")
async def workouts_page(request: Request, db: AsyncSession = Depends(get_db)):
    access_token = request.cookies.get("access_token")
    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    query = select(User).where(User.id == token["subject"]["user_id"])
    result = await db.execute(query)
    user = result.scalar()

    query = (
        select(Workout)
        .where(Workout.user_id == token["subject"]["user_id"])
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
        .where(Workout.user_id is None, Workout.efficiency == user.activity_level)
        .options(
            joinedload(Workout.workout_exercises).joinedload(WorkoutExercise.exercise),
            joinedload(Workout.workout_exercises).joinedload(
                WorkoutExercise.workout_sessions
            ),
        )
    )

    workouts = []
    result = await db.execute(query)
    data = result.unique().scalars().all()
    default_result = await db.execute(default_query)
    default_data = default_result.unique().scalars().all()
    data += default_data

    for workout in data:
        workout.workout_exercises.sort(key=lambda we: we.id)
        workouts.append(json.loads(workout_to_schema(workout).model_dump_json()))

    return templates.TemplateResponse(
        "workouts.html",
        {
            "request": request,
            "access_token": access_token,
            "workouts": workouts,
        },
    )


@app.get("/workouts/{idx}/start")
async def start_workout(
        request: Request,
        idx: int | str,
        _type: str = "workout",
        duration: int = 0,
        db: AsyncSession = Depends(get_db),
):
    access_token = request.cookies.get("access_token")
    workout = None
    exercise = None

    if not access_token:
        return RedirectResponse(url="/login")

    token = is_valid_jwt(access_token)

    if not token or token.get("subject", {}).get("user_id") is None:
        resp = RedirectResponse(url="/login")
        resp.delete_cookie("access_token")
        return resp

    if _type and _type not in ["workout", "exercise"]:
        return templates.TemplateResponse(
            "workout-start.html",
            {
                "request": request,
                "error": "Invalid type of workout or exercise",
            },
        )

    if _type == "exercise":
        if duration <= 0:
            return templates.TemplateResponse(
                "workout-start.html",
                {
                    "request": request,
                    "error": "Invalid duration for exercise",
                },
            )

        query = select(Exercise).where(Exercise.id == idx)
        result = await db.execute(query)
        exercise = result.scalar()

    if _type == "workout":
        if not idx.isdigit():
            return templates.TemplateResponse(
                "workout-start.html",
                {
                    "request": request,
                    "error": "Invalid workout id",
                },
            )

        query = (
            select(Workout)
            .where(
                Workout.user_id == token["subject"]["user_id"], Workout.id == int(idx)
            )
            .options(
                joinedload(Workout.user),
                joinedload(Workout.workout_exercises).joinedload(
                    WorkoutExercise.exercise
                ),
                joinedload(Workout.workout_exercises).joinedload(
                    WorkoutExercise.workout_sessions
                ),
            )
        )
        result = await db.execute(query)
        workout = result.scalar()

        default_query = (
            select(Workout)
            .where(
                Workout.user_id is None, Workout.id == int(idx)
            )
            .options(
                joinedload(Workout.workout_exercises).joinedload(
                    WorkoutExercise.exercise
                ),
                joinedload(Workout.workout_exercises).joinedload(
                    WorkoutExercise.workout_sessions
                ),
            )
        )
        default_result = await db.execute(default_query)
        default_workout = default_result.scalar()

        if not workout:
            workout = default_workout
        if not workout:
            return templates.TemplateResponse(
                "workout-start.html",
                {
                    "request": request,
                    "error": "Workout not found",
                },
            )

    if not workout and not exercise:
        return templates.TemplateResponse(
            "workout-start.html",
            {
                "request": request,
                "error": "Workout or exercise not found",
            },
        )

    if workout:
        workout.workout_exercises.sort(key=lambda we: we.id)
        workout = json.loads(workout_to_schema(workout).model_dump_json())
        if len(workout["exercises"]) == 0:
            return templates.TemplateResponse(
                "workout-start.html",
                {
                    "request": request,
                    "error": "No exercises found in workout",
                },
            )

    if exercise:
        exercise = json.loads(
            ExerciseSchema(
                id=exercise.id,
                name=exercise.name,
                description=exercise.description,
                video_link=exercise.video_link,
                gif_link=exercise.gif_link,
                created_at=exercise.created_at,
            ).model_dump_json()
        )

    return templates.TemplateResponse(
        "workout-start.html",
        {
            "request": request,
            "idx": idx,
            "type": _type,
            "duration": duration,
            "workout": workout,
            "exercise": exercise,
            "error": None,
        },
    )


app.include_router(auth_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(tokens_router, prefix="/api/v1")
app.include_router(exercises_router, prefix="/api/v1")
app.include_router(websockets_router, prefix="/api/v1")
app.include_router(workouts_router, prefix="/api/v1")
