from datetime import datetime

from pydantic import BaseModel

from app.schemas.exercises import ExerciseSchema
from app.models import Workout, WorkoutExercise, WorkoutSession


class WorkoutSessionSchema(BaseModel):
    id: int
    repetitions: int
    started_at: datetime
    finished_at: datetime

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class WorkoutExerciseSchema(BaseModel):
    id: int
    name: str
    exercise_id: str
    description: str
    video_link: str
    gif_link: str
    created_at: datetime
    repetitions: int
    rest_time: int

    sessions: list[WorkoutSessionSchema]

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class WorkoutSchema(BaseModel):
    id: int
    name: str
    description: str
    efficiency: int
    created_at: datetime

    exercises: list[WorkoutExerciseSchema]

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class AllWorkoutsSchema(BaseModel):
    workouts: list[WorkoutSchema]


class CreateWorkoutExerciseSchema(BaseModel):
    exercise_id: str
    repetitions: int
    rest_time: int = 0


class CreateWorkoutSessionSchema(BaseModel):
    workout_id: int
    workout_exercise_id: int
    repetitions: int
    started_at: datetime
    finished_at: datetime


class CreateWorkoutSchema(BaseModel):
    name: str
    description: str
    exercises: list[CreateWorkoutExerciseSchema]


class CreateWorkoutResponseSchema(BaseModel):
    message: str


class CreateWorkoutSessionResponseSchema(BaseModel):
    message: str


def workout_session_to_schema(workout_session: WorkoutSession):
    return WorkoutSessionSchema(
        id=workout_session.id,
        repetitions=workout_session.repetitions,
        started_at=workout_session.started_at,
        finished_at=workout_session.finished_at,
    )


def workout_exercise_to_schema(
    workout_exercise: WorkoutExercise,
) -> WorkoutExerciseSchema:
    sessions = [
        workout_session_to_schema(session)
        for session in workout_exercise.workout_sessions
    ]
    return WorkoutExerciseSchema(
        id=workout_exercise.id,
        exercise_id=workout_exercise.exercise_id,
        name=workout_exercise.exercise.name,
        description=workout_exercise.exercise.description,
        video_link=workout_exercise.exercise.video_link,
        gif_link=workout_exercise.exercise.gif_link,
        created_at=workout_exercise.created_at,
        sessions=sessions,
        repetitions=workout_exercise.repetitions,
        rest_time=workout_exercise.rest_time,
    )


def workout_to_schema(workout: Workout) -> WorkoutSchema:
    exercises = [
        workout_exercise_to_schema(workout_exercise)
        for workout_exercise in workout.workout_exercises
    ]
    return WorkoutSchema(
        id=workout.id,
        name=workout.name,
        description=workout.description,
        efficiency=workout.efficiency,
        created_at=workout.created_at,
        exercises=exercises,
    )
