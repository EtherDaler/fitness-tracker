from datetime import datetime

from pydantic import BaseModel


class ExerciseSchema(BaseModel):
    id: str
    name: str
    description: str
    video_link: str
    gif_link: str
    description: str
    created_at: datetime

    class Config:
        json_encoders = {datetime: lambda dt: dt.isoformat()}


class AllExercisesResponse(BaseModel):
    exercises: list[ExerciseSchema]
