from sqlalchemy import (
    Column,
    String,
    DateTime,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Exercise(Base):
    __tablename__ = "exercises"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    video_link = Column(String, nullable=False)
    gif_link = Column(String, nullable=False)
    description = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())

    workout_exercises = relationship("WorkoutExercise", back_populates="exercise")
