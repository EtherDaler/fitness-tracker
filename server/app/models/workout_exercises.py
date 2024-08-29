from sqlalchemy import Column, Integer, DateTime, ForeignKey, String
from sqlalchemy.sql import func
from app.core.database import Base
from sqlalchemy.orm import relationship


class WorkoutExercise(Base):
    __tablename__ = "workout_exercises"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workout_id = Column(Integer, ForeignKey("workouts.id"), nullable=False)
    exercise_id = Column(String, ForeignKey("exercises.id"), nullable=False)
    repetitions = Column(Integer, nullable=False)
    rest_time = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())

    workout = relationship("Workout", back_populates="workout_exercises")
    exercise = relationship("Exercise")
    workout_sessions = relationship("WorkoutSession", back_populates="workout_exercise")