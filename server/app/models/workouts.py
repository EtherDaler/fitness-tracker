from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class Workout(Base):
    __tablename__ = "workouts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    efficiency = Column(Integer, nullable=False, default=0)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())

    user = relationship("User", back_populates="workouts")
    workout_exercises = relationship("WorkoutExercise", back_populates="workout")

    def __repr__(self):
        return f"Workout(id={self.id}, name='{self.name}', user_id={self.user_id})"


class DefaultWorkout(Base):
    __tablename__ = "workouts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    efficiency = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())

    workout_exercises = relationship("WorkoutExercise", back_populates="workout")

    def __repr__(self):
        return f"Workout(id={self.id}, name='{self.name}')"