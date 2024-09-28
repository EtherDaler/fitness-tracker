import enum
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    Date
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.core.database import Base


class ActivityLevel(enum.Enum):
    NOT_GIVEN = 0  # Value not given during registration
    BEGINNER = 1  # Doesn't practice much
    AVERAGE = 2  # Practice sports 1-3 timer per week
    ATHLETE = 3  # Practice sports 3-5 times per week
    # add more activity levels here


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, nullable=False, index=True, unique=True)
    name = Column(String, nullable=True, default="")
    hashed_password: str = Column(String, nullable=False)
    gender = Column(String, nullable=True, default="")
    height = Column(Integer, nullable=True, default=0)
    weight = Column(Integer, nullable=True, default=0)
    profile_picture_url = Column(String, nullable=True, default="")
    activity_level = Column(
        Integer, nullable=True, default=ActivityLevel.NOT_GIVEN.value
    )
    age = Column(Integer, nullable=True, default=0)
    desired_weight = Column(Integer, nullable=True, default=0)
    gender = Column(String, nullable=True, default="male")
    created_at = Column(DateTime(timezone=True), default=func.now())
    subscribed = Column(Boolean, default=False)
    end_subsctibe = Column(Date, nullable=True)
    workouts = relationship("Workout", back_populates="user")
