from sqlalchemy import Column, Integer, DateTime, ForeignKey, String
from sqlalchemy.sql import func
from app.core.database import Base


class ResetPasswordToken(Base):
    __tablename__ = "reset_password_tokens"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, ForeignKey("users.email"), nullable=False)
    token = Column(String, nullable=False)
    expiration = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
