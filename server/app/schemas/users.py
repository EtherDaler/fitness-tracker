import enum
from typing import Optional

from pydantic import BaseModel

from app.models.users import ActivityLevel


class UserLoginSchema(BaseModel):
    email: str
    password: str


class UserRegisterSchema(BaseModel):
    email: str
    password: str
    name: str


class UserDataUpdateSchema(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    activity_level: Optional[ActivityLevel] = None
    password: Optional[str] = None
    age: Optional[int] = None
    desired_weight: Optional[int] = None

    # Throws an error if any extra fields are present
    class Config:
        extra = "forbid"


class UserSchema(BaseModel):
    email: str
    age: int
    gender: str
    name: str
    height: int
    weight: int
    desired_weight: int
    profile_picture_url: str
    activity_level: ActivityLevel | int


class UserLoginResponseSchema(BaseModel):
    email: str
    access_token: str
    name: str


class UserRegisterResponseSchema(BaseModel):
    email: str
    access_token: str
    name: str
    message: str


class FileUploadResponseSchema(BaseModel):
    file_id: str


class ForgotPasswordSchema(BaseModel):
    email: str


class ForgotPasswordResponseSchema(BaseModel):
    message: str
    token: str


class ResetUserPasswordSchema(BaseModel):
    new_password: str
    reset_token: str


class ResetUserPasswordResponseSchema(BaseModel):
    message: str
