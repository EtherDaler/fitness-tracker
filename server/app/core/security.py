import jwt

from passlib.context import CryptContext
from app.config import ACCESS_TOKEN_SECRET

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str):
    return pwd_context.hash(password)


def is_valid_jwt(token: str):
    try:
        return jwt.decode(token, ACCESS_TOKEN_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        return False
