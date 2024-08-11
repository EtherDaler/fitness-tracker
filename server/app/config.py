import os

import aiosmtplib
from dotenv import load_dotenv
from fastapi_jwt import JwtAccessBearer

load_dotenv()

BASE_URL = os.environ.get("BASE_URL")

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")

SMTP_HOST = os.environ.get("SMTP_HOST")
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_PORT = os.environ.get("SMTP_PORT") or 587
SMTP_FROM = os.environ.get("SMTP_FROM") or SMTP_USER
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME") or SMTP_FROM

ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET") or "FITNESS_SECRET_TOKEN"
FORGOT_PASSWORD_TOKEN_EXPIRATION = (
    int(os.getenv("FORGOT_PASSWORD_TOKEN_EXPIRATION")) or 24
)

access_security = JwtAccessBearer(auto_error=True, secret_key=ACCESS_TOKEN_SECRET)

smtp = aiosmtplib.SMTP(hostname=SMTP_HOST, port=SMTP_PORT, start_tls=False)
