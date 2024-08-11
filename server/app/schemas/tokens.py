from pydantic import BaseModel


class TokenVerifyResponse(BaseModel):
    message: str
