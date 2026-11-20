"""Cryptographically signed JWT authentication and authorization dependencies.

Verifies digital signature, expiration timestamps, and subject identity claims.
Derives user_id exclusively from verified JWT principals.
"""

import datetime
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
import jwt

from app.config import settings

router = APIRouter(prefix="/auth", tags=["Authentication"])


class AuthenticatedUser:
    def __init__(self, user_id: str, email: Optional[str] = None):
        self.user_id = user_id
        self.email = email or f"{user_id}@example.com"


class TokenRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique user identifier for token subject")
    email: Optional[str] = Field(None, description="Optional email address")
    expires_in_minutes: Optional[int] = Field(120, ge=1, le=10080, description="Token validity window in minutes")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_at: str
    user_id: str


def create_access_token(
    user_id: str,
    email: Optional[str] = None,
    expires_delta: Optional[datetime.timedelta] = None,
) -> str:
    """Issue a cryptographically signed HMAC-SHA256 JWT."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + datetime.timedelta(minutes=settings.jwt_expiration_minutes)

    payload = {
        "sub": str(user_id),
        "email": email or f"{user_id}@example.com",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": "ai-career-agent",
    }
    return jwt.encode(payload, settings.api_auth_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict:
    """Verify signature, issuer, and expiration of a JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.api_auth_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "sub"]},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please obtain a new token.",
        )
    except jwt.InvalidSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature. Token has been tampered with or forged.",
        )
    except jwt.InvalidTokenError as ite:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(ite)}",
        )


async def get_current_user(
    authorization: Optional[str] = Header(None, description="Bearer <JWT> token"),
) -> AuthenticatedUser:
    """Derive user identity exclusively from cryptographically verified Bearer JWT."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing or malformed 'Authorization: Bearer <token>' header.",
        )

    raw_token = authorization.split("Bearer ", 1)[1].strip()
    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token string is empty.",
        )

    claims = verify_token(raw_token)
    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload: missing subject ('sub') claim.",
        )

    return AuthenticatedUser(user_id=user_id, email=claims.get("email"))


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Generate signed JWT token for authentication",
    status_code=status.HTTP_200_OK,
)
async def generate_token_endpoint(request: TokenRequest):
    """Generates signed JWT token using the configured secret key."""
    delta = datetime.timedelta(minutes=request.expires_in_minutes or 120)
    token = create_access_token(user_id=request.user_id, email=request.email, expires_delta=delta)
    exp_iso = (datetime.datetime.now(datetime.timezone.utc) + delta).isoformat()
    return TokenResponse(
        access_token=token,
        token_type="Bearer",
        expires_at=exp_iso,
        user_id=request.user_id,
    )
