"""JWT 认证模块"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt
import structlog

logger = structlog.get_logger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET", "wyf-agent-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24

DEMO_USERS: dict[str, dict[str, str]] = {
    "admin": {"password": "admin123", "role": "admin"},
    "user": {"password": "user123", "role": "user"},
}


@dataclass
class AuthResult:
    success: bool
    token: str = ""
    username: str = ""
    error: str = ""


def authenticate(username: str, password: str) -> AuthResult:
    if username not in DEMO_USERS or DEMO_USERS[username]["password"] != password:
        logger.warning("auth_failed", username=username)
        return AuthResult(success=False, error="Invalid username or password")

    role = DEMO_USERS[username]["role"]
    payload = {
        "sub": username,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_HOURS * 3600,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info("auth_success", username=username, role=role)
    return AuthResult(success=True, token=token, username=username)


def verify_token(token: str) -> AuthResult:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub", "")
        return AuthResult(success=True, username=username, token=token)
    except jwt.ExpiredSignatureError:
        return AuthResult(success=False, error="Token expired")
    except jwt.InvalidTokenError as e:
        return AuthResult(success=False, error=f"Invalid token: {e}")


def get_user_role(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("role", "user")
    except jwt.InvalidTokenError:
        return "user"
