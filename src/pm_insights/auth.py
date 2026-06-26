from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from fastapi import Header, HTTPException

from pm_insights import db, settings

try:
    from passlib.context import CryptContext
except Exception:
    CryptContext = None

try:
    from jose import JWTError, jwt
except Exception:
    JWTError = Exception
    jwt = None


_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto") if CryptContext else None


def _b64_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def hash_password(password: str) -> str:
    if _pwd_context:
        try:
            return _pwd_context.hash(password)
        except Exception:
            pass

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"pbkdf2_sha256${_b64_encode(salt)}${_b64_encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    if _pwd_context and not password_hash.startswith("pbkdf2_sha256$"):
        try:
            return _pwd_context.verify(password, password_hash)
        except Exception:
            return False

    try:
        _, salt_raw, digest_raw = password_hash.split("$", 2)
        salt = _b64_decode(salt_raw)
        expected = _b64_decode(digest_raw)
    except ValueError:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return hmac.compare_digest(actual, expected)


def _fallback_encode(payload: dict[str, Any]) -> str:
    if settings.JWT_ALGORITHM != "HS256":
        raise HTTPException(500, "JWT fallback supports only HS256.")
    header = {"typ": "JWT", "alg": "HS256"}
    header_raw = _b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_raw = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_raw}.{payload_raw}".encode("ascii")
    signature = hmac.new(settings.JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_raw}.{payload_raw}.{_b64_encode(signature)}"


def _fallback_decode(token: str) -> dict[str, Any]:
    try:
        header_raw, payload_raw, signature_raw = token.split(".", 2)
    except ValueError as exc:
        raise HTTPException(401, "Invalid token.") from exc

    signing_input = f"{header_raw}.{payload_raw}".encode("ascii")
    expected = hmac.new(settings.JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64_decode(signature_raw)):
        raise HTTPException(401, "Invalid token.")

    payload = json.loads(_b64_decode(payload_raw).decode("utf-8"))
    exp = payload.get("exp")
    if exp is not None and int(exp) < int(time.time()):
        raise HTTPException(401, "Token expired.")
    return payload


def create_access_token(user: dict) -> str:
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "username": user["username"],
        "exp": int(time.time()) + settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }
    if jwt:
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return _fallback_encode(payload)


def decode_access_token(token: str) -> dict[str, Any]:
    if jwt:
        try:
            return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        except JWTError as exc:
            raise HTTPException(401, "Invalid token.") from exc
    return _fallback_decode(token)


def current_user_from_token(token: str) -> dict:
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(401, "Invalid token.")
    user = db.get_user_by_id(int(user_id))
    if not user or not user.get("is_active", True):
        raise HTTPException(401, "User is inactive or not found.")
    return user


def get_current_user(authorization: str | None = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated.")
    return current_user_from_token(authorization.split(" ", 1)[1].strip())


def maybe_current_user(authorization: str | None = Header(None)) -> dict | None:
    if not settings.REQUIRE_AUTH:
        return None
    return get_current_user(authorization)
