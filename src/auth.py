from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


PBKDF2_ITERS = int(os.getenv("PBKDF2_ITERS") or "160000")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    if salt is None:
        salt = secrets.token_bytes(16)
    pwd = password.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pwd, salt, PBKDF2_ITERS, dklen=32)
    return dk, salt


def verify_password(password: str, expected_hash: bytes, salt: bytes) -> bool:
    got, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(got, expected_hash)


def _jwt_signing_key() -> bytes:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        raise RuntimeError("缺少环境变量 JWT_SECRET（多商家登录必须设置）")
    return secret.encode("utf-8")


def jwt_encode(payload: Dict[str, Any], exp_seconds: int = 7 * 24 * 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = dict(payload)
    payload.setdefault("iat", now)
    payload.setdefault("exp", now + exp_seconds)

    signing_input = (
        f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))}."
        f"{_b64url_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))}"
    ).encode("ascii")

    sig = hmac.new(_jwt_signing_key(), signing_input, hashlib.sha256).digest()
    return signing_input.decode("ascii") + "." + _b64url_encode(sig)


def jwt_decode(token: str) -> Dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("token 格式错误")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(_jwt_signing_key(), signing_input, hashlib.sha256).digest()
    got_sig = _b64url_decode(sig_b64)
    if not hmac.compare_digest(got_sig, expected_sig):
        raise ValueError("token 签名无效")

    payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    exp = int(payload.get("exp") or 0)
    if exp and int(time.time()) > exp:
        raise ValueError("token 已过期")
    return payload


@dataclass(frozen=True)
class CurrentUser:
    id: int
    email: str
    role: str  # admin | merchant

