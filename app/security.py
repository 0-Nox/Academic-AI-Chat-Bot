"""
Auth layer (Section 19 — Security Requirements).

- All API endpoints require token-based auth (JWT) -> enforced via
  `get_current_user` as a FastAPI dependency.
- Role-based access control restricts document ingestion / log access to
  Admin -> enforced via `require_role`.
- Passwords are hashed (bcrypt), never stored or logged in plaintext.

No live ERP/LMS integration exists yet (Constraints, Section 12), so this
ships with a small set of demo users seeded at startup. Swap
`authenticate_user` for a real ERP identity check later without touching
any other module.
"""

from datetime import timedelta
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app import config, db

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


class TokenData(BaseModel):
    username: str
    role: str


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(username: str, role: str) -> str:
    expire = db.utcnow() + timedelta(minutes=config.JWT_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = db.users.find_one({"username": username})
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_error
    except jwt.PyJWTError:
        raise credentials_error
    return TokenData(username=username, role=role)


def require_role(*allowed_roles: str):
    """Dependency factory: `Depends(require_role("admin"))` on a route."""

    def _check(user: TokenData = Depends(get_current_user)) -> TokenData:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return user

    return _check


def seed_demo_users() -> None:
    """Idempotent seed for the three roles in Section 10 (User Roles).
    Only runs against an empty users collection so it never clobbers real
    accounts once they exist."""
    if db.users.count_documents({}) > 0:
        return
    demo_users = [
        {"username": "student1", "password": "student123", "role": "student", "name": "Demo Student"},
        {"username": "faculty1", "password": "faculty123", "role": "faculty", "name": "Demo Faculty"},
        {"username": "admin1", "password": "admin123", "role": "admin", "name": "Demo Admin"},
    ]
    for u in demo_users:
        db.users.insert_one(
            {
                "username": u["username"],
                "password_hash": hash_password(u["password"]),
                "role": u["role"],
                "name": u["name"],
                "created_at": db.utcnow(),
            }
        )
    print(
        "Seeded demo users: student1/student123, faculty1/faculty123, "
        "admin1/admin123 (change these before any real deployment)."
    )
