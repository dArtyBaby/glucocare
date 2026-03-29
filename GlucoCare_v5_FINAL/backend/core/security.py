# core/security.py
"""
WHY: JWT verification MUST happen in a FastAPI Dependency, not middleware alone.
Dependencies run per-endpoint and can inject the verified user into route handlers.
This gives you:
  1. Automatic 401 responses if the token is missing or invalid
  2. The authenticated user object available in every protected route
  3. Easy role-based access control (patient vs doctor vs admin)

EDGE CASE THINKING:
  - What if the token is expired? → jose raises JWTError, we return 401
  - What if the user was deleted from Supabase but token still valid? → DB check catches it
  - What if someone sends a self-signed token with a different secret? → Signature verification fails
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db
from models.user import User

settings = get_settings()
security = HTTPBearer()


async def verify_supabase_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Verifies a Supabase-issued JWT and returns the authenticated User.

    Supabase signs JWTs with your project's JWT_SECRET.
    The token payload contains: sub (user id), email, role, exp.

    IMPORTANT: We verify the token cryptographically, then cross-check
    the user exists in OUR database. This prevents ghost users
    (Supabase user deleted but token still in circulation).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SUPABASE_JWT_SECRET,
            algorithms=[settings.ALGORITHM],
            options={"verify_aud": False},  # Supabase uses custom audience
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise credentials_exception

    # Cross-check against our own database
    from services.user_service import UserService
    user = await UserService.get_by_supabase_id(db, user_id)
    if user is None:
        raise credentials_exception

    return user


async def require_doctor(
    current_user: User = Depends(verify_supabase_jwt),
) -> User:
    """
    Role-based guard: Only doctors can access doctor-specific routes.
    WHY separate dependency: Composable, reusable, testable.
    """
    if current_user.role not in ("doctor", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this resource.",
        )
    return current_user


async def require_patient(
    current_user: User = Depends(verify_supabase_jwt),
) -> User:
    """Guard for patient-only routes."""
    if current_user.role not in ("patient", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is for patients only.",
        )
    return current_user
