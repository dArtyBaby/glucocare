# api/v1/endpoints/glucy_routes.py
"""Glucy state endpoint — called by frontend on every page load."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.security import verify_supabase_jwt
from models.user import User
from services.glucy_service import compute_patient_state, compute_admin_state

router = APIRouter()

@router.get("/glucy/state")
async def get_glucy_state(
    glucose: float = None,
    food_category: str = None,
    current_user: User = Depends(verify_supabase_jwt),
):
    """Return Glucy's current emotional state based on user health data."""
    from datetime import date
    last_login = current_user.last_login_date
    streak = current_user.current_streak or 0

    state = compute_patient_state(
        glucose=glucose,
        last_login_date=last_login,
        streak=streak,
        food_category=food_category,
    )
    return {"glucy": state, "user_name": current_user.full_name}


@router.get("/glucy/admin-state")
async def get_admin_glucy_state(
    role: str = "doctor",
    alert_count: int = 0,
    system_healthy: bool = True,
    current_user: User = Depends(verify_supabase_jwt),
):
    """Return Glucy's state for admin portal roles."""
    state = compute_admin_state(role=role, alert_count=alert_count, system_healthy=system_healthy)
    return {"glucy": state}
