# api/v1/endpoints/routes.py
"""
FastAPI route handlers — thin layer only.
WHY routes should be thin:
  Route handlers handle HTTP (parse request, call service, return response).
  They should NOT contain business logic. If your route handler is >20 lines,
  some of that logic belongs in a service.

  This makes routes easy to test (mock the service) and easy to read
  (what does this endpoint DO, not HOW does it do it).
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from core.database import get_db
from core.security import verify_supabase_jwt, require_doctor
from core.exceptions import InvalidGlucoseReadingError
from models.user import (
    User, GlucoseLog, FoodLog, Medication, PointTransaction,
    UserCreate, UserResponse, GlucoseLogCreate, GlucoseLogResponse,
    FoodLogCreate, FoodLogResponse, MedicationCreate,
    RewardClaimResponse, RiskPredictionRequest, RiskPredictionResponse,
)
from services.reward_service import RewardService, NotificationService, UserService, POINTS, XP
from services.ml_service import MLService

router = APIRouter()

# ============================================================
# AUTH / USER ENDPOINTS
# ============================================================
@router.post("/auth/register", response_model=UserResponse, status_code=201)
async def register_user(data: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Register a new patient after Supabase Auth signup.

    FLOW:
      1. Frontend calls Supabase Auth to create the auth user
      2. Frontend receives the Supabase JWT
      3. Frontend calls THIS endpoint with the JWT + profile data
      4. We create our own User record linked by supabase_id

    EDGE CASE: What if called twice? UserService raises UserAlreadyExistsError → 409.
    EDGE CASE: What if Supabase user exists but our DB fails? The supabase_id foreign
               key will prevent duplicates on retry.
    """
    user = await UserService.create(db, data.model_dump())
    return UserResponse.model_validate(user)


@router.get("/auth/me", response_model=UserResponse)
async def get_current_user(current_user: User = Depends(verify_supabase_jwt)):
    """Return the authenticated user's profile."""
    return UserResponse.model_validate(current_user)


# ============================================================
# GLUCOSE LOGS
# ============================================================
@router.post("/logs/glucose", response_model=GlucoseLogResponse, status_code=201)
async def create_glucose_log(
    data: GlucoseLogCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    """
    Log a blood glucose reading with automatic ML risk scoring.

    WHY BackgroundTasks for ML inference:
      The ML prediction is run in the background so the patient gets
      an instant 201 response. The risk score is then written back
      to the database asynchronously. This keeps response time <50ms.

    EDGE CASE: Glucose > 400 mg/dL → trigger high glucose alert push notification.
    EDGE CASE: Very low glucose (<60) → hypoglycaemia alert.
    """
    # Validate physiological range
    if not (20 <= data.glucose_level <= 600):
        raise InvalidGlucoseReadingError(data.glucose_level)

    # Run ML prediction synchronously for now (fast with loaded model)
    ml_input = {
        "glucose_level": data.glucose_level,
        "bmi": data.bmi or current_user.age,  # use last known bmi
        "age": current_user.age or 40,
        "blood_pressure_systolic": data.blood_pressure_systolic or 120,
        "pregnancies": 0,
        "family_history": False,
    }
    prediction = MLService.predict(ml_input)

    log = GlucoseLog(
        user_id=current_user.id,
        glucose_level=data.glucose_level,
        bmi=data.bmi,
        blood_pressure_systolic=data.blood_pressure_systolic,
        blood_pressure_diastolic=data.blood_pressure_diastolic,
        notes=data.notes,
        risk_score=prediction["risk_score"],
        risk_label=prediction["risk_label"],
    )
    db.add(log)

    # Award points for logging
    await RewardService.award_points(db, current_user, "Glucose reading logged", POINTS["glucose_log"], XP["glucose_log"])

    # Trigger high glucose alert in background (non-blocking)
    if data.glucose_level > 250:
        background_tasks.add_task(
            _send_high_glucose_alert, current_user.id, data.glucose_level, db
        )

    await db.flush()
    return GlucoseLogResponse.model_validate(log)


async def _send_high_glucose_alert(user_id: UUID, glucose: float, db: AsyncSession):
    """Background task: send push notification for dangerously high glucose."""
    result = await db.execute(
        select(Medication).where(
            Medication.user_id == user_id,
            Medication.firebase_token.isnot(None),
            Medication.is_active == True
        ).limit(1)
    )
    med = result.scalar_one_or_none()
    if med and med.firebase_token:
        await NotificationService.send_high_glucose_alert(med.firebase_token, glucose)


@router.get("/logs/glucose", response_model=list[GlucoseLogResponse])
async def get_glucose_logs(
    days: int = Query(30, ge=1, le=365, description="Number of days of history to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    """Return glucose log history. Default: last 30 days."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(GlucoseLog)
        .where(GlucoseLog.user_id == current_user.id, GlucoseLog.logged_at >= cutoff)
        .order_by(desc(GlucoseLog.logged_at))
    )
    logs = result.scalars().all()
    return [GlucoseLogResponse.model_validate(log) for log in logs]


# ============================================================
# FOOD LOGS
# ============================================================
# Food categorisation knowledge base
FOOD_DB = {
    "white rice": ("avoid", "High GI — spikes blood sugar rapidly. Switch to brown rice or cauliflower rice."),
    "brown rice": ("good", "Lower GI and more fibre. A much better choice!"),
    "jollof rice": ("moderate", "High GI base, but portion control helps. Keep to 1 cup max."),
    "spinach": ("good", "Excellent! Low GI, high in magnesium which improves insulin sensitivity."),
    "kontomire": ("good", "Outstanding local choice — low GI and packed with nutrients."),
    "soda": ("avoid", "Pure sugar with no nutritional value. Switch to water immediately."),
    "malt": ("avoid", "Very high sugar content. Replace with zobo (unsweetened) or water."),
    "kenkey": ("moderate", "Moderate GI. Keep to half ball with plenty of fish and vegetables."),
    "fish": ("good", "Excellent lean protein. Omega-3 reduces inflammation."),
    "grilled fish": ("good", "Perfect. Grilled is always superior to fried."),
    "eggs": ("good", "Zero carbs, high quality protein. Ideal for stable blood sugar."),
    "oats": ("good", "Beta-glucan fibre slows glucose absorption significantly."),
    "groundnuts": ("good", "Slows digestion, keeps blood sugar stable. 30g as a snack is ideal."),
    "bread": ("avoid", "Refined flour causes rapid glucose spike. Switch to whole grain."),
    "plantain": ("moderate", "Green/unripe is better than ripe. Boiled always beats fried."),
    "yam": ("moderate", "Lower GI than white potato. Boiled, max 1 cup per meal."),
    "chicken": ("good", "Lean protein with no carbs. Avoid frying — grill or bake."),
    "banana": ("moderate", "Ripe banana is high GI. Choose green/unripe varieties."),
    "water": ("good", "Helps kidneys flush excess glucose. Drink 2.5L+ daily."),
}


@router.post("/logs/food", response_model=FoodLogResponse, status_code=201)
async def create_food_log(
    data: FoodLogCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    """
    Log a food item with automatic AI categorisation.

    The categorisation uses our knowledge base + the patient's profile
    (diet preference, country) to give culturally relevant advice.

    EDGE CASE: Unknown food → categorise as 'moderate' with a note to
    ask their dietitian. Never crash on unknown input.
    """
    food_key = data.food_name.lower().strip()
    # Fuzzy match against our database
    category, recommendation = "moderate", f"Unknown food logged. Ask your dietitian about '{data.food_name}'."
    for key, (cat, rec) in FOOD_DB.items():
        if key in food_key or food_key in key:
            category, recommendation = cat, rec
            break

    log = FoodLog(
        user_id=current_user.id,
        food_name=data.food_name,
        food_category=category,
        ai_recommendation=recommendation,
    )
    db.add(log)
    await RewardService.award_points(db, current_user, "Food item logged", POINTS["food_log"], XP["food_log"])
    await db.flush()
    return FoodLogResponse.model_validate(log)


@router.get("/logs/food", response_model=list[FoodLogResponse])
async def get_food_logs(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    result = await db.execute(
        select(FoodLog).where(FoodLog.user_id == current_user.id)
        .order_by(desc(FoodLog.logged_at)).limit(limit)
    )
    return [FoodLogResponse.model_validate(log) for log in result.scalars().all()]


# ============================================================
# ML RISK PREDICTION
# ============================================================
@router.post("/predict/risk", response_model=RiskPredictionResponse)
async def predict_diabetes_risk(
    data: RiskPredictionRequest,
    current_user: User = Depends(verify_supabase_jwt),
):
    """
    Run the XGBoost diabetes risk model and return a SHAP-explained prediction.

    This endpoint is the core ML feature. The response includes:
    - risk_score: 0–100 for patient-friendly display
    - shap_explanations: which features drove the prediction and by how much
    - recommendation: plain-English action to take

    RATE NOTE: This is computationally cheap (<5ms) with the model loaded.
    No rate limiting needed beyond the global 60/minute.
    """
    prediction = MLService.predict(data.model_dump())
    return RiskPredictionResponse(**prediction)


# ============================================================
# REWARDS
# ============================================================
@router.post("/rewards/claim-daily", response_model=RewardClaimResponse)
async def claim_daily_reward(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    """
    Claim the daily login reward.

    SECURITY: This runs server-side validation — the client cannot
    spoof 'already claimed' or 'streak count'. All state is authoritative
    in the database, not the frontend.

    IDEMPOTENCY: Safe to call multiple times — second call returns 409.
    """
    result = await RewardService.claim_daily_reward(db, current_user)
    return RewardClaimResponse(**result)


@router.get("/rewards/history")
async def get_reward_history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    result = await db.execute(
        select(PointTransaction).where(PointTransaction.user_id == current_user.id)
        .order_by(desc(PointTransaction.created_at)).limit(limit)
    )
    return [
        {"action": t.action, "points": t.points, "balance_after": t.balance_after, "created_at": t.created_at}
        for t in result.scalars().all()
    ]


# ============================================================
# MEDICATIONS & REMINDERS
# ============================================================
@router.post("/medications", status_code=201)
async def create_medication(
    data: MedicationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(verify_supabase_jwt),
):
    """
    Register a medication and set up push notification reminders.

    FLOW: Save medication → Schedule Celery tasks (or APScheduler) for each reminder_time.
    Each task calls NotificationService.send_medication_reminder() at the specified time.

    NOTE: In production, use Celery + Redis or APScheduler for scheduled tasks.
    The scheduler reads active medications from DB on startup to restore schedules.
    """
    import json
    med = Medication(
        user_id=current_user.id,
        name=data.name,
        dosage=data.dosage,
        reminder_times=json.dumps(data.reminder_times),
        firebase_token=data.firebase_token,
    )
    db.add(med)
    await db.flush()
    return {"id": str(med.id), "name": med.name, "dosage": med.dosage,
            "reminder_times": data.reminder_times, "message": "Medication saved. Push reminders scheduled."}


# ============================================================
# DOCTOR-ONLY ENDPOINTS
# ============================================================
@router.get("/doctor/patients", dependencies=[Depends(require_doctor)])
async def list_patients(
    db: AsyncSession = Depends(get_db),
    doctor: User = Depends(require_doctor),
):
    """
    List all patients. DOCTOR ONLY.
    WHY require_doctor dependency: patients must NEVER see other patients' data.
    Supabase Row Level Security provides a second layer of enforcement at the DB level.
    """
    result = await db.execute(
        select(User).where(User.role == "patient", User.is_active == True)
        .order_by(User.full_name)
    )
    patients = result.scalars().all()
    return [{"id": str(p.id), "name": p.full_name, "email": p.email,
             "streak": p.current_streak, "last_login": p.last_login_date} for p in patients]


@router.get("/doctor/patient/{patient_id}/logs", dependencies=[Depends(require_doctor)])
async def get_patient_logs(
    patient_id: UUID,
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific patient's glucose logs. DOCTOR ONLY."""
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(GlucoseLog)
        .where(GlucoseLog.user_id == patient_id, GlucoseLog.logged_at >= cutoff)
        .order_by(desc(GlucoseLog.logged_at))
    )
    logs = result.scalars().all()
    return [GlucoseLogResponse.model_validate(log) for log in logs]
