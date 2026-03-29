# models/user.py
"""
WHY separate Pydantic schemas from SQLAlchemy models:
  SQLAlchemy models define the DATABASE structure (tables, columns, relationships).
  Pydantic schemas define the API CONTRACT (what comes in, what goes out).
  Mixing them couples your DB schema to your API — any DB change breaks your API.

  Pattern: SQLAlchemy model → Service layer → Pydantic schema → Client
  Data never flows directly from DB to client.
"""
from datetime import datetime, date
from typing import Optional
from enum import Enum
import uuid

from sqlalchemy import String, Boolean, Integer, Float, ForeignKey, Text, Date, DateTime
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr, Field, field_validator

from core.database import Base


# ============================================================
# ENUMS
# ============================================================
class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class DietPreference(str, Enum):
    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    HALAL = "halal"
    KOSHER = "kosher"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


# ============================================================
# SQLALCHEMY MODELS (Database Tables)
# ============================================================
class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    supabase_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SAEnum(UserRole), default=UserRole.PATIENT, nullable=False)
    gender: Mapped[Optional[Gender]] = mapped_column(SAEnum(Gender), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_diabetic: Mapped[bool] = mapped_column(Boolean, default=False)
    diet_preference: Mapped[DietPreference] = mapped_column(SAEnum(DietPreference), default=DietPreference.OMNIVORE)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Rewards
    points_balance: Mapped[int] = mapped_column(Integer, default=0)
    xp_total: Mapped[int] = mapped_column(Integer, default=0)
    current_streak: Mapped[int] = mapped_column(Integer, default=0)
    longest_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_login_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_reward_claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    glucose_logs: Mapped[list["GlucoseLog"]] = relationship("GlucoseLog", back_populates="user", cascade="all, delete-orphan")
    food_logs: Mapped[list["FoodLog"]] = relationship("FoodLog", back_populates="user", cascade="all, delete-orphan")
    point_transactions: Mapped[list["PointTransaction"]] = relationship("PointTransaction", back_populates="user")
    medications: Mapped[list["Medication"]] = relationship("Medication", back_populates="user", cascade="all, delete-orphan")


class GlucoseLog(Base):
    __tablename__ = "glucose_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    glucose_level: Mapped[float] = mapped_column(Float, nullable=False)  # mg/dL
    bmi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    blood_pressure_systolic: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    blood_pressure_diastolic: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # ML prediction stored alongside the log
    risk_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="glucose_logs")


class FoodLog(Base):
    __tablename__ = "food_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    food_name: Mapped[str] = mapped_column(String(255), nullable=False)
    food_category: Mapped[str] = mapped_column(String(50), nullable=False)  # good / moderate / avoid
    calories_estimate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ai_recommendation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="food_logs")


class Medication(Base):
    __tablename__ = "medications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dosage: Mapped[str] = mapped_column(String(100), nullable=False)
    reminder_times: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array: ["08:00", "14:00", "20:00"]
    firebase_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Device FCM token
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped["User"] = relationship("User", back_populates="medications")


class PointTransaction(Base):
    __tablename__ = "point_transactions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    points: Mapped[int] = mapped_column(Integer, nullable=False)  # positive = earned, negative = spent
    balance_after: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="point_transactions")


# ============================================================
# PYDANTIC SCHEMAS (API Contract)
# ============================================================
class UserCreate(BaseModel):
    supabase_id: str
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    gender: Optional[Gender] = None
    age: Optional[int] = Field(None, ge=10, le=120)
    country: Optional[str] = None
    is_diabetic: bool = False
    diet_preference: DietPreference = DietPreference.OMNIVORE

    @field_validator("full_name")
    @classmethod
    def name_must_contain_space(cls, v: str) -> str:
        # Don't enforce this strictly — some cultures use single names
        return v.strip()


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: UserRole
    gender: Optional[Gender]
    age: Optional[int]
    country: Optional[str]
    is_diabetic: bool
    diet_preference: DietPreference
    points_balance: int
    xp_total: int
    current_streak: int
    created_at: datetime

    model_config = {"from_attributes": True}


class GlucoseLogCreate(BaseModel):
    glucose_level: float = Field(..., ge=20, le=600, description="Blood glucose in mg/dL")
    bmi: Optional[float] = Field(None, ge=10, le=80)
    blood_pressure_systolic: Optional[int] = Field(None, ge=60, le=250)
    blood_pressure_diastolic: Optional[int] = Field(None, ge=40, le=150)
    notes: Optional[str] = Field(None, max_length=500)


class GlucoseLogResponse(BaseModel):
    id: uuid.UUID
    glucose_level: float
    bmi: Optional[float]
    blood_pressure_systolic: Optional[int]
    blood_pressure_diastolic: Optional[int]
    risk_score: Optional[float]
    risk_label: Optional[str]
    notes: Optional[str]
    logged_at: datetime

    model_config = {"from_attributes": True}


class FoodLogCreate(BaseModel):
    food_name: str = Field(..., min_length=1, max_length=255)


class FoodLogResponse(BaseModel):
    id: uuid.UUID
    food_name: str
    food_category: str
    ai_recommendation: Optional[str]
    logged_at: datetime

    model_config = {"from_attributes": True}


class MedicationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    dosage: str = Field(..., min_length=1, max_length=100)
    reminder_times: list[str] = Field(..., min_length=1)  # e.g. ["08:00", "14:00"]
    firebase_token: Optional[str] = None

    @field_validator("reminder_times")
    @classmethod
    def validate_times(cls, times: list[str]) -> list[str]:
        import re
        pattern = re.compile(r"^\d{2}:\d{2}$")
        for t in times:
            if not pattern.match(t):
                raise ValueError(f"'{t}' is not a valid time format. Use HH:MM.")
        return times


class RewardClaimResponse(BaseModel):
    success: bool
    points_earned: int
    new_balance: int
    new_streak: int
    message: str
    bonus_unlocked: Optional[str] = None


class RiskPredictionRequest(BaseModel):
    glucose_level: float = Field(..., ge=20, le=600)
    bmi: float = Field(..., ge=10, le=80)
    age: int = Field(..., ge=10, le=120)
    blood_pressure_systolic: int = Field(..., ge=60, le=250)
    pregnancies: int = Field(0, ge=0, le=20)
    family_history: bool = False
    insulin: Optional[float] = Field(None, ge=0, le=900)
    skin_thickness: Optional[float] = Field(None, ge=0, le=100)


class RiskPredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0, le=100)
    risk_label: str
    risk_color: str
    recommendation: str
    shap_explanations: list[dict]
    model_confidence: float
