# services/reward_service.py
"""
WHY the reward logic must run in a database transaction:
  If we update points and then the transaction record fails (or vice versa),
  you end up with inconsistent state — points added but no audit trail,
  or an audit trail with no actual points.

  Using a single database transaction (atomic operation) ensures BOTH writes
  succeed together or BOTH are rolled back. This is the core of financial-grade
  data integrity.

EDGE CASES:
  - User claims reward twice in same 24h window → RewardAlreadyClaimedError
  - User's streak should reset (> 1 day gap) → we check and reset
  - Streak milestone reached → bonus points awarded automatically
"""
import logging
from datetime import datetime, date, timezone, timedelta
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.user import User, PointTransaction
from core.exceptions import RewardAlreadyClaimedError

logger = logging.getLogger(__name__)

# Point values — centralised so changing a value here reflects everywhere
POINTS = {
    "daily_login": 10,
    "full_daily_log": 50,
    "streak_7_bonus": 100,
    "streak_14_bonus": 150,
    "streak_30_bonus": 300,
    "signup_bonus": 100,
    "glucose_log": 15,
    "food_log": 10,
}

XP = {
    "daily_login": 20,
    "glucose_log": 25,
    "food_log": 15,
    "full_daily_log": 75,
}


class RewardService:

    @staticmethod
    async def claim_daily_reward(db: AsyncSession, user: User) -> dict:
        """
        Claim daily login reward with streak tracking.

        ALGORITHM:
          1. Check if already claimed today (idempotency guard)
          2. Calculate new streak (reset if > 1 day gap)
          3. Determine base points + any streak bonus
          4. Write points + transaction + user update atomically
        """
        now = datetime.now(timezone.utc)
        today = now.date()

        # Guard: Already claimed today?
        if user.last_reward_claimed_at is not None:
            last_claimed_date = user.last_reward_claimed_at.date()
            if last_claimed_date == today:
                raise RewardAlreadyClaimedError()

        # Calculate streak
        new_streak = RewardService._calculate_streak(user.last_login_date, today, user.current_streak)

        # Calculate points
        pts = POINTS["daily_login"]
        xp = XP["daily_login"]
        bonus_unlocked: Optional[str] = None

        # Streak milestone bonuses
        if new_streak == 7:
            pts += POINTS["streak_7_bonus"]
            bonus_unlocked = "🎉 7-Day Streak! Unlocked: Snack Guide PDF"
        elif new_streak == 14:
            pts += POINTS["streak_14_bonus"]
            bonus_unlocked = "🏆 14-Day Streak! Unlocked: Monthly Trend Report"
        elif new_streak == 30:
            pts += POINTS["streak_30_bonus"]
            bonus_unlocked = "🌟 30-Day Streak! Unlocked: A1C Prediction Tool"

        new_balance = user.points_balance + pts
        new_xp = user.xp_total + xp

        # Write all changes in a single transaction
        # The session.commit() in get_db() ensures atomicity
        user.points_balance = new_balance
        user.xp_total = new_xp
        user.current_streak = new_streak
        user.longest_streak = max(user.longest_streak, new_streak)
        user.last_login_date = today
        user.last_reward_claimed_at = now

        # Audit trail: NEVER modify points without a transaction record
        transaction = PointTransaction(
            user_id=user.id,
            action=f"Daily login reward (Day {new_streak})",
            points=pts,
            balance_after=new_balance,
        )
        db.add(transaction)

        logger.info(f"Reward claimed: user={user.id}, pts={pts}, streak={new_streak}")

        return {
            "success": True,
            "points_earned": pts,
            "new_balance": new_balance,
            "new_streak": new_streak,
            "message": f"+{pts} points earned! 🎉 Day {new_streak} streak!",
            "bonus_unlocked": bonus_unlocked,
        }

    @staticmethod
    def _calculate_streak(last_login: Optional[date], today: date, current_streak: int) -> int:
        """
        Streak logic:
          - First login ever → streak = 1
          - Logged in yesterday → streak += 1
          - Logged in today already → streak unchanged (handled by guard above)
          - More than 1 day gap → streak resets to 1
        """
        if last_login is None:
            return 1
        days_since = (today - last_login).days
        if days_since == 1:
            return current_streak + 1
        elif days_since == 0:
            return current_streak  # Same day, shouldn't reach here
        else:
            return 1  # Streak broken

    @staticmethod
    async def award_points(db: AsyncSession, user: User, action: str, pts: int, xp: int = 0) -> int:
        """Award points for a specific action (glucose log, food log, etc.)"""
        new_balance = user.points_balance + pts
        user.points_balance = new_balance
        user.xp_total += xp

        transaction = PointTransaction(
            user_id=user.id,
            action=action,
            points=pts,
            balance_after=new_balance,
        )
        db.add(transaction)
        return new_balance

    @staticmethod
    async def spend_points(db: AsyncSession, user: User, action: str, cost: int) -> int:
        """Spend points on a reward. Raises ValueError if insufficient balance."""
        if user.points_balance < cost:
            raise ValueError(f"Insufficient points. Need {cost}, have {user.points_balance}.")
        new_balance = user.points_balance - cost
        user.points_balance = new_balance
        transaction = PointTransaction(
            user_id=user.id,
            action=f"Spent: {action}",
            points=-cost,
            balance_after=new_balance,
        )
        db.add(transaction)
        return new_balance


# ============================================================
# NOTIFICATION SERVICE (Firebase Cloud Messaging)
# ============================================================
class NotificationService:
    """
    WHY Firebase for push notifications:
      FCM is free, cross-platform (iOS + Android + Web), and integrates
      with React Native / PWA natively. The backend triggers notifications;
      the device receives them even when the app is closed.

    SECURITY NOTE: The Firebase Admin SDK uses a service account JSON file.
    NEVER commit this file to git. It goes in .env / secrets manager.
    """
    _app = None

    @classmethod
    def init(cls, credentials_path: str) -> None:
        """Initialise Firebase Admin SDK at startup."""
        try:
            import firebase_admin
            from firebase_admin import credentials
            cred = credentials.Certificate(credentials_path)
            cls._app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialised successfully.")
        except Exception as e:
            logger.warning(f"Firebase initialisation failed: {e}. Push notifications disabled.")

    @classmethod
    async def send_medication_reminder(cls, fcm_token: str, medication_name: str, dosage: str) -> bool:
        """Send a medication reminder push notification."""
        if cls._app is None:
            logger.warning("Firebase not initialised. Skipping push notification.")
            return False
        try:
            from firebase_admin import messaging
            message = messaging.Message(
                notification=messaging.Notification(
                    title="💊 Time for your medication!",
                    body=f"Don't forget: {medication_name} {dosage}",
                ),
                data={
                    "type": "medication_reminder",
                    "medication_name": medication_name,
                    "dosage": dosage,
                },
                token=fcm_token,
            )
            response = messaging.send(message)
            logger.info(f"Push notification sent: {response}")
            return True
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return False

    @classmethod
    async def send_high_glucose_alert(cls, fcm_token: str, glucose_level: float) -> bool:
        """Alert patient when glucose reading is dangerously high."""
        if cls._app is None:
            return False
        try:
            from firebase_admin import messaging
            message = messaging.Message(
                notification=messaging.Notification(
                    title="⚠️ High Blood Sugar Alert",
                    body=f"Your reading of {glucose_level} mg/dL is high. Please check with your doctor.",
                ),
                android=messaging.AndroidConfig(priority="high"),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(sound="default", badge=1)
                    )
                ),
                token=fcm_token,
            )
            messaging.send(message)
            return True
        except Exception as e:
            logger.error(f"Alert failed: {e}")
            return False


# ============================================================
# USER SERVICE
# ============================================================
class UserService:

    @staticmethod
    async def get_by_supabase_id(db: AsyncSession, supabase_id: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.supabase_id == supabase_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id) -> Optional[User]:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def create(db: AsyncSession, data: dict) -> User:
        from core.exceptions import UserAlreadyExistsError
        # Check for existing user
        existing = await db.execute(select(User).where(User.email == data["email"]))
        if existing.scalar_one_or_none():
            raise UserAlreadyExistsError(data["email"])

        user = User(**data, points_balance=POINTS["signup_bonus"])
        db.add(user)

        # Award signup bonus transaction
        await db.flush()  # Get the user ID before creating transaction
        transaction = PointTransaction(
            user_id=user.id,
            action="Welcome bonus for joining GlucoCare!",
            points=POINTS["signup_bonus"],
            balance_after=POINTS["signup_bonus"],
        )
        db.add(transaction)
        return user
