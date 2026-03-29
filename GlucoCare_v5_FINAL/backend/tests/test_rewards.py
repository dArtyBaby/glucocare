# tests/test_rewards.py
"""
Professional test structure for GlucoCare.

WHY testing matters for healthcare apps:
  A bug in a fintech app might cost money. A bug in a healthcare app
  might cause a patient to miss a medication or trust a wrong risk score.
  Tests are non-negotiable here.

Test categories:
  Unit tests: Test a single function with mocked dependencies (fast, isolated)
  Integration tests: Test the full request → DB → response cycle (slower, realistic)

Run all tests: pytest tests/ -v
Run with coverage: pytest tests/ --cov=. --cov-report=html
"""
import pytest
from datetime import date, datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from services.reward_service import RewardService, POINTS
from core.exceptions import RewardAlreadyClaimedError


# ============================================================
# FIXTURES
# ============================================================
def make_user(**kwargs):
    """Factory for test User objects."""
    user = MagicMock()
    user.id = uuid.uuid4()
    user.full_name = "Test Patient"
    user.email = "test@glucocare.app"
    user.points_balance = kwargs.get("points_balance", 0)
    user.xp_total = kwargs.get("xp_total", 0)
    user.current_streak = kwargs.get("current_streak", 0)
    user.longest_streak = kwargs.get("longest_streak", 0)
    user.last_login_date = kwargs.get("last_login_date", None)
    user.last_reward_claimed_at = kwargs.get("last_reward_claimed_at", None)
    return user


# ============================================================
# UNIT TESTS: Streak Calculation
# ============================================================
class TestStreakCalculation:
    """
    WHY test streak logic exhaustively:
      The streak algorithm has 4 distinct code paths (first login, consecutive,
      same day, gap). Missing any one of them creates unfair user experiences —
      reset streaks that shouldn't be reset, or streaks that never reset.
    """

    def test_first_ever_login(self):
        """Brand new user — no last_login_date — starts at 1."""
        result = RewardService._calculate_streak(None, date.today(), 0)
        assert result == 1, "First login should start streak at 1"

    def test_consecutive_day_increments(self):
        """Logged in yesterday → streak grows."""
        yesterday = date.today() - timedelta(days=1)
        result = RewardService._calculate_streak(yesterday, date.today(), 5)
        assert result == 6, "Consecutive day should increment streak"

    def test_same_day_no_change(self):
        """Multiple logins same day → streak unchanged."""
        today = date.today()
        result = RewardService._calculate_streak(today, today, 5)
        assert result == 5, "Same-day login should not change streak"

    def test_two_day_gap_resets(self):
        """Missed a day → streak resets to 1."""
        two_days_ago = date.today() - timedelta(days=2)
        result = RewardService._calculate_streak(two_days_ago, date.today(), 10)
        assert result == 1, "Gap of 2+ days should reset streak to 1"

    def test_week_gap_resets(self):
        """Week-long gap → still resets to 1, not 0."""
        last_week = date.today() - timedelta(days=7)
        result = RewardService._calculate_streak(last_week, date.today(), 25)
        assert result == 1, "Streak reset should be 1, not 0 (today counts)"

    def test_streak_bonus_at_7(self):
        """Reaching day 7 streak triggers the bonus."""
        # 6 days already → today is day 7
        yesterday = date.today() - timedelta(days=1)
        result = RewardService._calculate_streak(yesterday, date.today(), 6)
        assert result == 7


# ============================================================
# UNIT TESTS: Reward Claim Logic
# ============================================================
class TestRewardClaim:

    @pytest.mark.asyncio
    async def test_claim_raises_if_already_claimed_today(self):
        """
        EDGE CASE: Calling claim-daily twice in same day returns 409.
        This prevents clients from earning infinite points by calling the API in a loop.
        """
        user = make_user(
            last_reward_claimed_at=datetime.now(timezone.utc)  # Claimed right now
        )
        db = AsyncMock()

        with pytest.raises(RewardAlreadyClaimedError):
            await RewardService.claim_daily_reward(db, user)

    @pytest.mark.asyncio
    async def test_first_claim_awards_login_points(self):
        """First ever claim awards the daily login points."""
        user = make_user(points_balance=100, current_streak=0, last_login_date=None)
        db = AsyncMock()
        db.add = MagicMock()
        db.flush = AsyncMock()

        result = await RewardService.claim_daily_reward(db, user)

        assert result["success"] is True
        assert result["points_earned"] >= POINTS["daily_login"]
        assert result["new_streak"] == 1

    @pytest.mark.asyncio
    async def test_seven_day_streak_awards_bonus(self):
        """Day 7 streak should award streak_7_bonus on top of login points."""
        yesterday = date.today() - timedelta(days=1)
        user = make_user(
            points_balance=500,
            current_streak=6,  # Today becomes day 7
            last_login_date=yesterday,
            last_reward_claimed_at=datetime.now(timezone.utc) - timedelta(hours=25),
        )
        db = AsyncMock()
        db.add = MagicMock()

        result = await RewardService.claim_daily_reward(db, user)

        expected_pts = POINTS["daily_login"] + POINTS["streak_7_bonus"]
        assert result["points_earned"] == expected_pts
        assert result["new_streak"] == 7
        assert result["bonus_unlocked"] is not None
        assert "7-Day" in result["bonus_unlocked"]

    @pytest.mark.asyncio
    async def test_points_transaction_created(self):
        """Every reward claim must create an audit trail transaction."""
        user = make_user(points_balance=0, last_login_date=None)
        db = AsyncMock()
        db.add = MagicMock()

        await RewardService.claim_daily_reward(db, user)

        # db.add should have been called (transaction + user update)
        assert db.add.called, "A PointTransaction must be created for every reward"

    @pytest.mark.asyncio
    async def test_insufficient_points_to_spend(self):
        """Spending more points than balance should raise ValueError."""
        user = make_user(points_balance=50)
        db = AsyncMock()
        db.add = MagicMock()

        with pytest.raises(ValueError, match="Insufficient points"):
            await RewardService.spend_points(db, user, "Dark Mode theme", 200)


# ============================================================
# UNIT TESTS: ML Service
# ============================================================
class TestMLService:

    def test_heuristic_high_glucose_gives_high_risk(self):
        """Glucose 300 mg/dL should produce a very high risk score."""
        from services.ml_service import MLService
        result = MLService._heuristic_predict({
            "glucose_level": 300,
            "bmi": 35,
            "age": 60,
            "blood_pressure_systolic": 150,
            "family_history": True,
        })
        assert result["risk_score"] >= 70, "Very high glucose + multiple factors should be high risk"
        assert result["risk_label"] in ("High Risk", "Very High Risk")

    def test_heuristic_healthy_gives_low_risk(self):
        """Healthy values should produce low risk."""
        from services.ml_service import MLService
        result = MLService._heuristic_predict({
            "glucose_level": 85,
            "bmi": 22,
            "age": 28,
            "blood_pressure_systolic": 110,
            "family_history": False,
        })
        assert result["risk_score"] < 25
        assert result["risk_label"] == "Low Risk"

    def test_feature_extraction_shape(self):
        """Feature array must have exactly 8 features in the correct order."""
        from services.ml_service import MLService
        features = MLService._extract_features({
            "pregnancies": 2,
            "glucose_level": 120,
            "blood_pressure_systolic": 80,
            "skin_thickness": 20,
            "insulin": 90,
            "bmi": 25,
            "family_history": True,
            "age": 35,
        })
        assert features.shape == (1, 8), f"Expected (1, 8), got {features.shape}"

    def test_family_history_bool_converts_to_int(self):
        """True/False family history must convert to 1/0 for model."""
        from services.ml_service import MLService
        features_yes = MLService._extract_features({"family_history": True, "glucose_level": 100,
                                                     "bmi": 25, "age": 40, "blood_pressure_systolic": 120})
        features_no = MLService._extract_features({"family_history": False, "glucose_level": 100,
                                                    "bmi": 25, "age": 40, "blood_pressure_systolic": 120})
        assert features_yes[0][6] == 1
        assert features_no[0][6] == 0


# ============================================================
# INTEGRATION TESTS (require running test database)
# ============================================================
# These tests are marked with @pytest.mark.integration and only run
# when --integration flag is passed: pytest tests/ -m integration

@pytest.mark.integration
class TestGlucoseLogIntegration:
    """Full request → DB → response cycle tests."""

    @pytest.fixture
    async def client(self):
        """Create test client with a fresh test database."""
        from httpx import AsyncClient
        from main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    async def test_glucose_log_returns_risk_score(self, client, auth_headers):
        """POST /logs/glucose should return a risk score from the ML model."""
        response = await client.post(
            "/api/v1/logs/glucose",
            json={"glucose_level": 142, "bmi": 28.0, "blood_pressure_systolic": 130},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 100

    async def test_invalid_glucose_rejected(self, client, auth_headers):
        """Glucose outside valid range (20–600) should return 422."""
        response = await client.post(
            "/api/v1/logs/glucose",
            json={"glucose_level": 999},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ============================================================
# HOW TO UPDATE THIS TEST FILE
# ============================================================
# 1. Adding a new feature? Write the test FIRST (TDD).
#    - Define what the function should return for valid input
#    - Define what it should raise for invalid input
#    - THEN write the implementation
#
# 2. Found a bug? Write a failing test that reproduces it FIRST.
#    Then fix the bug. The test prevents regression.
#
# 3. Changing streak calculation? Add a new test case for the edge case
#    that prompted the change. Never delete existing tests.
#
# 4. Adding a new reward type? Add it to POINTS dict in reward_service.py
#    AND add a test case in TestRewardClaim verifying the correct amount.
#
# 5. Run before every git push:
#    pytest tests/ -v --tb=short
#    pytest tests/ --cov=. --cov-fail-under=70
