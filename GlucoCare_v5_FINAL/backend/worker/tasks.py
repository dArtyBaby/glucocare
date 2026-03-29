# worker/tasks.py
"""
Celery tasks run in worker processes, NOT in the FastAPI event loop.
WHY this matters: SQLAlchemy async sessions are tied to an event loop.
In Celery tasks (synchronous), use psycopg2 directly or a new sync engine.
For simplicity here we use the Supabase Python client (synchronous).

IMPORTANT for future developers:
  If you migrate to async Celery workers, replace supabase-py with
  asyncpg + SQLAlchemy async session carefully — they need explicit
  event loop management per worker.
"""
import logging
import json
from datetime import datetime, timezone, timedelta, date
from celery import shared_task

logger = logging.getLogger(__name__)

# Sunday motivational diabetes facts
SUNDAY_FACTS = [
    "🌟 People with well-controlled diabetes live just as long as those without. Your discipline today is adding years to your life!",
    "💪 A 30-minute walk, 5 days a week, can reduce your A1C by up to 1%. That's as powerful as some medications!",
    "🥗 Studies show the Mediterranean diet reduces diabetes complications by 30%. Every healthy meal is a victory!",
    "🩺 Regular blood sugar monitoring is the #1 predictor of long-term diabetes outcomes. You're already winning!",
    "🧠 Stress raises blood sugar. Deep breathing for 5 minutes lowers cortisol and stabilises glucose. Try it today!",
    "🌍 Over 90% of Type 2 diabetes cases are preventable. You're already doing the most important thing: managing it!",
    "💙 You are not your diagnosis. Diabetes is something you have, not something you are. Keep going!",
    "🔬 Research shows that people who track their meals have 40% better glucose control. Your food log is your superpower!",
    "😴 Poor sleep raises blood sugar the next morning by up to 23 mg/dL. Protecting your sleep protects your health!",
    "🏆 Consistency beats perfection every time. One logged meal is better than a 'perfect' day you skipped entirely!",
]


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def check_and_send_medication_reminders(self):
    """
    Run every minute. Find all medications due in the next 2 minutes and fire push notifications.

    ALGORITHM:
      1. Get current UTC time
      2. Query medications where reminder_times contains the current HH:MM
      3. For each match, send FCM push notification
      4. Log delivery attempt

    EDGE CASES:
      - FCM token invalid → mark token as None in DB, stop retrying
      - Firebase service down → retry with backoff (handled by Celery)
      - Task takes >1 minute → next run may overlap; use task lock via Redis
    """
    from supabase import create_client
    import os

    supabase = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],  # Need admin access
    )

    now_utc = datetime.now(timezone.utc)
    current_time = now_utc.strftime("%H:%M")

    try:
        # Fetch active medications with Firebase tokens
        result = supabase.table("medications").select(
            "id, user_id, name, dosage, reminder_times, firebase_token"
        ).eq("is_active", True).not_.is_("firebase_token", "null").execute()

        medications = result.data or []
        sent_count = 0

        for med in medications:
            try:
                times = json.loads(med["reminder_times"]) if isinstance(med["reminder_times"], str) else med["reminder_times"]
                if current_time in times:
                    success = _send_fcm_notification(
                        token=med["firebase_token"],
                        title="💊 Medication Reminder",
                        body=f"Time to take your {med['name']} {med['dosage']}!",
                        data={"type": "medication", "medication_id": med["id"]},
                    )
                    if success:
                        sent_count += 1
                    elif success is None:
                        # Token is invalid — clean up
                        supabase.table("medications").update(
                            {"firebase_token": None}
                        ).eq("id", med["id"]).execute()
                        logger.warning(f"Invalid FCM token cleaned up for medication {med['id']}")

            except Exception as e:
                logger.error(f"Error processing medication {med.get('id')}: {e}")
                continue

        logger.info(f"Medication reminders: {sent_count} sent at {current_time}")
        return {"sent": sent_count, "checked": len(medications), "time": current_time}

    except Exception as exc:
        logger.error(f"Reminder task failed: {exc}")
        raise self.retry(exc=exc)


@shared_task
def check_broken_streaks():
    """
    Run daily at 00:05 UTC.
    Find users who didn't log yesterday and reset their streaks.

    WHY NOT do this in the login endpoint:
      If a user never logs in, their streak count stays at an old value.
      A background job ensures streaks are accurate regardless of login frequency.
      It's also cheaper: one batch DB query vs. per-request computation.
    """
    from supabase import create_client
    import os

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()

    # Find users with a streak > 0 who didn't log yesterday
    result = supabase.table("users").select(
        "id, full_name, current_streak, last_login_date"
    ).gt("current_streak", 0).lt("last_login_date", yesterday).execute()

    broken = result.data or []
    reset_count = 0

    for user in broken:
        supabase.table("users").update({"current_streak": 0}).eq("id", user["id"]).execute()
        reset_count += 1
        logger.info(f"Streak reset for user {user['id']} (was {user['current_streak']} days)")

    logger.info(f"Daily streak check: {reset_count} streaks reset")
    return {"reset_count": reset_count}


@shared_task
def send_sunday_diabetes_facts():
    """
    Every Sunday at 9am UTC: send a motivational diabetes fact to all active users.
    Uses round-robin selection based on week number so facts rotate without repetition.
    """
    from supabase import create_client
    import os

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    # Pick fact based on week number (rotates through all 10 facts)
    week_num = datetime.now(timezone.utc).isocalendar()[1]
    fact = SUNDAY_FACTS[week_num % len(SUNDAY_FACTS)]

    # Get all users with FCM tokens
    meds = supabase.table("medications").select(
        "firebase_token, user_id"
    ).eq("is_active", True).not_.is_("firebase_token", "null").execute()

    # Deduplicate by user (one notification per user)
    seen_users = set()
    sent = 0
    for med in (meds.data or []):
        if med["user_id"] not in seen_users:
            seen_users.add(med["user_id"])
            _send_fcm_notification(
                token=med["firebase_token"],
                title="🎓 Fun Fact Sunday!",
                body=fact,
                data={"type": "sunday_fact"},
            )
            sent += 1

    logger.info(f"Sunday facts sent to {sent} users")
    return {"sent": sent, "fact_index": week_num % len(SUNDAY_FACTS)}


@shared_task
def award_full_daily_log_bonus(user_id: str):
    """
    Called when a user completes breakfast + lunch + dinner + glucose log.
    Awards the 50-point full daily log bonus.
    WHY a separate task: Completion check runs asynchronously after each log entry,
    preventing any single endpoint from blocking on multiple DB queries.
    """
    from supabase import create_client
    import os

    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()

    # Check food logs today
    food_result = supabase.table("food_logs").select("id").eq(
        "user_id", user_id
    ).gte("logged_at", today_start).execute()

    # Check glucose logs today
    glucose_result = supabase.table("glucose_logs").select("id").eq(
        "user_id", user_id
    ).gte("logged_at", today_start).execute()

    food_count = len(food_result.data or [])
    glucose_count = len(glucose_result.data or [])

    # Full day = at least 3 food entries + 1 glucose reading
    if food_count >= 3 and glucose_count >= 1:
        # Check if bonus already awarded today to prevent duplicates
        bonus_check = supabase.table("point_transactions").select("id").eq(
            "user_id", user_id
        ).eq("action", "Full daily log bonus").gte("created_at", today_start).execute()

        if not bonus_check.data:
            # Award bonus
            user = supabase.table("users").select("points_balance").eq("id", user_id).single().execute()
            new_balance = user.data["points_balance"] + 50

            supabase.table("users").update({"points_balance": new_balance}).eq("id", user_id).execute()
            supabase.table("point_transactions").insert({
                "user_id": user_id,
                "action": "Full daily log bonus",
                "points": 50,
                "balance_after": new_balance,
            }).execute()
            logger.info(f"Full daily log bonus awarded to user {user_id}")
            return {"awarded": True, "points": 50}

    return {"awarded": False, "food_count": food_count, "glucose_count": glucose_count}


def _send_fcm_notification(token: str, title: str, body: str, data: dict = None) -> bool | None:
    """
    Send a single FCM push notification.
    Returns:
      True  → sent successfully
      False → failed (retry-able error)
      None  → token invalid (clean up from DB)
    """
    try:
        import firebase_admin
        from firebase_admin import messaging

        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in (data or {}).items()},
            android=messaging.AndroidConfig(priority="high"),
            apns=messaging.APNSConfig(
                payload=messaging.APNSPayload(aps=messaging.Aps(sound="default"))
            ),
            token=token,
        )
        messaging.send(message)
        return True

    except Exception as e:
        error_str = str(e).lower()
        if "registration-token-not-registered" in error_str or "invalid-registration-token" in error_str:
            return None  # Token dead — caller should clean up DB
        logger.error(f"FCM send failed: {e}")
        return False
