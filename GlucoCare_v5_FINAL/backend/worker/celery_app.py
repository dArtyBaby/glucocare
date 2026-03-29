# worker/celery_app.py
"""
WHY Celery for medication reminders:
  HTTP requests have a lifecycle: request comes in, response goes out.
  You CANNOT keep an HTTP connection open for 8 hours waiting to send a reminder.
  Celery runs tasks in separate worker processes on a schedule.

  Architecture:
    FastAPI (API) → saves reminder time to DB
    Celery Beat (scheduler) → reads DB, fires task at correct time
    Celery Worker → executes task, calls Firebase FCM
    Firebase → sends push to patient's phone

EDGE CASES:
  - Worker crashes mid-reminder → Celery retries with exponential backoff
  - Firebase token expired → catch MessagingError, mark token invalid in DB
  - Patient uninstalls app → FCM returns UNREGISTERED → deactivate medication record
  - Duplicate reminders → Celery task ID deduplication (idempotency key)
"""
import asyncio
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# WHY Redis as broker:
#   Redis is fast, simple, and supports Celery's task queue protocol.
#   In production, use Redis Cloud or AWS ElastiCache — never a single Redis
#   instance without persistence (you'd lose all pending reminders on restart).
celery_app = Celery(
    "glucocare_worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1",  # Separate DB for results
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # WHY task_acks_late=True:
    #   By default, Celery acknowledges (removes from queue) a task when it's
    #   RECEIVED by the worker, before execution. If the worker crashes during
    #   execution, the task is lost.
    #   With acks_late=True, the task is only acknowledged after successful
    #   completion — ensuring it's retried on worker failure.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Retry settings
    task_max_retries=3,
    task_default_retry_delay=60,  # 60 seconds between retries
    # Beat scheduler: check for due tasks every 60 seconds
    beat_schedule={
        "check-medication-reminders": {
            "task": "worker.tasks.check_and_send_medication_reminders",
            "schedule": crontab(minute="*"),  # Every minute
        },
        "reset-daily-streaks": {
            "task": "worker.tasks.check_broken_streaks",
            "schedule": crontab(hour=0, minute=5),  # 00:05 UTC daily
        },
        "send-sunday-facts": {
            "task": "worker.tasks.send_sunday_diabetes_facts",
            "schedule": crontab(hour=9, minute=0, day_of_week=0),  # Sunday 9am UTC
        },
    },
)
