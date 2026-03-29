# main.py
"""
FastAPI application entry point.

WHY the lifespan context manager (not @app.on_event):
  on_event("startup") is deprecated in FastAPI 0.93+.
  The lifespan approach is cleaner: setup before yield, teardown after yield.
  It also makes testing easier — you can inject a different lifespan in tests.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from core.config import get_settings
from core.database import init_db
from core.exceptions import register_exception_handlers
from services.ml_service import MLService
from services.reward_service import NotificationService
from api.v1.endpoints.routes import router

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    Everything before 'yield' = startup.
    Everything after 'yield' = shutdown (cleanup, close connections).
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # 1. Initialise database tables
    await init_db()

    # 2. Load XGBoost model into memory (expensive, do once)
    MLService.load(settings.MODEL_PATH, settings.SCALER_PATH)

    # 3. Initialise Firebase for push notifications
    NotificationService.init(settings.FIREBASE_CREDENTIALS_PATH)

    logger.info("All services initialised. API ready. 🚀")

    yield  # App runs here

    # Shutdown cleanup
    logger.info("Shutting down GlucoCare API...")


# ============================================================
# App instance
# ============================================================
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Diabetes management platform with AI risk prediction, gamified rewards, and doctor communication.",
    docs_url="/docs",  # Hide Swagger in production
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ============================================================
# MIDDLEWARE (order matters — outermost middleware runs first)
# ============================================================

# 1. CORS — must be first so pre-flight OPTIONS requests are handled
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)

# 2. Trusted hosts — prevents Host header injection attacks
if not settings.DEBUG:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["glucocare.app", "api.glucocare.app"])

# ============================================================
# EXCEPTION HANDLERS
# ============================================================
register_exception_handlers(app)

# ============================================================
# ROUTES
# ============================================================
app.include_router(router, prefix=settings.API_V1_PREFIX, tags=["GlucoCare API"])


@app.get("/health", include_in_schema=False)
async def health_check():
    """
    WHY a /health endpoint:
    Load balancers, Docker health checks, and Kubernetes liveness probes
    all need a cheap endpoint to verify the service is alive.
    Never put business logic here — it must return in <5ms.
    """
    return {"status": "healthy", "version": settings.APP_VERSION}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
    )
