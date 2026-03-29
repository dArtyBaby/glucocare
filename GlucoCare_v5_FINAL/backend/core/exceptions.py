# core/exceptions.py
"""
WHY global exception handlers:
Without these, any unhandled exception in your app returns a raw 500 with
a Python traceback — leaking internal implementation details to attackers
and confusing to users.

Global handlers catch specific exception types and return structured,
safe JSON responses. The pattern is: catch broadly, log fully, respond minimally.

RULE: Never return stack traces or internal error messages to clients in production.
"""
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import OperationalError, IntegrityError

logger = logging.getLogger(__name__)


class GlucoException(Exception):
    """Base application exception. All custom errors inherit from this."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class UserAlreadyExistsError(GlucoException):
    def __init__(self, email: str):
        super().__init__(f"A user with email '{email}' already exists.", 409)


class UserNotFoundError(GlucoException):
    def __init__(self, user_id: str):
        super().__init__(f"User with ID '{user_id}' was not found.", 404)


class RewardAlreadyClaimedError(GlucoException):
    def __init__(self):
        super().__init__("Daily reward already claimed. Come back tomorrow!", 409)


class InvalidGlucoseReadingError(GlucoException):
    def __init__(self, value: float):
        super().__init__(f"Glucose reading {value} mg/dL is outside valid range (20–600).", 422)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers to the FastAPI app instance."""

    @app.exception_handler(GlucoException)
    async def gluco_exception_handler(request: Request, exc: GlucoException):
        """Our custom domain exceptions — safe to expose message to client."""
        logger.warning(f"Domain error on {request.url}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """
        Pydantic validation failures — return which fields failed and why.
        WHY format like this: Frontend can map errors directly to form fields.
        """
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })
        logger.info(f"Validation error on {request.url}: {errors}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"success": False, "error": "Validation failed", "details": errors},
        )

    @app.exception_handler(OperationalError)
    async def db_operational_handler(request: Request, exc: OperationalError):
        """
        Database connection failures — tell client service is unavailable.
        NEVER expose the connection string or SQL error to the client.
        """
        logger.critical(f"Database operational error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": "Database temporarily unavailable. Please try again."},
        )

    @app.exception_handler(IntegrityError)
    async def db_integrity_handler(request: Request, exc: IntegrityError):
        """Duplicate keys, foreign key violations, etc."""
        logger.error(f"Database integrity error: {exc}")
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"success": False, "error": "A data conflict occurred. This record may already exist."},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """
        Catch-all for anything we didn't anticipate.
        Log the full traceback internally but return a safe generic message.
        """
        logger.critical(
            f"Unhandled exception on {request.method} {request.url}: {type(exc).__name__}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "An internal server error occurred."},
        )
