"""
Exception handler cho API
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from schemas.response import ErrorResponse
import logging

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Xử lý lỗi validation"""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        field = ".".join(str(loc) for loc in error["loc"])
        error_messages.append(f"{field}: {error['msg']}")
    
    error_message = "; ".join(error_messages)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ErrorResponse(
            code="96",
            message=f"Lỗi validation: {error_message}"
        ).dict()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Xử lý lỗi hệ thống chung"""
    logger.error(f"Lỗi hệ thống: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=ErrorResponse(
            code="96",
            message=f"Lỗi hệ thống: {str(exc)}"
        ).dict()
    )

