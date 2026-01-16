"""
Schema cho API Response chuẩn
"""
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """Base response schema với format chuẩn"""
    code: str
    message: str
    data: Optional[T] = None

    class Config:
        from_attributes = True


class SuccessResponse(BaseResponse[T]):
    """Response cho thành công (code 200)"""
    code: str = "200"
    message: str = "Thành công"


class ErrorResponse(BaseResponse[None]):
    """Response cho lỗi hệ thống (code 96)"""
    code: str = "96"
    message: str
    data: None = None

