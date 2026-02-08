"""
Schema cho Chat API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ConversationMessage(BaseModel):
    """Schema cho một tin nhắn trong cuộc trò chuyện"""
    role: str = Field(..., description="Vai trò: 'user' hoặc 'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")
    timestamp: Optional[str] = Field(None, description="Thời gian gửi tin nhắn")


class ChatRequest(BaseModel):
    """Schema request cho chat API"""
    message: str = Field(..., description="Tin nhắn hiện tại của khách hàng")
    conversations: List[ConversationMessage] = Field(
        default_factory=list,
        description="Danh sách các cuộc trò chuyện gần đây"
    )
    customer_id: int = Field(..., description="ID của khách hàng")
    business_id: int = Field(..., description="ID của business")


class ChatResponse(BaseModel):
    """Schema response cho chat API"""
    response: str = Field(..., description="Phản hồi từ bot")
    intent: Optional[str] = Field(None, description="Intent được phân loại")
    confidence: Optional[float] = Field(None, description="Độ tin cậy của phân loại intent")



