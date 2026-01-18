"""
API routes cho Chat Bot
"""
from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
import logging

from schemas.chat import ChatRequest, ChatResponse
from schemas.response import SuccessResponse, ErrorResponse
from database import get_db
from services.chat_orchestrator import ChatOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat Bot"])


@router.post("/message", status_code=status.HTTP_200_OK)
async def chat_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Xử lý tin nhắn từ khách hàng và trả về phản hồi từ bot
    
    - **message**: Tin nhắn hiện tại của khách hàng
    - **conversations**: Danh sách các cuộc trò chuyện gần đây
    - **customer_id**: ID của khách hàng
    - **business_id**: ID của business
    """
    try:
        # Khởi tạo orchestrator
        orchestrator = ChatOrchestrator(db=db)
        
        # Chuyển đổi conversations sang format dict
        conversations = [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            }
            for msg in request.conversations
        ]
        
        # Xử lý chat
        result = orchestrator.process_chat(
            message=request.message,
            conversations=conversations,
            customer_id=request.customer_id,
            business_id=request.business_id
        )
        
        # Tạo response
        data = ChatResponse(
            response=result['response'],
            intent=result.get('intent'),
            confidence=result.get('confidence')
        )
        
        return SuccessResponse(
            code="200",
            message="Xử lý tin nhắn thành công",
            data=data
        )
    
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {str(e)}")
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi xử lý tin nhắn: {str(e)}"
        )


