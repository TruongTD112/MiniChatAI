"""
API routes cho Chat Bot.
Logic chat: instruction cố định + context sản phẩm lấy từ bảng Business (và Product theo business_id).
Context được cache theo business_id — mỗi business có cache riêng.
"""
from fastapi import APIRouter, status, Depends
from sqlalchemy.orm import Session
import logging

from schemas.chat import ChatRequest, ChatResponse
from schemas.response import SuccessResponse, ErrorResponse
from database import get_db
from services.gemini_service import GeminiService
from services.business_context_service import (
    get_business_context_service,
    DEFAULT_CHAT_INSTRUCTION,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat Bot"])


@router.post("/message", status_code=status.HTTP_200_OK)
async def chat_message(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Xử lý tin nhắn từ khách hàng và trả về phản hồi từ bot.

    - **message**: Tin nhắn hiện tại của khách hàng
    - **conversations**: Danh sách các cuộc trò chuyện gần đây
    - **customer_id**: ID của khách hàng
    - **business_id**: ID của business (dùng để lấy thông tin cửa hàng + sản phẩm; context cache theo business_id)
    """
    try:
        business_id = request.business_id

        # Lấy context sản phẩm từ Business + Product, cache theo business_id
        context_service = get_business_context_service()
        product_context = context_service.get_product_context(db, business_id, use_cache=False)

        # Chuyển conversations sang format cho GeminiService
        conversations = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversations
        ]

        # Instruction cố định; context sản phẩm theo từng business
        gemini_service = GeminiService()
        response_text = gemini_service.generate_chat_response(
            message=request.message,
            conversations=conversations,
            instruction=DEFAULT_CHAT_INSTRUCTION,
            product_context=product_context,
        )

        data = ChatResponse(
            response=response_text,
            intent=None,
            confidence=None,
        )
        return SuccessResponse(
            code="200",
            message="Xử lý tin nhắn thành công",
            data=data,
        )
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {str(e)}")
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi xử lý tin nhắn: {str(e)}",
        )



