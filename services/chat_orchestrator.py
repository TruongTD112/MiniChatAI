"""
Bộ điều phối trung tâm cho chat bot
"""
import time
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
import logging

from services.intent_service import IntentService
from services.gemini_service import GeminiService
from services.context_builders import (
    GreetingsContextBuilder,
    StoreInfoContextBuilder,
    PolicyShippingContextBuilder,
    ProductSearchTextContextBuilder,
    ProductSearchImageContextBuilder,
    ProductUsageContextBuilder,
    OthersContextBuilder,
    PlaceOrderContextBuilder,
    HistoryInquiryContextBuilder
)

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Bộ điều phối trung tâm để xử lý chat"""
    
    def __init__(self, db: Session):
        """
        Khởi tạo orchestrator
        
        Args:
            db: Database session
        """
        self.db = db
        self.intent_service = IntentService()
        self.gemini_service = GeminiService()
        
        # Map intent type đến context builder
        self.context_builders = {
            'greetings': GreetingsContextBuilder,
            'store_info': StoreInfoContextBuilder,
            'policy_shipping': PolicyShippingContextBuilder,
            'product_search_text': ProductSearchTextContextBuilder,
            'product_search_image': ProductSearchImageContextBuilder,
            'product_usage': ProductUsageContextBuilder,
            'others': OthersContextBuilder,
            'place_order': PlaceOrderContextBuilder,
            'history_inquiry': HistoryInquiryContextBuilder
        }
    
    def process_chat(
        self,
        message: str,
        conversations: List[Dict],
        customer_id: int,
        business_id: int
    ) -> Dict:
        """
        Xử lý chat request
        
        Args:
            message: Tin nhắn hiện tại
            conversations: Lịch sử trò chuyện
            customer_id: ID khách hàng
            business_id: ID business
            
        Returns:
            Dict: {'response': str, 'intent': str, 'confidence': float}
        """
        total_start_time = time.perf_counter()
        try:
            # Bước 1: Lấy danh sách intent đang bật của business
            db_start_time = time.perf_counter()
            available_intents = self.intent_service.get_active_intents_by_business(
                self.db, business_id
            )
            db_elapsed = time.perf_counter() - db_start_time
            logger.info(f"[DB Query] Lấy intent cho business_id={business_id} - Thời gian: {db_elapsed:.3f}s")
            
            if not available_intents:
                logger.warning(f"Không tìm thấy intent nào cho business_id={business_id}")
                # Fallback về intent others
                intent_result = {
                    'intent': 'others',
                    'confidence': 0.5,
                    'related_intents': []
                }
            else:
                # Bước 2: Phân loại intent bằng Gemini
                intent_result = self.gemini_service.classify_intent(
                    message=message,
                    conversations=conversations,
                    available_intents=available_intents
                )
            
            intent_type = intent_result.get('intent', 'others')
            confidence = intent_result.get('confidence', 0.5)
            
            # Bước 3: Xây dựng context dựa trên intent
            context_start_time = time.perf_counter()
            context_builder_class = self.context_builders.get(
                intent_type,
                OthersContextBuilder  # Fallback về others
            )
            
            context_builder = context_builder_class(
                db=self.db,
                business_id=business_id,
                customer_id=customer_id
            )
            
            context = context_builder.build_context(
                message=message,
                conversations=conversations
            )
            context_elapsed = time.perf_counter() - context_start_time
            logger.info(f"[Context Builder] Xây dựng context cho intent '{intent_type}' - Thời gian: {context_elapsed:.3f}s")
            
            # Bước 4: Gọi Gemini để tạo phản hồi
            response = self.gemini_service.generate_response(
                message=message,
                conversations=conversations,
                context=context,
                intent=intent_type
            )
            
            total_elapsed = time.perf_counter() - total_start_time
            logger.info(
                f"[Tổng thời gian] Xử lý chat request - business_id={business_id}, intent={intent_type} - "
                f"Thời gian tổng: {total_elapsed:.3f}s"
            )
            
            return {
                'response': response,
                'intent': intent_type,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý chat: {str(e)}")
            # Trả về phản hồi mặc định khi có lỗi
            return {
                'response': 'Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau.',
                'intent': 'others',
                'confidence': 0.0
            }

