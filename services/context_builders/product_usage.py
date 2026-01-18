"""
Context builder cho intent product_usage
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.product import Product


class ProductUsageContextBuilder(BaseContextBuilder):
    """Context builder cho product_usage intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho product_usage"""
        try:
            context_parts = [
                "Intent: Hướng dẫn sử dụng sản phẩm",
                "Khách hàng đang hỏi về cách sử dụng, bảo quản, hoặc thông tin chi tiết về sản phẩm."
            ]
            
            # Tìm tên sản phẩm trong message hoặc lịch sử trò chuyện
            # Có thể cải thiện bằng cách extract tên sản phẩm từ message
            context_parts.append(
                "Hãy cung cấp hướng dẫn sử dụng chi tiết, cách bảo quản, và các lưu ý quan trọng. "
                "Nếu không có thông tin cụ thể, hãy đề xuất khách hàng xem thông tin trên website "
                "hoặc liên hệ trực tiếp để được tư vấn."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Hướng dẫn sử dụng sản phẩm. Hãy cung cấp hướng dẫn chi tiết."


