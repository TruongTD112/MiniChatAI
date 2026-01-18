"""
Context builder cho intent history_inquiry
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder


class HistoryInquiryContextBuilder(BaseContextBuilder):
    """Context builder cho history_inquiry intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho history_inquiry"""
        try:
            context_parts = [
                "Intent: Tra cứu lịch sử đơn hàng",
                f"Khách hàng (ID: {self.customer_id}) đang hỏi về lịch sử đơn hàng."
            ]
            
            # Có thể query database để lấy lịch sử đơn hàng thực tế
            # Tạm thời chỉ có context cơ bản
            context_parts.append(
                "Hãy hỏi khách hàng về thông tin cần tra cứu: số điện thoại, mã đơn hàng, "
                "hoặc thời gian đặt hàng. Sau đó hướng dẫn cách tra cứu trên website hoặc "
                "liên hệ bộ phận chăm sóc khách hàng."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Tra cứu lịch sử đơn hàng. Hãy hướng dẫn khách hàng cách tra cứu."


