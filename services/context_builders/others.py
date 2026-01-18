"""
Context builder cho intent others
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.business import Business


class OthersContextBuilder(BaseContextBuilder):
    """Context builder cho others intent (intent mặc định)"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho others"""
        try:
            context_parts = [
                "Intent: Câu hỏi khác",
                "Khách hàng đang hỏi về vấn đề không thuộc các intent đã định nghĩa."
            ]
            
            # Lấy thông tin business
            business = self.db.query(Business).filter(
                Business.id == self.business_id,
                Business.status == 1
            ).first()
            
            if business:
                context_parts.append(f"Tên cửa hàng: {business.name}")
                if business.phone:
                    context_parts.append(f"Số điện thoại hỗ trợ: {business.phone}")
            
            context_parts.append(
                "Hãy trả lời một cách hữu ích và thân thiện. Nếu không thể trả lời, "
                "hãy đề xuất khách hàng liên hệ trực tiếp qua điện thoại hoặc email để được hỗ trợ tốt hơn."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Câu hỏi khác. Hãy trả lời một cách hữu ích và thân thiện."


