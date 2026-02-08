"""
Context builder cho intent greetings
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.business import Business


class GreetingsContextBuilder(BaseContextBuilder):
    """Context builder cho greetings intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho greetings"""
        try:
            # Lấy thông tin business
            business = self.db.query(Business).filter(
                Business.id == self.business_id,
                Business.status == 1
            ).first()
            
            context_parts = [
                "Intent: Chào hỏi khách hàng",
                "Bạn là một chatbot bán hàng thân thiện và chuyên nghiệp."
            ]
            
            if business:
                context_parts.append(f"Tên cửa hàng: {business.name}")
                if business.description:
                    context_parts.append(f"Mô tả: {business.description}")
            
            context_parts.append(
                "Hãy chào hỏi khách hàng một cách thân thiện, giới thiệu ngắn gọn về cửa hàng "
                "và hỏi xem khách hàng cần hỗ trợ gì."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Chào hỏi khách hàng. Hãy chào hỏi một cách thân thiện."



