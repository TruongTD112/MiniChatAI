"""
Context builder cho intent place_order
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.business import Business


class PlaceOrderContextBuilder(BaseContextBuilder):
    """Context builder cho place_order intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho place_order"""
        try:
            context_parts = [
                "Intent: Đặt hàng",
                "Khách hàng muốn đặt hàng sản phẩm."
            ]
            
            # Lấy thông tin business
            business = self.db.query(Business).filter(
                Business.id == self.business_id,
                Business.status == 1
            ).first()
            
            if business:
                if business.phone:
                    context_parts.append(f"Số điện thoại đặt hàng: {business.phone}")
                if business.address:
                    context_parts.append(f"Địa chỉ: {business.address}")
            
            context_parts.append(
                "Hãy hướng dẫn khách hàng cách đặt hàng: qua website, điện thoại, hoặc trực tiếp tại cửa hàng. "
                "Hỏi thêm thông tin về sản phẩm và số lượng khách hàng muốn đặt."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Đặt hàng. Hãy hướng dẫn khách hàng cách đặt hàng."


