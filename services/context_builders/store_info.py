"""
Context builder cho intent store_info
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.business import Business


class StoreInfoContextBuilder(BaseContextBuilder):
    """Context builder cho store_info intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho store_info"""
        try:
            # Lấy thông tin business
            business = self.db.query(Business).filter(
                Business.id == self.business_id,
                Business.status == 1
            ).first()
            
            context_parts = [
                "Intent: Thông tin cửa hàng",
                "Khách hàng đang hỏi về thông tin cửa hàng."
            ]
            
            if business:
                context_parts.append(f"Tên cửa hàng: {business.name}")
                if business.phone:
                    context_parts.append(f"Số điện thoại: {business.phone}")
                if business.address:
                    context_parts.append(f"Địa chỉ: {business.address}")
                if business.description:
                    context_parts.append(f"Mô tả: {business.description}")
                if business.meta_data:
                    context_parts.append(f"Thông tin bổ sung: {business.meta_data}")
            
            context_parts.append(
                "Hãy cung cấp thông tin cửa hàng một cách đầy đủ và rõ ràng."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Thông tin cửa hàng. Hãy cung cấp thông tin cửa hàng."

