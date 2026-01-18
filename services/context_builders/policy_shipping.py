"""
Context builder cho intent policy_shipping
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.business import Business


class PolicyShippingContextBuilder(BaseContextBuilder):
    """Context builder cho policy_shipping intent"""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho policy_shipping"""
        try:
            # Lấy thông tin business
            business = self.db.query(Business).filter(
                Business.id == self.business_id,
                Business.status == 1
            ).first()
            
            context_parts = [
                "Intent: Chính sách vận chuyển và đổi trả",
                "Khách hàng đang hỏi về chính sách vận chuyển, đổi trả, hoặc các chính sách khác."
            ]
            
            if business and business.meta_data:
                # Tìm thông tin về shipping trong metadata
                metadata = business.meta_data
                if isinstance(metadata, dict):
                    if 'shipping_policy' in metadata:
                        context_parts.append(f"Chính sách vận chuyển: {metadata['shipping_policy']}")
                    if 'return_policy' in metadata:
                        context_parts.append(f"Chính sách đổi trả: {metadata['return_policy']}")
                    if 'payment_policy' in metadata:
                        context_parts.append(f"Chính sách thanh toán: {metadata['payment_policy']}")
            
            context_parts.append(
                "Hãy giải thích rõ ràng về các chính sách. Nếu không có thông tin chi tiết, "
                "hãy nói rõ và đề xuất khách hàng liên hệ trực tiếp."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Chính sách vận chuyển. Hãy giải thích về chính sách vận chuyển và đổi trả."

