"""
Base class cho context builders
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from sqlalchemy.orm import Session


class BaseContextBuilder(ABC):
    """Base class cho tất cả context builders"""
    
    def __init__(self, db: Session, business_id: int, customer_id: int):
        """
        Khởi tạo context builder
        
        Args:
            db: Database session
            business_id: ID của business
            customer_id: ID của khách hàng
        """
        self.db = db
        self.business_id = business_id
        self.customer_id = customer_id
    
    @abstractmethod
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """
        Xây dựng context cho intent
        
        Args:
            message: Tin nhắn hiện tại
            conversations: Lịch sử trò chuyện
            
        Returns:
            str: Context đã được xây dựng
        """
        pass


