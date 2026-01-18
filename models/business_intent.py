"""
Model cho bảng Business_Intent
"""
from sqlalchemy import Column, BigInteger, Integer, Text, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from datetime import datetime
from models.base import Base


class BusinessIntent(Base):
    """Model đại diện cho bảng Business_Intent"""
    __tablename__ = 'Business_Intent'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    business_id = Column(BigInteger, nullable=True)
    intent_id = Column(Integer, nullable=True)
    template_override = Column(Text)
    status = Column(Integer, default=1)
    created_at = Column(TIMESTAMP, server_default=func.now())

    def to_dict(self):
        """Chuyển đổi model thành dictionary"""
        return {
            'id': self.id,
            'business_id': self.business_id,
            'intent_id': self.intent_id,
            'template_override': self.template_override,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

