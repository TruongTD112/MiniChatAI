"""
Model cho bảng Business
"""
from sqlalchemy import Column, BigInteger, String, Text, Integer, TIMESTAMP, JSON
from sqlalchemy.sql import func
from datetime import datetime
import json
from models.base import Base


class Business(Base):
    """Model đại diện cho bảng Business"""
    __tablename__ = 'Business'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    phone = Column(String(20))
    address = Column(String(255))
    description = Column(Text)
    status = Column(Integer, default=1, comment='1: active, 0: inactive')
    meta_data = Column('metadata', JSON)  # Đổi tên attribute để tránh conflict với SQLAlchemy metadata
    style = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Chuyển đổi model thành dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'phone': self.phone,
            'address': self.address,
            'description': self.description,
            'status': self.status,
            'metadata': self.meta_data if self.meta_data else None,
            'style': self.style if self.style else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

