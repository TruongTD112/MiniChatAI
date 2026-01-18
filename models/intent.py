"""
Model cho bảng Intent
"""
from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from sqlalchemy.sql import func
from datetime import datetime
from models.base import Base


class Intent(Base):
    """Model đại diện cho bảng Intent"""
    __tablename__ = 'Intent'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    type = Column(String(255))
    template = Column(Text)
    status = Column(Integer, default=1)
    description = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

    def to_dict(self):
        """Chuyển đổi model thành dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'template': self.template,
            'status': self.status,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

