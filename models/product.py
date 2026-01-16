"""
Model cho bảng Product
"""
from sqlalchemy import Column, Integer, String, Text, DECIMAL, TIMESTAMP, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import json

Base = declarative_base()


class Product(Base):
    """Model đại diện cho bảng product"""
    __tablename__ = 'product'

    id = Column(Integer, primary_key=True, autoincrement=True)
    business_id = Column(Integer, ForeignKey('business.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(DECIMAL(10, 2), nullable=False)
    main_image_url = Column(String(255))
    detail_image_url = Column(Text)  # List ảnh ngăn cách bởi dấu phẩy
    quantity_avail = Column(Integer, default=0)
    status = Column(String(50), default='1')  # 1-available, 2-sold_out, 3-no_longer_sell
    metadata = Column(Text)  # JSON string
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        """Chuyển đổi model thành dictionary"""
        return {
            'id': self.id,
            'business_id': self.business_id,
            'name': self.name,
            'description': self.description,
            'price': float(self.price) if self.price else None,
            'main_image_url': self.main_image_url,
            'detail_image_url': self.detail_image_url,
            'quantity_avail': self.quantity_avail,
            'status': self.status,
            'metadata': json.loads(self.metadata) if self.metadata else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    def get_text_for_embedding(self):
        """
        Tạo text để tạo embedding vector từ thông tin sản phẩm
        Kết hợp name, description và metadata để tạo text đầy đủ
        """
        text_parts = []
        
        if self.name:
            text_parts.append(f"Tên sản phẩm: {self.name}")
        
        if self.description:
            text_parts.append(f"Mô tả: {self.description}")
        
        if self.metadata:
            try:
                metadata_dict = json.loads(self.metadata)
                # Thêm các thuộc tính từ metadata vào text
                for key, value in metadata_dict.items():
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
            except:
                pass
        
        return " ".join(text_parts)

