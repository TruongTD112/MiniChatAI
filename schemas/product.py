"""
Schema Pydantic cho Product API
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ProductBase(BaseModel):
    """Schema cơ bản cho Product"""
    business_id: int
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    main_image_url: Optional[str] = Field(None, max_length=255)
    detail_image_url: Optional[str] = None
    quantity_avail: int = Field(default=0, ge=0)
    status: str = Field(default='1', pattern='^(1|2|3)$')
    metadata: Optional[Dict[str, Any]] = None


class ProductCreate(ProductBase):
    """Schema để tạo mới Product"""
    pass


class ProductUpdate(BaseModel):
    """Schema để cập nhật Product"""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    price: Optional[float] = Field(None, gt=0)
    main_image_url: Optional[str] = Field(None, max_length=255)
    detail_image_url: Optional[str] = None
    quantity_avail: Optional[int] = Field(None, ge=0)
    status: Optional[str] = Field(None, pattern='^(1|2|3)$')
    metadata: Optional[Dict[str, Any]] = None


class ProductResponse(ProductBase):
    """Schema response cho Product"""
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProductVectorRequest(BaseModel):
    """Schema để lưu vector cho product - nhận toàn bộ thông tin sản phẩm từ backend"""
    product_id: int = Field(..., description="ID của sản phẩm (dùng làm vector_id trong Pinecone)")
    namespace: str = Field(..., description="Namespace trong Pinecone để lưu vector")
    # Thông tin sản phẩm từ backend
    business_id: int
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    price: float = Field(..., gt=0)
    main_image_url: Optional[str] = Field(None, max_length=255)
    detail_image_url: Optional[str] = None
    quantity_avail: int = Field(default=0, ge=0)
    status: str = Field(default='1', pattern='^(1|2|3)$')
    metadata: Optional[Dict[str, Any]] = None


class ProductVectorData(BaseModel):
    """Data response sau khi lưu vector"""
    product_id: int
    namespace: str
    vector_id: str


class ProductSearchData(BaseModel):
    """Data response cho search"""
    results: List[ProductSearchResult]
    total: int


class DeleteVectorData(BaseModel):
    """Data response sau khi xóa vector"""
    product_id: int
    namespace: str


class BatchUpsertData(BaseModel):
    """Data response cho batch upsert"""
    success_count: int
    error_count: int
    results: List[Dict[str, Any]]
    errors: List[str]


class ProductSearchRequest(BaseModel):
    """Schema để search product bằng vector"""
    query_text: str = Field(..., description="Text query để tìm kiếm")
    namespace: str = Field(..., description="Namespace trong Pinecone để search")
    top_k: int = Field(default=10, ge=1, le=100, description="Số lượng kết quả trả về")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter metadata trong Pinecone")


class ProductSearchResult(BaseModel):
    """Schema cho một kết quả search"""
    product_id: int
    score: float
    product: ProductResponse


class ProductSearchResponse(BaseModel):
    """Schema response cho search"""
    results: List[ProductSearchResult]
    total: int

