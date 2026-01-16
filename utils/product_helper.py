"""
Helper functions cho Product operations
"""
from typing import Dict, Any, Optional


def create_text_for_embedding(
    name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Tạo text để tạo embedding vector từ thông tin sản phẩm
    Kết hợp name, description và metadata để tạo text đầy đủ
    
    Args:
        name: Tên sản phẩm
        description: Mô tả sản phẩm
        metadata: Dictionary chứa các thuộc tính khác
    
    Returns:
        str: Text đã được kết hợp để tạo embedding
    """
    text_parts = []
    
    if name:
        text_parts.append(f"Tên sản phẩm: {name}")
    
    if description:
        text_parts.append(f"Mô tả: {description}")
    
    if metadata:
        # Thêm các thuộc tính từ metadata vào text
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                # Nếu là list, join thành string
                text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
    
    return " ".join(text_parts)


def prepare_metadata_for_pinecone(
    product_id: int,
    business_id: int,
    name: str,
    price: float,
    status: str,
    quantity_avail: int,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Chuẩn bị metadata để lưu vào Pinecone
    
    Args:
        product_id: ID sản phẩm
        business_id: ID business
        name: Tên sản phẩm
        price: Giá sản phẩm
        status: Trạng thái
        quantity_avail: Số lượng có sẵn
        metadata: Metadata bổ sung
    
    Returns:
        Dict: Metadata đã được chuẩn bị
    """
    pinecone_metadata = {
        'product_id': product_id,
        'business_id': business_id,
        'name': name,
        'price': float(price) if price else 0,
        'status': status,
        'quantity_avail': quantity_avail
    }
    
    # Thêm metadata bổ sung nếu có
    if metadata:
        pinecone_metadata.update(metadata)
    
    return pinecone_metadata

