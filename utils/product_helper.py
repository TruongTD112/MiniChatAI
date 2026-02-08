"""
Helper functions cho Product operations
"""
from typing import Dict, Any, Optional, List


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
    description: Optional[str] = None,
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
        description: Mô tả sản phẩm
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
    
    # Thêm description nếu có
    if description:
        pinecone_metadata['description'] = description
    
    # Thêm metadata bổ sung nếu có
    if metadata:
        pinecone_metadata.update(metadata)
    
    return pinecone_metadata


def extract_image_urls(
    main_image_url: Optional[str] = None,
    detail_image_url: Optional[str] = None
) -> List[str]:
    """
    Trích xuất danh sách các image URLs từ main_image_url và detail_image_url
    
    Args:
        main_image_url: URL ảnh chính (có thể là string hoặc None)
        detail_image_url: URL ảnh chi tiết (có thể là string nhiều URL ngăn cách bởi dấu phẩy)
    
    Returns:
        List[str]: Danh sách các image URLs (không trùng lặp, loại bỏ None/empty)
    """
    image_urls = []
    
    # Thêm main_image_url nếu có
    if main_image_url and main_image_url.strip():
        image_urls.append(main_image_url.strip())
    
    # Xử lý detail_image_url (có thể là nhiều URL ngăn cách bởi dấu phẩy)
    if detail_image_url and detail_image_url.strip():
        # Tách các URL bằng dấu phẩy
        detail_urls = [url.strip() for url in detail_image_url.split(',') if url.strip()]
        image_urls.extend(detail_urls)
    
    # Loại bỏ trùng lặp nhưng giữ nguyên thứ tự
    seen = set()
    unique_urls = []
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    return unique_urls


def get_primary_image_url(
    main_image_url: Optional[str] = None,
    detail_image_url: Optional[str] = None
) -> Optional[str]:
    """
    Lấy URL ảnh chính để tạo embedding (ưu tiên main_image_url)
    
    Args:
        main_image_url: URL ảnh chính
        detail_image_url: URL ảnh chi tiết
    
    Returns:
        Optional[str]: URL ảnh chính hoặc None
    """
    if main_image_url and main_image_url.strip():
        return main_image_url.strip()
    
    if detail_image_url and detail_image_url.strip():
        # Lấy URL đầu tiên từ detail_image_url
        first_url = detail_image_url.split(',')[0].strip()
        if first_url:
            return first_url
    
    return None

