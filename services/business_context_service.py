"""
Service lấy context sản phẩm từ Business và Product theo business_id.
Cache context theo business_id để mỗi business có cache riêng, giảm query DB.
"""
import json
import time
import logging
from typing import Optional
from sqlalchemy.orm import Session

from models.business import Business
from models.product import Product

logger = logging.getLogger(__name__)

# Instruction cố định cho chatbot (giống logic Gradio)
DEFAULT_CHAT_INSTRUCTION = (
    "Bạn là Chuyên viên tư vấn bán hàng tận tâm của cửa hàng. "
    "Hãy trả lời các câu hỏi về sản phẩm, thông tin cửa hàng một cách thân thiện và hữu ích. "
    "Chỉ dùng tiếng Việt. Trả lời ngắn gọn, lịch sự, tự nhiên như bạn bè, không cần quá khách sáo"
    "IMPORTANT IMPORTANT Trả lời ngắn gọn 1-2 câu dựa trên Context. mỗi câu tầm 10 chữ, xuống dòng cho mỗi câu, bỏ dấu chấm ở cuối câu"
    "Nếu không có thông tin trong context, hãy nói rõ và gửi lại nếu sau có thông tin"
    "Thiếu thông tin: Chat hỏi khách một cách khéo léo để thu thập đủ."
    "Nguyên tắc ngôn ngữ: Trả lời trực diện, ngôn ngữ đời thường (Dạ, vâng, ạ, nhé, nha, hen)"  
    "IMPORTANT cần upsell sản phẩm nếu khách hàng do dự hoặc không muốn mua, nhưng lại có sản phẩm có khả năng phù hợp"
    "IMPORTANT IMPORTANT Nếu xác định được đối tượng khách hàng đang hỏi rồi thì không cần đề cập lại tên sản phẩm nữa"
    "Nếu khách hàng hỏi ảnh hãy trả về url của ảnh"
    "IMPORTANT IMPORTANT IMPORTANT Cấm bịa thông tin không có trong Context. Confidence < 100% thì báo khách chờ để check lại."
    "Nếu khách muốn Đặt hàng: - Kiểm tra xem đã đủ 5 thông tin: Tên, ID sản phẩm, Số lượng, Địa chỉ, Số điện thoại chưa. (Thiếu thông tin: Chat hỏi khách một cách khéo léo để thu thập đủ"
)


class BusinessContextService:
    """
    Lấy và cache context sản phẩm theo business_id.
    Mỗi business_id có cache riêng; TTL (giây) để làm mới khi dữ liệu thay đổi.
    """

    def __init__(self, ttl_seconds: int = 0):
        """
        Args:
            ttl_seconds: Thời gian sống cache (mặc định 0 = không cache).
        """
        self._cache: dict[int, tuple[str, float]] = {}  # business_id -> (context, expiry_at)
        self._ttl = ttl_seconds

    def get_product_context(self, db: Session, business_id: int, use_cache: bool = False) -> str:
        """
        Lấy context sản phẩm cho business: thông tin cửa hàng + danh sách sản phẩm.
        Cache theo business_id; mỗi business có context cache riêng.

        Args:
            db: Database session
            business_id: ID business
            use_cache: Có dùng cache hay không

        Returns:
            Chuỗi context để đưa vào prompt (instruction + product context).
        """
        if use_cache and self._ttl > 0:
            now = time.time()
            cached = self._cache.get(business_id)
            if cached is not None:
                context, expiry_at = cached
                if now < expiry_at:
                    logger.debug(f"[BusinessContext] Cache hit business_id={business_id}")
                    return context
                # Hết hạn -> xóa và build lại
                self._cache.pop(business_id, None)

        context = self._build_context(db, business_id)
        if use_cache and self._ttl > 0:
            self._cache[business_id] = (context, time.time() + self._ttl)
            logger.debug(f"[BusinessContext] Cached context for business_id={business_id} (ttl={self._ttl}s)")

        return context

    def invalidate_cache(self, business_id: Optional[int] = None):
        """
        Xóa cache. Nếu business_id=None thì xóa toàn bộ.
        Gọi khi cập nhật thông tin business hoặc sản phẩm.
        """
        if business_id is not None:
            self._cache.pop(business_id, None)
            logger.info(f"[BusinessContext] Invalidated cache for business_id={business_id}")
        else:
            self._cache.clear()
            logger.info("[BusinessContext] Invalidated all context cache")

    def _build_context(self, db: Session, business_id: int) -> str:
        """Query DB: Business + Products theo business_id, build chuỗi context."""
        parts = []

        # 1. Thông tin cửa hàng từ bảng Business
        business = db.query(Business).filter(
            Business.id == business_id,
            Business.status == 1
        ).first()

        if business:
            parts.append("--- THÔNG TIN CỬA HÀNG ---")
            parts.append(f"Tên cửa hàng: {business.name}")
            if business.phone:
                parts.append(f"Số điện thoại: {business.phone}")
            if business.address:
                parts.append(f"Địa chỉ: {business.address}")
            if business.description:
                parts.append(f"Mô tả: {business.description}")
            if business.meta_data:
                if isinstance(business.meta_data, dict):
                    for k, v in business.meta_data.items():
                        parts.append(f"{k}: {v}")
                else:
                    parts.append(f"Thông tin bổ sung: {business.meta_data}")
            parts.append("")

        # 2. Danh sách sản phẩm từ bảng Product (status = '1' = available)
        products = db.query(Product).filter(
            Product.business_id == business_id,
            Product.status == '1'
        ).all()

        if products:
            parts.append("--- DANH SÁCH SẢN PHẨM ---")
            for p in products:
                item = [
                    f"Tên: {p.name}",
                    f"Giá: {float(p.price):,.0f} VNĐ" if p.price else "",
                    f"Mô tả: {p.description}" if p.description else "",
                ]
                if p.main_image_url:
                    item.append(f"Ảnh chính: {p.main_image_url}")
                if p.detail_image_url:
                    item.append(f"Ảnh chi tiết: {p.detail_image_url}")
                if p.quantity_avail is not None:
                    item.append(f"Số lượng còn: {p.quantity_avail}")
                if p.meta_data:
                    try:
                        meta = json.loads(p.meta_data) if isinstance(p.meta_data, str) else p.meta_data
                        if isinstance(meta, dict):
                            for k, v in meta.items():
                                item.append(f"{k}: {v}")
                    except Exception:
                        pass
                parts.append("\n".join(x for x in item if x))
                parts.append("")
            parts.append("")
        else:
            parts.append("--- DANH SÁCH SẢN PHẨM ---")
            parts.append("Chưa có sản phẩm nào.")
            parts.append("")

        return "\n".join(parts).strip()


# Singleton cho app (cache dùng chung, mỗi business_id một context)
_business_context_service: Optional[BusinessContextService] = None


def get_business_context_service(ttl_seconds: int = 0) -> BusinessContextService:
    """Lấy singleton BusinessContextService."""
    global _business_context_service
    if _business_context_service is None:
        _business_context_service = BusinessContextService(ttl_seconds=ttl_seconds)
    return _business_context_service
