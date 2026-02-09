"""
Context builder cho intent product_search_text
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from models.product import Product
from services.embedding_service import get_embedding_service
from services.pinecone_service import get_pinecone_service


class ProductSearchTextContextBuilder(BaseContextBuilder):
    """Context builder cho product_search_text intent"""
    
    def __init__(self, db: Session, business_id: int, customer_id: int):
        super().__init__(db, business_id, customer_id)
        self.embedding_service = get_embedding_service()
        self.pinecone_service = get_pinecone_service()
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho product_search_text"""
        try:
            context_parts = [
                "Intent: Tìm kiếm sản phẩm bằng text",
                f"Khách hàng đang tìm kiếm sản phẩm với từ khóa: '{message}'"
            ]
            
            # Tìm kiếm sản phẩm trong Pinecone
            try:
                # Tạo embedding từ message
                query_vector = self.embedding_service.create_embedding(message)
                
                # Search trong Pinecone với namespace là business_id
                namespace = f"business_{self.business_id}"
                results = self.pinecone_service.search_vectors(
                    query_vector=query_vector,
                    namespace=namespace,
                    top_k=10,  # Lấy nhiều hơn để có thể deduplicate
                    filter={'status': '1'}
                )
                
                if results:
                    # Deduplicate theo product_id - giữ lại sản phẩm có score cao nhất
                    unique_products = {}
                    for result in results:
                        vector_id = result.get('id', '')
                        metadata = result.get('metadata', {})
                        
                        # Extract product_id từ vector_id
                        # Format: {product_id}_text hoặc {product_id}_image_*
                        product_id = None
                        if '_' in vector_id:
                            parts = vector_id.split('_')
                            if parts:
                                try:
                                    product_id = int(parts[0])
                                except ValueError:
                                    # Nếu không parse được, thử lấy từ metadata
                                    product_id = metadata.get('product_id')
                        else:
                            try:
                                product_id = int(vector_id)
                            except ValueError:
                                product_id = metadata.get('product_id')
                        
                        if product_id:
                            # Giữ lại sản phẩm có score cao nhất
                            if product_id not in unique_products or result.get('score', 0) > unique_products[product_id].get('score', 0):
                                unique_products[product_id] = {
                                    'product_id': product_id,
                                    'name': metadata.get('name', 'Không có tên'),
                                    'price': metadata.get('price', 0),
                                    'score': result.get('score', 0),
                                    'metadata': metadata
                                }
                    
                    # Sắp xếp theo score và lấy top 5
                    sorted_products = sorted(
                        unique_products.values(),
                        key=lambda x: x['score'],
                        reverse=True
                    )[:5]
                    
                    if sorted_products:
                        context_parts.append("\nCác sản phẩm tìm thấy:")
                        for idx, product in enumerate(sorted_products, 1):
                            context_parts.append(
                                f"{idx}. {product['name']} - Giá: {product['price']:,.0f} VNĐ"
                            )
                    else:
                        context_parts.append("Không tìm thấy sản phẩm phù hợp.")
                else:
                    context_parts.append("Không tìm thấy sản phẩm phù hợp.")
                    
            except Exception as search_error:
                context_parts.append("Không thể tìm kiếm sản phẩm lúc này.")
            
            context_parts.append(
                "QUAN TRỌNG: Hãy trả lời NGẮN GỌN, chỉ 1-2 câu. Giới thiệu sản phẩm một cách súc tích, "
                "không dài dòng. Nếu có nhiều sản phẩm, chỉ liệt kê tên và giá ngắn gọn. "
                "Nếu không tìm thấy, chỉ cần nói ngắn gọn và hỏi từ khóa khác."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return f"Intent: Tìm kiếm sản phẩm. Từ khóa: '{message}'"

