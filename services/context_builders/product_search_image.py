"""
Context builder cho intent product_search_image
"""
from typing import Dict, List
from sqlalchemy.orm import Session
from services.context_builders.base import BaseContextBuilder
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
import re


class ProductSearchImageContextBuilder(BaseContextBuilder):
    """Context builder cho product_search_image intent"""
    
    def __init__(self, db: Session, business_id: int, customer_id: int):
        super().__init__(db, business_id, customer_id)
        self.embedding_service = EmbeddingService()
        self.pinecone_service = PineconeService()
    
    def extract_image_url(self, message: str) -> str:
        """Trích xuất URL ảnh từ message"""
        # Tìm URL trong message
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, message)
        return urls[0] if urls else ""
    
    def build_context(self, message: str, conversations: List[Dict]) -> str:
        """Xây dựng context cho product_search_image"""
        try:
            context_parts = [
                "Intent: Tìm kiếm sản phẩm bằng hình ảnh",
                "Khách hàng đang tìm kiếm sản phẩm tương tự bằng cách gửi hình ảnh."
            ]
            
            # Trích xuất URL ảnh từ message
            image_url = self.extract_image_url(message)
            
            if image_url:
                try:
                    # Tạo embedding từ ảnh
                    image_vector = self.embedding_service.create_image_embedding(image_url)
                    
                    if image_vector:
                        # Search trong Pinecone
                        namespace = f"business_{self.business_id}"
                        results = self.pinecone_service.search_vectors(
                            query_vector=image_vector,
                            namespace=namespace,
                            top_k=10,  # Lấy nhiều hơn để có thể deduplicate
                            filter={'status': '1', 'business_id': self.business_id, 'vector_type': 'image'}
                        )
                        
                        if results:
                            # Deduplicate theo product_id - giữ lại sản phẩm có score cao nhất
                            unique_products = {}
                            for result in results:
                                vector_id = result.get('id', '')
                                metadata = result.get('metadata', {})
                                
                                # Extract product_id từ vector_id
                                product_id = None
                                if '_' in vector_id:
                                    parts = vector_id.split('_')
                                    if parts:
                                        try:
                                            product_id = int(parts[0])
                                        except ValueError:
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
                                context_parts.append("\nCác sản phẩm tương tự tìm thấy:")
                                for idx, product in enumerate(sorted_products, 1):
                                    context_parts.append(
                                        f"{idx}. {product['name']} - Giá: {product['price']:,.0f} VNĐ"
                                    )
                            else:
                                context_parts.append("Không tìm thấy sản phẩm tương tự.")
                        else:
                            context_parts.append("Không tìm thấy sản phẩm tương tự.")
                    else:
                        context_parts.append("Không thể xử lý hình ảnh được gửi.")
                        
                except Exception as search_error:
                    context_parts.append("Không thể tìm kiếm sản phẩm lúc này.")
            else:
                context_parts.append("Không tìm thấy URL hình ảnh trong tin nhắn.")
            
            context_parts.append(
                "QUAN TRỌNG: Hãy trả lời NGẮN GỌN, chỉ 1-2 câu. Giới thiệu sản phẩm một cách súc tích, "
                "không dài dòng. Nếu có nhiều sản phẩm, chỉ liệt kê tên và giá ngắn gọn. "
                "Nếu không tìm thấy, chỉ cần nói ngắn gọn và đề xuất mô tả bằng text."
            )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            return "Intent: Tìm kiếm sản phẩm bằng hình ảnh."

