"""
Service để tương tác với Pinecone Vector Database
"""
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import logging
from config import Config

logger = logging.getLogger(__name__)


class PineconeService:
    """Service quản lý kết nối và thao tác với Pinecone"""
    
    def __init__(self):
        """Khởi tạo Pinecone client"""
        if not Config.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY không được tìm thấy trong environment variables")
        
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index_name = Config.PINECONE_INDEX_NAME
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Đảm bảo index tồn tại, nếu chưa thì tạo mới"""
        try:
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                # Tạo index mới với dimension từ config
                self.pc.create_index(
                    name=self.index_name,
                    dimension=Config.PINECONE_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud=Config.PINECONE_CLOUD,
                        region=Config.PINECONE_REGION
                    )
                )
                logger.info(f"Đã tạo index mới: {self.index_name}")
            else:
                logger.info(f"Index {self.index_name} đã tồn tại")
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra/tạo index: {str(e)}")
            raise
    
    def get_index(self):
        """Lấy index object"""
        return self.pc.Index(self.index_name)
    
    def upsert_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Dict[str, Any],
        namespace: str
    ) -> bool:
        """
        Lưu hoặc cập nhật vector vào Pinecone
        
        Args:
            vector_id: ID duy nhất của vector (thường là product_id)
            vector: List các giá trị float của vector embedding
            metadata: Metadata kèm theo vector
            namespace: Namespace trong Pinecone
        
        Returns:
            bool: True nếu thành công
        """
        try:
            index = self.get_index()
            
            # Chuẩn bị metadata - Pinecone chỉ hỗ trợ một số kiểu dữ liệu
            pinecone_metadata = self._prepare_metadata(metadata)
            
            # Upsert vector
            index.upsert(
                vectors=[{
                    'id': str(vector_id),
                    'values': vector,
                    'metadata': pinecone_metadata
                }],
                namespace=namespace
            )
            
            logger.info(f"Đã upsert vector {vector_id} vào namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi upsert vector: {str(e)}")
            raise
    
    def search_vectors(
        self,
        query_vector: List[float],
        namespace: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm vectors tương tự
        
        Args:
            query_vector: Vector query để tìm kiếm
            namespace: Namespace trong Pinecone
            top_k: Số lượng kết quả trả về
            filter: Filter metadata (ví dụ: {'status': '1'})
        
        Returns:
            List các kết quả với id, score, và metadata
        """
        try:
            index = self.get_index()
            
            # Chuẩn bị filter nếu có
            pinecone_filter = self._prepare_filter(filter) if filter else None
            
            # Search
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Format kết quả
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata or {}
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Lỗi khi search vectors: {str(e)}")
            raise
    
    def delete_vector(self, vector_id: str, namespace: str) -> bool:
        """
        Xóa vector khỏi Pinecone
        
        Args:
            vector_id: ID của vector cần xóa
            namespace: Namespace trong Pinecone
        
        Returns:
            bool: True nếu thành công
        """
        try:
            index = self.get_index()
            index.delete(ids=[str(vector_id)], namespace=namespace)
            logger.info(f"Đã xóa vector {vector_id} từ namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa vector: {str(e)}")
            raise
    
    def delete_all_vectors(self, namespace: str) -> bool:
        """
        Xóa tất cả vectors trong namespace
        
        Args:
            namespace: Namespace cần xóa
        
        Returns:
            bool: True nếu thành công
        """
        try:
            index = self.get_index()
            index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Đã xóa tất cả vectors trong namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa tất cả vectors: {str(e)}")
            raise
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuẩn bị metadata cho Pinecone
        Pinecone chỉ hỗ trợ: str, int, float, bool, list[str], list[int], list[float]
        """
        pinecone_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_metadata[key] = value
            elif isinstance(value, list):
                # Kiểm tra kiểu của list
                if value and isinstance(value[0], str):
                    pinecone_metadata[key] = value
                elif value and isinstance(value[0], int):
                    pinecone_metadata[key] = value
                elif value and isinstance(value[0], float):
                    pinecone_metadata[key] = value
            elif value is None:
                # Bỏ qua None values
                continue
            else:
                # Convert sang string nếu không hỗ trợ
                pinecone_metadata[key] = str(value)
        
        return pinecone_metadata
    
    def _prepare_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuẩn bị filter cho Pinecone query
        Pinecone hỗ trợ filter với các toán tử: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
        """
        # Nếu filter đơn giản, convert sang format Pinecone
        pinecone_filter = {}
        for key, value in filter_dict.items():
            if isinstance(value, (str, int, float, bool)):
                pinecone_filter[key] = {"$eq": value}
            elif isinstance(value, list):
                pinecone_filter[key] = {"$in": value}
            else:
                pinecone_filter[key] = value
        
        return pinecone_filter

