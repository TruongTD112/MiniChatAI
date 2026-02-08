"""
Service để tương tác với Pinecone Vector Database
"""
import time
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
            
            # Đo thời gian truy vấn vector database
            start_time = time.perf_counter()
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=pinecone_filter
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Format kết quả
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata or {}
                })
            
            logger.info(
                f"[Vector DB] Tìm kiếm trong namespace '{namespace}' - "
                f"Tìm thấy {len(formatted_results)} kết quả - "
                f"Thời gian xử lý: {elapsed_time:.3f}s"
            )
            
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
    
    def delete_vectors(self, vector_ids: List[str], namespace: str) -> bool:
        """
        Xóa nhiều vectors khỏi Pinecone
        
        Args:
            vector_ids: List các ID của vectors cần xóa
            namespace: Namespace trong Pinecone
        
        Returns:
            bool: True nếu thành công
        """
        try:
            if not vector_ids:
                return True
            
            index = self.get_index()
            index.delete(ids=[str(vid) for vid in vector_ids], namespace=namespace)
            logger.info(f"Đã xóa {len(vector_ids)} vectors từ namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi xóa vectors: {str(e)}")
            raise
    
    def upsert_vectors_batch(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str
    ) -> bool:
        """
        Lưu hoặc cập nhật nhiều vectors vào Pinecone cùng lúc
        
        Args:
            vectors: List các dict với keys: 'id', 'values', 'metadata'
            namespace: Namespace trong Pinecone
        
        Returns:
            bool: True nếu thành công
        """
        try:
            if not vectors:
                return True
            
            index = self.get_index()
            
            # Chuẩn bị vectors cho Pinecone
            pinecone_vectors = []
            for vec in vectors:
                pinecone_metadata = self._prepare_metadata(vec.get('metadata', {}))
                pinecone_vectors.append({
                    'id': str(vec['id']),
                    'values': vec['values'],
                    'metadata': pinecone_metadata
                })
            
            # Upsert batch
            index.upsert(vectors=pinecone_vectors, namespace=namespace)
            logger.info(f"Đã upsert {len(pinecone_vectors)} vectors vào namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi upsert vectors batch: {str(e)}")
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
        Pinecone hỗ trợ filter với các toán tử: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
        
        Hỗ trợ các format:
        - {"status": "1"} -> {"status": {"$eq": "1"}}
        - {"price": {"$gte": 100, "$lte": 500}} -> giữ nguyên
        - {"category": ["quần áo", "giày dép"]} -> {"category": {"$in": ["quần áo", "giày dép"]}}
        """
        pinecone_filter = {}
        
        for key, value in filter_dict.items():
            # Nếu value đã là dict với toán tử Pinecone ($eq, $gte, etc.), giữ nguyên
            if isinstance(value, dict) and any(op.startswith('$') for op in value.keys()):
                pinecone_filter[key] = value
            # Nếu là list, dùng $in
            elif isinstance(value, list):
                pinecone_filter[key] = {"$in": value}
            # Nếu là primitive type, dùng $eq
            elif isinstance(value, (str, int, float, bool)):
                pinecone_filter[key] = {"$eq": value}
            else:
                # Fallback: convert sang string và dùng $eq
                pinecone_filter[key] = {"$eq": str(value)}
        
        return pinecone_filter

