"""
Service để tạo embeddings từ text
"""
from typing import List
import logging
from openai import OpenAI
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service để tạo embeddings từ text sử dụng OpenAI"""
    
    def __init__(self):
        """Khởi tạo OpenAI client"""
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY không được tìm thấy trong environment variables")
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        self.model = Config.EMBEDDING_MODEL
        self.dimension = Config.EMBEDDING_DIMENSION
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Tạo embedding vector từ text
        
        Args:
            text: Text cần tạo embedding
        
        Returns:
            List[float]: Vector embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {str(e)}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Tạo embeddings cho nhiều texts cùng lúc
        
        Args:
            texts: List các texts cần tạo embedding
        
        Returns:
            List[List[float]]: List các vectors
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension
            )
            
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings batch: {str(e)}")
            raise

