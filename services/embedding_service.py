"""
Service để tạo embeddings từ text và image sử dụng Google Vertex AI
"""
from typing import List, Optional
import logging
import os
import time
import base64
import requests
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from config import Config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service để tạo embeddings từ text và image sử dụng Google Vertex AI"""
    
    def __init__(self):
        """Khởi tạo Google Vertex AI client"""
        # Kiểm tra và xử lý credentials path
        credentials_path = Config.GOOGLE_APPLICATION_CREDENTIALS
        
        # Convert relative path thành absolute path nếu cần
        if credentials_path and not os.path.isabs(credentials_path):
            # Nếu là relative path, thử tìm từ thư mục gốc project
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            credentials_path = os.path.join(base_dir, credentials_path)
            # Nếu không tìm thấy, thử từ thư mục services
            if not os.path.exists(credentials_path):
                services_dir = os.path.dirname(os.path.abspath(__file__))
                credentials_path = os.path.join(services_dir, os.path.basename(Config.GOOGLE_APPLICATION_CREDENTIALS))
        
        # Kiểm tra file có tồn tại không
        if not credentials_path or not os.path.exists(credentials_path):
            raise ValueError(
                f"File credentials không tồn tại: {credentials_path}\n"
                f"Vui lòng đảm bảo file vertexAI.json tồn tại hoặc set GOOGLE_APPLICATION_CREDENTIALS trong .env"
            )
        
        if not Config.GOOGLE_PROJECT_ID:
            raise ValueError("GOOGLE_PROJECT_ID không được tìm thấy trong environment variables")
        
        # Set credentials path (phải là absolute path)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(credentials_path)
        logger.info(f"Sử dụng credentials từ: {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
        
        # Khởi tạo Vertex AI
        self.project_id = Config.GOOGLE_PROJECT_ID
        self.location = Config.GOOGLE_LOCATION
        self.dimension = Config.EMBEDDING_DIMENSION
        
        try:
            aiplatform.init(project=self.project_id, location=self.location)
            self.client = aiplatform.gapic.PredictionServiceClient(
                client_options={"api_endpoint": f"{self.location}-aiplatform.googleapis.com"}
            )
            logger.info(f"Đã khởi tạo Vertex AI client cho project {self.project_id}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo Vertex AI client: {str(e)}")
            raise
        
        # Endpoint cho multimodal embedding model
        self.endpoint = f"projects/{self.project_id}/locations/{self.location}/publishers/google/models/multimodalembedding@001"
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Tạo embedding vector từ text sử dụng Google Vertex AI
        
        Args:
            text: Text cần tạo embedding
        
        Returns:
            List[float]: Vector embedding
        """
        try:
            # Tạo instance với text
            instance = struct_pb2.Struct()
            instance.update({"text": text})
            
            # Set dimension
            parameters = struct_pb2.Struct()
            parameters.update({"dimension": self.dimension})
            
            # Đo thời gian tạo embedding
            start_time = time.perf_counter()
            res = self.client.predict(
                endpoint=self.endpoint,
                instances=[instance],
                parameters=parameters
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Lấy text embedding
            embedding = list(res.predictions[0]['textEmbedding'])
            
            logger.info(
                f"[Embedding] Tạo text embedding - dimension: {len(embedding)} - "
                f"Thời gian xử lý: {elapsed_time:.3f}s"
            )
            return embedding
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo text embedding: {str(e)}")
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
            # Vertex AI không hỗ trợ batch trực tiếp, gọi từng cái
            embeddings = []
            for text in texts:
                embedding = self.create_embedding(text)
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            logger.error(f"Lỗi khi tạo embeddings batch: {str(e)}")
            raise
    
    def create_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """
        Tạo embedding vector từ ảnh (URL) sử dụng Google Vertex AI
        
        Args:
            image_url: URL của ảnh cần tạo embedding
        
        Returns:
            List[float]: Vector embedding hoặc None nếu không thể tạo
        """
        try:
            if not image_url:
                return None
            
            # Download ảnh từ URL và convert sang base64
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            
            # Tạo instance với image
            instance = struct_pb2.Struct()
            instance.update({"image": {"bytesBase64Encoded": image_base64}})
            
            # Set dimension
            parameters = struct_pb2.Struct()
            parameters.update({"dimension": self.dimension})
            
            # Đo thời gian tạo embedding
            start_time = time.perf_counter()
            res = self.client.predict(
                endpoint=self.endpoint,
                instances=[instance],
                parameters=parameters
            )
            elapsed_time = time.perf_counter() - start_time
            
            # Lấy image embedding
            embedding = list(res.predictions[0]['imageEmbedding'])
            
            logger.info(
                f"[Embedding] Tạo image embedding cho {image_url} - dimension: {len(embedding)} - "
                f"Thời gian xử lý: {elapsed_time:.3f}s"
            )
            return embedding
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo image embedding từ {image_url}: {str(e)}")
            return None
    
    def create_image_embeddings_batch(self, image_urls: List[str]) -> List[Optional[List[float]]]:
        """
        Tạo embeddings cho nhiều ảnh cùng lúc
        
        Args:
            image_urls: List các URLs của ảnh
        
        Returns:
            List[Optional[List[float]]]: List các vectors (có thể None nếu không tạo được)
        """
        embeddings = []
        for image_url in image_urls:
            embedding = self.create_image_embedding(image_url)
            embeddings.append(embedding)
        return embeddings

