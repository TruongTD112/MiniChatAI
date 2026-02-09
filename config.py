"""
Configuration cho ứng dụng
"""
import os
from dotenv import load_dotenv

# Load environment variables từ file .env
load_dotenv()


class Config:
    """Class chứa các cấu hình của ứng dụng"""
    
    # Database MySQL
    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_PORT = os.getenv('MYSQL_PORT')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
    MYSQL_USERNAME = os.getenv('MYSQL_USERNAME')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    
    # Database URL (có thể set trực tiếp hoặc tự động tạo từ các biến trên)
    _db_url = os.getenv('DATABASE_URL')
    if not _db_url and all([MYSQL_HOST, MYSQL_PORT, MYSQL_DATABASE, MYSQL_USERNAME, MYSQL_PASSWORD]):
        _db_url = f'mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}'
    DATABASE_URL = _db_url
    
    # Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'products')
    PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '1408'))
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    
    # Google Vertex AI (cho embedding)
    # Mặc định tìm file vertexAI.json trong thư mục services/
    _default_credentials = os.path.join(os.path.dirname(__file__), 'services', 'vertexAI.json')
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', _default_credentials)
    GOOGLE_PROJECT_ID = os.getenv('GOOGLE_PROJECT_ID')
    GOOGLE_LOCATION = os.getenv('GOOGLE_LOCATION', 'us-central1')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1408'))
    
    # Gemini LLM
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Giới hạn tải ảnh (bytes) khi tạo embedding - giảm băng thông
    MAX_IMAGE_DOWNLOAD_BYTES = int(os.getenv('MAX_IMAGE_DOWNLOAD_BYTES', '2_097_152'))  # mặc định 2MB
    
    @classmethod
    def validate(cls):
        """Validate các cấu hình bắt buộc"""
        errors = []
        
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY không được tìm thấy")
        
        if not cls.GOOGLE_APPLICATION_CREDENTIALS:
            errors.append("GOOGLE_APPLICATION_CREDENTIALS không được tìm thấy")
        
        if not cls.GOOGLE_PROJECT_ID:
            errors.append("GOOGLE_PROJECT_ID không được tìm thấy")
        
        if cls.PINECONE_DIMENSION != cls.EMBEDDING_DIMENSION:
            errors.append(
                f"PINECONE_DIMENSION ({cls.PINECONE_DIMENSION}) "
                f"phải khớp với EMBEDDING_DIMENSION ({cls.EMBEDDING_DIMENSION})"
            )
        
        if errors:
            raise ValueError("Lỗi cấu hình:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True

