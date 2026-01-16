"""
Configuration cho ứng dụng
"""
import os
from dotenv import load_dotenv

# Load environment variables từ file .env
load_dotenv()


class Config:
    """Class chứa các cấu hình của ứng dụng"""
    
    # Database
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'mysql+pymysql://user:password@localhost:3306/dbname'
    )
    
    # Pinecone
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'products')
    PINECONE_DIMENSION = int(os.getenv('PINECONE_DIMENSION', '1536'))
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')
    
    # OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '1536'))
    
    @classmethod
    def validate(cls):
        """Validate các cấu hình bắt buộc"""
        errors = []
        
        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY không được tìm thấy")
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY không được tìm thấy")
        
        if cls.PINECONE_DIMENSION != cls.EMBEDDING_DIMENSION:
            errors.append(
                f"PINECONE_DIMENSION ({cls.PINECONE_DIMENSION}) "
                f"phải khớp với EMBEDDING_DIMENSION ({cls.EMBEDDING_DIMENSION})"
            )
        
        if errors:
            raise ValueError("Lỗi cấu hình:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True

