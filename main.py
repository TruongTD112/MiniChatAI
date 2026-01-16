"""
Main application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import logging

from config import Config
from api.routes import product_vector
from middleware.exception_handler import (
    validation_exception_handler,
    general_exception_handler
)

# Validate config khi khởi động
try:
    Config.validate()
except ValueError as e:
    logging.error(str(e))
    raise

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Khởi tạo FastAPI app
app = FastAPI(
    title="Product Vector API",
    description="API để quản lý vector database cho sản phẩm sử dụng Pinecone",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production, nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Đăng ký routers
app.include_router(product_vector.router)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Product Vector API đang hoạt động",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint chi tiết"""
    return {
        "status": "healthy",
        "service": "Product Vector API"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

