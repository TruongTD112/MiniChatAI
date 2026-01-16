"""
API routes cho Product Vector operations
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from schemas.product import (
    ProductVectorRequest,
    ProductVectorData,
    ProductSearchRequest,
    ProductSearchData,
    ProductSearchResult,
    DeleteVectorData,
    BatchUpsertData
)
from schemas.response import SuccessResponse, ErrorResponse
from services.pinecone_service import PineconeService
from services.embedding_service import EmbeddingService
from utils.product_helper import create_text_for_embedding, prepare_metadata_for_pinecone

router = APIRouter(prefix="/api/products/vector", tags=["Product Vector"])

# Khởi tạo services
pinecone_service = PineconeService()
embedding_service = EmbeddingService()


@router.post("/upsert", status_code=status.HTTP_200_OK)
async def upsert_product_vector(
    request: ProductVectorRequest
):
    """
    Lưu hoặc cập nhật vector cho một sản phẩm vào Pinecone
    
    Backend gửi toàn bộ thông tin sản phẩm trong request body:
    - **product_id**: ID của sản phẩm (dùng làm vector_id trong Pinecone)
    - **namespace**: Namespace trong Pinecone để lưu vector
    - **business_id**: ID của business
    - **name**: Tên sản phẩm
    - **description**: Mô tả sản phẩm
    - **price**: Giá sản phẩm
    - **status**: Trạng thái sản phẩm
    - **quantity_avail**: Số lượng có sẵn
    - **metadata**: Metadata bổ sung (optional)
    """
    try:
        # Tạo text để embedding từ thông tin sản phẩm trong request
        text_for_embedding = create_text_for_embedding(
            name=request.name,
            description=request.description,
            metadata=request.metadata
        )
        
        # Tạo embedding vector
        vector = embedding_service.create_embedding(text_for_embedding)
        
        # Chuẩn bị metadata để lưu vào Pinecone
        metadata = prepare_metadata_for_pinecone(
            product_id=request.product_id,
            business_id=request.business_id,
            name=request.name,
            price=request.price,
            status=request.status,
            quantity_avail=request.quantity_avail,
            metadata=request.metadata
        )
        
        # Lưu vector vào Pinecone
        vector_id = str(request.product_id)
        pinecone_service.upsert_vector(
            vector_id=vector_id,
            vector=vector,
            metadata=metadata,
            namespace=request.namespace
        )
        
        data = ProductVectorData(
            product_id=request.product_id,
            namespace=request.namespace,
            vector_id=vector_id
        )
        
        return SuccessResponse(
            code="200",
            message="Đã lưu vector thành công",
            data=data
        )
    
    except Exception as e:
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi lưu vector: {str(e)}"
        )


@router.post("/search", status_code=status.HTTP_200_OK)
async def search_products_by_vector(
    request: ProductSearchRequest
):
    """
    Tìm kiếm sản phẩm bằng vector similarity search
    
    Trả về thông tin sản phẩm từ metadata trong Pinecone, không cần query database
    
    - **query_text**: Text query để tìm kiếm
    - **namespace**: Namespace trong Pinecone để search
    - **top_k**: Số lượng kết quả trả về (mặc định: 10)
    - **filter**: Filter metadata (ví dụ: {"status": "1"})
    """
    try:
        # Tạo embedding từ query text
        query_vector = embedding_service.create_embedding(request.query_text)
        
        # Search trong Pinecone
        search_results = pinecone_service.search_vectors(
            query_vector=query_vector,
            namespace=request.namespace,
            top_k=request.top_k,
            filter=request.filter
        )
        
        # Lấy thông tin sản phẩm từ metadata trong Pinecone
        results = []
        for result in search_results:
            metadata = result.get('metadata', {})
            product_id = int(result['id'])
            
            # Tạo ProductResponse từ metadata
            from schemas.product import ProductResponse
            from datetime import datetime
            
            product_response = ProductResponse(
                id=product_id,
                business_id=metadata.get('business_id', 0),
                name=metadata.get('name', ''),
                description=None,  # Không lưu description trong metadata để tiết kiệm
                price=metadata.get('price', 0),
                main_image_url=None,
                detail_image_url=None,
                quantity_avail=metadata.get('quantity_avail', 0),
                status=metadata.get('status', '1'),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ['product_id', 'business_id', 'name', 'price', 'status', 'quantity_avail']},
                created_at=datetime.now(),  # Không có trong metadata
                updated_at=datetime.now()   # Không có trong metadata
            )
            
            results.append(ProductSearchResult(
                product_id=product_id,
                score=result['score'],
                product=product_response
            ))
        
        data = ProductSearchData(
            results=results,
            total=len(results)
        )
        
        return SuccessResponse(
            code="200",
            message="Tìm kiếm thành công",
            data=data
        )
    
    except Exception as e:
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi tìm kiếm: {str(e)}"
        )


@router.delete("/{product_id}", status_code=status.HTTP_200_OK)
async def delete_product_vector(
    product_id: int,
    namespace: str
):
    """
    Xóa vector của sản phẩm khỏi Pinecone
    
    - **product_id**: ID của sản phẩm cần xóa vector
    - **namespace**: Namespace trong Pinecone
    """
    try:
        # Xóa vector từ Pinecone
        vector_id = str(product_id)
        pinecone_service.delete_vector(
            vector_id=vector_id,
            namespace=namespace
        )
        
        data = DeleteVectorData(
            product_id=product_id,
            namespace=namespace
        )
        
        return SuccessResponse(
            code="200",
            message=f"Đã xóa vector của sản phẩm {product_id} khỏi namespace {namespace}",
            data=data
        )
    
    except Exception as e:
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi xóa vector: {str(e)}"
        )


@router.post("/batch-upsert", status_code=status.HTTP_200_OK)
async def batch_upsert_product_vectors(
    products: List[ProductVectorRequest],
    namespace: str
):
    """
    Lưu vector cho nhiều sản phẩm cùng lúc
    
    Backend gửi list các ProductVectorRequest, mỗi request chứa đầy đủ thông tin sản phẩm
    
    - **products**: List các ProductVectorRequest (mỗi item chứa đầy đủ thông tin sản phẩm)
    - **namespace**: Namespace trong Pinecone (có thể override namespace trong từng request)
    """
    try:
        results = []
        errors = []
        
        for product_request in products:
            try:
                # Sử dụng namespace từ request hoặc từ parameter
                target_namespace = product_request.namespace if product_request.namespace else namespace
                
                # Tạo text và embedding từ thông tin trong request
                text_for_embedding = create_text_for_embedding(
                    name=product_request.name,
                    description=product_request.description,
                    metadata=product_request.metadata
                )
                vector = embedding_service.create_embedding(text_for_embedding)
                
                # Chuẩn bị metadata
                metadata = prepare_metadata_for_pinecone(
                    product_id=product_request.product_id,
                    business_id=product_request.business_id,
                    name=product_request.name,
                    price=product_request.price,
                    status=product_request.status,
                    quantity_avail=product_request.quantity_avail,
                    metadata=product_request.metadata
                )
                
                # Lưu vector
                vector_id = str(product_request.product_id)
                pinecone_service.upsert_vector(
                    vector_id=vector_id,
                    vector=vector,
                    metadata=metadata,
                    namespace=target_namespace
                )
                
                results.append({
                    "product_id": product_request.product_id,
                    "namespace": target_namespace,
                    "status": "success"
                })
            
            except Exception as e:
                errors.append(f"Lỗi với sản phẩm {product_request.product_id}: {str(e)}")
        
        data = BatchUpsertData(
            success_count=len(results),
            error_count=len(errors),
            results=results,
            errors=errors
        )
        
        return SuccessResponse(
            code="200",
            message=f"Đã xử lý {len(results)}/{len(products)} sản phẩm",
            data=data
        )
    
    except Exception as e:
        return ErrorResponse(
            code="96",
            message=f"Lỗi khi batch upsert: {str(e)}"
        )

