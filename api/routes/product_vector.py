"""
API routes cho Product Vector operations
"""
from fastapi import APIRouter, HTTPException, status, Query
from typing import List
import logging

logger = logging.getLogger(__name__)

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
from utils.product_helper import (
    create_text_for_embedding, 
    prepare_metadata_for_pinecone,
    extract_image_urls
)

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
        
        # Tạo text embedding vector
        text_vector = embedding_service.create_embedding(text_for_embedding)
        
        # Chuẩn bị metadata để lưu vào Pinecone
        metadata = prepare_metadata_for_pinecone(
            product_id=request.product_id,
            business_id=request.business_id,
            name=request.name,
            price=request.price,
            status=request.status,
            quantity_avail=request.quantity_avail,
            description=request.description,
            metadata=request.metadata
        )
        
        # Thêm image URLs vào metadata
        if request.main_image_url:
            metadata['main_image_url'] = request.main_image_url
        if request.detail_image_url:
            metadata['detail_image_url'] = request.detail_image_url
        
        # Lưu text vector vào Pinecone
        text_vector_id = f"{request.product_id}_text"
        vectors_to_upsert = [{
            'id': text_vector_id,
            'values': text_vector,
            'metadata': {**metadata, 'vector_type': 'text'}
        }]
        
        # Tạo image embeddings cho tất cả các ảnh (main + detail)
        all_image_urls = extract_image_urls(
            request.main_image_url,
            request.detail_image_url
        )
        
        if all_image_urls:
            for index, image_url in enumerate(all_image_urls):
                try:
                    image_vector = embedding_service.create_image_embedding(image_url)
                    if image_vector:
                        # Đặt tên vector: image_0 cho ảnh đầu tiên, image_1 cho ảnh thứ 2, etc.
                        # Nếu là main_image_url (index 0) và có main_image_url, có thể đặt tên đặc biệt
                        if index == 0 and request.main_image_url and image_url == request.main_image_url.strip():
                            image_vector_id = f"{request.product_id}_image_main"
                        else:
                            image_vector_id = f"{request.product_id}_image_{index}"
                        
                        vectors_to_upsert.append({
                            'id': image_vector_id,
                            'values': image_vector,
                            'metadata': {**metadata, 'vector_type': 'image', 'image_index': index}
                        })
                except Exception as img_error:
                    # Log lỗi nhưng không fail toàn bộ request nếu image embedding thất bại
                    logger.warning(f"Không thể tạo image embedding cho ảnh {index} của product {request.product_id}: {str(img_error)}")
        
        # Upsert tất cả vectors cùng lúc
        pinecone_service.upsert_vectors_batch(
            vectors=vectors_to_upsert,
            namespace=request.namespace
        )
        
        data = ProductVectorData(
            product_id=request.product_id,
            namespace=request.namespace,
            vector_id=text_vector_id  # Trả về text vector ID làm chính
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
        # Validate request
        if not request.query_text and not request.query_image_url:
            return ErrorResponse(
                code="400",
                message="Phải có ít nhất query_text hoặc query_image_url"
            )
        
        all_results = []
        search_type = request.search_type.lower()
        
        # Chuẩn bị filter chung từ request.filter
        base_filter = {**request.filter} if request.filter else {}
        
        # Search theo text nếu có query_text và search_type cho phép
        if request.query_text and search_type in ['text', 'both']:
            try:
                query_vector = embedding_service.create_embedding(request.query_text)
                
                # Thêm filter để chỉ search text vectors
                text_filter = {**base_filter}
                text_filter['vector_type'] = 'text'
                
                text_results = pinecone_service.search_vectors(
                    query_vector=query_vector,
                    namespace=request.namespace,
                    top_k=request.top_k,
                    filter=text_filter
                )
                all_results.extend(text_results)
            except Exception as e:
                logger.warning(f"Lỗi khi search text: {str(e)}")
        
        # Search theo image nếu có query_image_url và search_type cho phép
        if request.query_image_url and search_type in ['image', 'both']:
            try:
                image_vector = embedding_service.create_image_embedding(request.query_image_url)
                if image_vector:
                    # Thêm filter để chỉ search image vectors
                    image_filter = {**base_filter}
                    image_filter['vector_type'] = 'image'
                    
                    image_results = pinecone_service.search_vectors(
                        query_vector=image_vector,
                        namespace=request.namespace,
                        top_k=request.top_k,
                        filter=image_filter
                    )
                    all_results.extend(image_results)
            except Exception as e:
                logger.warning(f"Lỗi khi search image: {str(e)}")
        
        # Merge và deduplicate kết quả theo product_id, lấy score cao nhất
        product_scores = {}
        product_metadata = {}
        
        for result in all_results:
            # Extract product_id từ vector_id
            # Format có thể là: {product_id}_text, {product_id}_image_main, {product_id}_image_0, etc.
            vector_id = result['id']
            logger.debug(f"Đang xử lý vector_id: {vector_id}")
            
            product_id = None
            
            # Thử các cách extract product_id
            if '_' in vector_id:
                # Format: {product_id}_text hoặc {product_id}_image_* hoặc {product_id}_image_main
                parts = vector_id.split('_')
                if len(parts) >= 2:
                    # Lấy phần đầu tiên (product_id)
                    product_id_str = parts[0]
                    logger.debug(f"Extract product_id_str từ vector_id '{vector_id}': '{product_id_str}'")
                    
                    try:
                        product_id = int(product_id_str)
                        logger.debug(f"Parse thành công product_id: {product_id}")
                    except ValueError as e:
                        logger.warning(
                            f"Không thể parse product_id từ vector_id '{vector_id}': "
                            f"product_id_str='{product_id_str}', error={str(e)}"
                        )
                        continue
                else:
                    logger.warning(f"Vector_id '{vector_id}' có format không hợp lệ (ít hơn 2 phần khi split)")
                    continue
            else:
                # Fallback: thử parse trực tiếp (nếu không có dấu _)
                logger.debug(f"Vector_id '{vector_id}' không có dấu _, thử parse trực tiếp")
                try:
                    product_id = int(vector_id)
                    logger.debug(f"Parse trực tiếp thành công product_id: {product_id}")
                except ValueError as e:
                    logger.warning(
                        f"Không thể parse product_id trực tiếp từ vector_id '{vector_id}': {str(e)}"
                    )
                    continue
            
            if product_id is None:
                logger.warning(f"Không thể extract product_id từ vector_id: {vector_id}, bỏ qua")
                continue
            
            # Lưu score cao nhất cho mỗi product
            if product_id not in product_scores or result['score'] > product_scores[product_id]:
                product_scores[product_id] = result['score']
                product_metadata[product_id] = result.get('metadata', {})
        
        # Sắp xếp theo score và lấy top_k
        sorted_products = sorted(
            product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:request.top_k]
        
        # Tạo kết quả
        results = []
        from schemas.product import ProductResponse
        from datetime import datetime
        
        for product_id, score in sorted_products:
            metadata = product_metadata[product_id]
            
            product_response = ProductResponse(
                id=product_id,
                business_id=metadata.get('business_id', 0),
                name=metadata.get('name', ''),
                description=None,
                price=metadata.get('price', 0),
                main_image_url=metadata.get('main_image_url'),
                detail_image_url=metadata.get('detail_image_url'),
                quantity_avail=metadata.get('quantity_avail', 0),
                status=metadata.get('status', '1'),
                metadata={k: v for k, v in metadata.items() 
                         if k not in ['product_id', 'business_id', 'name', 'price', 'status', 
                                     'quantity_avail', 'vector_type', 'main_image_url', 'detail_image_url']},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            results.append(ProductSearchResult(
                product_id=product_id,
                score=score,
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
    namespace: str = Query(..., description="Namespace trong Pinecone")
):
    """
    Xóa vector của sản phẩm khỏi Pinecone
    
    - **product_id**: ID của sản phẩm cần xóa vector (trong URL path)
    - **namespace**: Namespace trong Pinecone (query parameter, bắt buộc)
    """
    try:
        # Xóa text vector và tất cả image vectors từ Pinecone
        # Vì không biết chính xác số lượng image vectors, sẽ xóa text vector và 
        # thử xóa các image vectors từ 0 đến 50 (Pinecone sẽ bỏ qua các vector không tồn tại)
        text_vector_id = f"{product_id}_text"
        vector_ids_to_delete = [text_vector_id]
        
        # Thêm image_main và các image vectors có thể có (0-50 để cover hầu hết trường hợp)
        vector_ids_to_delete.append(f"{product_id}_image_main")
        for i in range(50):  # Hỗ trợ tối đa 50 ảnh
            vector_ids_to_delete.append(f"{product_id}_image_{i}")
        
        # Pinecone sẽ tự động bỏ qua các vector không tồn tại, không gây lỗi
        pinecone_service.delete_vectors(
            vector_ids=vector_ids_to_delete,
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
        all_vectors_to_upsert = []
        
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
                text_vector = embedding_service.create_embedding(text_for_embedding)
                
                # Chuẩn bị metadata
                metadata = prepare_metadata_for_pinecone(
                    product_id=product_request.product_id,
                    business_id=product_request.business_id,
                    name=product_request.name,
                    price=product_request.price,
                    status=product_request.status,
                    quantity_avail=product_request.quantity_avail,
                    description=product_request.description,
                    metadata=product_request.metadata
                )
                
                # Thêm image URLs vào metadata
                if product_request.main_image_url:
                    metadata['main_image_url'] = product_request.main_image_url
                if product_request.detail_image_url:
                    metadata['detail_image_url'] = product_request.detail_image_url
                
                # Thêm text vector
                text_vector_id = f"{product_request.product_id}_text"
                all_vectors_to_upsert.append({
                    'id': text_vector_id,
                    'values': text_vector,
                    'metadata': {**metadata, 'vector_type': 'text'},
                    'namespace': target_namespace
                })
                
                # Tạo image embeddings cho tất cả các ảnh (main + detail)
                all_image_urls = extract_image_urls(
                    product_request.main_image_url,
                    product_request.detail_image_url
                )
                
                if all_image_urls:
                    for index, image_url in enumerate(all_image_urls):
                        try:
                            image_vector = embedding_service.create_image_embedding(image_url)
                            if image_vector:
                                # Đặt tên vector: image_main cho ảnh đầu tiên nếu là main, hoặc image_0, image_1, etc.
                                if index == 0 and product_request.main_image_url and image_url == product_request.main_image_url.strip():
                                    image_vector_id = f"{product_request.product_id}_image_main"
                                else:
                                    image_vector_id = f"{product_request.product_id}_image_{index}"
                                
                                all_vectors_to_upsert.append({
                                    'id': image_vector_id,
                                    'values': image_vector,
                                    'metadata': {**metadata, 'vector_type': 'image', 'image_index': index},
                                    'namespace': target_namespace
                                })
                        except Exception as img_error:
                            logger.warning(f"Không thể tạo image embedding cho ảnh {index} của product {product_request.product_id}: {str(img_error)}")
                
                results.append({
                    "product_id": product_request.product_id,
                    "namespace": target_namespace,
                    "status": "success"
                })
            
            except Exception as e:
                errors.append(f"Lỗi với sản phẩm {product_request.product_id}: {str(e)}")
        
        # Upsert tất cả vectors theo namespace
        if all_vectors_to_upsert:
            # Nhóm vectors theo namespace
            vectors_by_namespace = {}
            for vec in all_vectors_to_upsert:
                ns = vec.pop('namespace')
                if ns not in vectors_by_namespace:
                    vectors_by_namespace[ns] = []
                vectors_by_namespace[ns].append(vec)
            
            # Upsert từng namespace
            for ns, vectors in vectors_by_namespace.items():
                try:
                    pinecone_service.upsert_vectors_batch(
                        vectors=vectors,
                        namespace=ns
                    )
                except Exception as e:
                    logger.error(f"Lỗi khi upsert vectors vào namespace {ns}: {str(e)}")
                    errors.append(f"Lỗi khi upsert vào namespace {ns}: {str(e)}")
        
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

