# Product Vector API

API để quản lý vector database cho sản phẩm sử dụng Pinecone.

## Tính năng

- Lưu/update vector embeddings cho sản phẩm vào Pinecone với namespace
- Tìm kiếm sản phẩm bằng vector similarity search
- Xóa vector của sản phẩm
- Batch upsert nhiều sản phẩm cùng lúc

## Cài đặt

1. Cài đặt dependencies:

```bash
pip install -r requirements.txt
```

2. Cấu hình environment variables:

Sao chép file `.env.example` thành `.env` và điền các thông tin:

```bash
cp .env.example .env
```

Các biến môi trường cần thiết:
- `DATABASE_URL`: Connection string đến MySQL database
- `PINECONE_API_KEY`: API key của Pinecone
- `PINECONE_INDEX_NAME`: Tên index trong Pinecone (mặc định: products)
- `OPENAI_API_KEY`: API key của OpenAI để tạo embeddings

3. Chạy ứng dụng:

```bash
uvicorn main:app --reload
```

API sẽ chạy tại `http://localhost:8000`

## API Endpoints

### 1. Lưu/Update vector cho sản phẩm

**POST** `/api/products/vector/upsert`

Backend gửi toàn bộ thông tin sản phẩm trong request body:

Request body:
```json
{
  "product_id": 1,
  "namespace": "business_123",
  "business_id": 100,
  "name": "Áo thun nam",
  "description": "Áo thun nam chất liệu cotton 100%",
  "price": 199000,
  "main_image_url": "https://example.com/image.jpg",
  "detail_image_url": "https://example.com/img1.jpg,https://example.com/img2.jpg",
  "quantity_avail": 50,
  "status": "1",
  "metadata": {
    "category": "quần áo",
    "brand": "Nike",
    "size": ["S", "M", "L"]
  }
}
```

Response:
```json
{
  "code": "200",
  "message": "Đã lưu vector thành công",
  "data": {
    "product_id": 1,
    "namespace": "business_123",
    "vector_id": "1"
  }
}
```

Lỗi hệ thống (code 96):
```json
{
  "code": "96",
  "message": "Lỗi khi lưu vector: [chi tiết lỗi]",
  "data": null
}
```

### 2. Tìm kiếm sản phẩm bằng vector

**POST** `/api/products/vector/search`

Request body:
```json
{
  "query_text": "áo thun nam",
  "namespace": "business_123",
  "top_k": 10,
  "filter": {
    "status": "1"
  }
}
```

Response (thông tin sản phẩm được lấy từ metadata trong Pinecone):
```json
{
  "code": "200",
  "message": "Tìm kiếm thành công",
  "data": {
    "results": [
      {
        "product_id": 1,
        "score": 0.95,
        "product": {
          "id": 1,
          "business_id": 100,
          "name": "Áo thun nam",
          "description": null,
          "price": 199000,
          "main_image_url": null,
          "detail_image_url": null,
          "quantity_avail": 50,
          "status": "1",
          "metadata": {
            "category": "quần áo",
            "brand": "Nike"
          },
          "created_at": "2024-01-01T00:00:00",
          "updated_at": "2024-01-01T00:00:00"
        }
      }
    ],
    "total": 1
  }
}
```

### 3. Xóa vector của sản phẩm

**DELETE** `/api/products/vector/{product_id}?namespace=business_123`

Response:
```json
{
  "code": "200",
  "message": "Đã xóa vector của sản phẩm 1 khỏi namespace business_123",
  "data": {
    "product_id": 1,
    "namespace": "business_123"
  }
}
```

### 4. Batch upsert nhiều sản phẩm

**POST** `/api/products/vector/batch-upsert?namespace=business_123`

Request body (list các ProductVectorRequest):
```json
[
  {
    "product_id": 1,
    "namespace": "business_123",
    "business_id": 100,
    "name": "Áo thun nam",
    "description": "Mô tả sản phẩm 1",
    "price": 199000,
    "quantity_avail": 50,
    "status": "1",
    "metadata": {}
  },
  {
    "product_id": 2,
    "namespace": "business_123",
    "business_id": 100,
    "name": "Quần jean",
    "description": "Mô tả sản phẩm 2",
    "price": 399000,
    "quantity_avail": 30,
    "status": "1",
    "metadata": {}
  }
]
```

Response:
```json
{
  "code": "200",
  "message": "Đã xử lý 2/2 sản phẩm",
  "data": {
    "success_count": 2,
    "error_count": 0,
    "results": [
      {
        "product_id": 1,
        "namespace": "business_123",
        "status": "success"
      },
      {
        "product_id": 2,
        "namespace": "business_123",
        "status": "success"
      }
    ],
    "errors": []
  }
}
```

## Cấu trúc dự án

```
.
├── main.py                 # Entry point của ứng dụng
├── config.py               # Configuration và environment variables
├── database.py             # Database configuration (không cần thiết nếu không dùng DB)
├── models/
│   └── product.py         # Product model (reference, không dùng trong API)
├── schemas/
│   └── product.py         # Pydantic schemas cho request/response
├── services/
│   ├── pinecone_service.py    # Service tương tác với Pinecone
│   └── embedding_service.py   # Service tạo embeddings
├── utils/
│   └── product_helper.py      # Helper functions cho product operations
└── api/
    └── routes/
        └── product_vector.py  # API routes
```

## Format Response

Tất cả API responses đều tuân theo format chuẩn:

```json
{
  "code": "200",        // "200" cho thành công, "96" cho lỗi hệ thống
  "message": "Thông báo",
  "data": { ... }       // Object chứa dữ liệu (null nếu lỗi)
}
```

### Mã code:
- **"200"**: Thành công
- **"96"**: Lỗi hệ thống

## Lưu ý

- **Backend gửi toàn bộ thông tin sản phẩm**: API không query database, nhận trực tiếp thông tin sản phẩm từ request body
- **Vector embeddings**: Được tạo từ kết hợp `name`, `description` và `metadata` của sản phẩm
- **Metadata trong Pinecone**: Thông tin sản phẩm được lưu trong metadata của vector, có thể dùng để filter và trả về kết quả
- **Namespace**: Mỗi namespace trong Pinecone có thể chứa vectors của nhiều sản phẩm, cho phép phân tách theo business hoặc category
- **Dimension**: Mặc định là 1536 (cho OpenAI text-embedding-3-small), đảm bảo dimension trong Pinecone index khớp với dimension của embedding model
- **Search results**: Kết quả search được lấy từ metadata trong Pinecone, không cần query database
- **Exception handling**: Tất cả lỗi validation và lỗi hệ thống đều được xử lý và trả về format chuẩn với code "96"

