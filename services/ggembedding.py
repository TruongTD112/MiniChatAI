import os
import base64
import requests
import google.cloud.aiplatform as aiplatform
from google.protobuf import struct_pb2
from pinecone import Pinecone

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
# Trỏ đến file chìa khóa JSON bạn vừa tải về
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexAI.json"

PROJECT_ID = "gen-lang-client-0324909658"
LOCATION = "us-central1"
PINECONE_API_KEY = "pcsk_4Yj6Cr_UBfX99wUDDMZ3RynJiNjwojXUeSsvi2vcMUEEcXeFUhfhEN3TpdRjmiqZwLz7kH"
INDEX_NAME = "fashion-shop-02"
NAMESPACE = "image-data"

# Khởi tạo Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)
client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)

# Khởi tạo Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ==========================================
# 2. CÁC HÀM XỬ LÝ CHÍNH
# ==========================================

def get_embedding(text=None, image_url=None):
    """
    Sử dụng Vertex AI để tạo vector 768 chiều.
    Có thể truyền vào 'text' HOẶC 'image_url'.
    """
    endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/multimodalembedding@001"

    instance = struct_pb2.Struct()

    # Xử lý nếu đầu vào là Ảnh
    if image_url:
        response = requests.get(image_url)
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        instance.update({"image": {"bytesBase64Encoded": image_base64}})

    # Xử lý nếu đầu vào là Văn bản
    elif text:
        instance.update({"text": text})

    # Ép kiểu về 768 chiều để khớp với Pinecone Index của bạn
    parameters = struct_pb2.Struct()
    parameters.update({"dimension": 1408})

    # Gọi API Google Cloud
    res = client.predict(endpoint=endpoint, instances=[instance], parameters=parameters)

    # Trả về vector tương ứng
    if image_url:
        return list(res.predictions[0]['imageEmbedding'])
    else:
        return list(res.predictions[0]['textEmbedding'])


def upsert_image(img_id, img_url, metadata=None):
    """Lưu một ảnh vào Pinecone"""
    print(f"--- Đang tạo vector cho ảnh: {img_id} ---")
    vector = get_embedding(image_url=img_url)

    if metadata is None:
        metadata = {}
    metadata["url"] = img_url

    index.upsert(
        vectors=[{"id": img_id, "values": vector, "metadata": metadata}],
        namespace=NAMESPACE
    )
    print(f"✅ Đã lưu thành công {img_id}")


def search_images(query_text, top_k=3):
    """Tìm kiếm ảnh bằng từ khóa văn bản"""
    print(f"--- Đang tìm kiếm ảnh với từ khóa: '{query_text}' ---")
    query_vector = get_embedding(text=query_text)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True
    )
    return results['matches']


# ==========================================
# 3. CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    try:
        # TEST 1: Lưu thử một ảnh
        test_url = "https://res.cloudinary.com/dlqkmnlgm/image/upload/v1760975966/foodshare/bkm0glueylc2lkanv0yw.jpg"  # Thay bằng link ảnh thật của bạn
        upsert_image("ao_so_mi_001", test_url, {"name": "Áo sơ mi nam công sở", "price": "350k"})

        # TEST 2: Tìm kiếm thử
        search_query = "áo sơ mi"
        matches = search_images(search_query)

        print(f"\nKết quả tìm kiếm cho '{search_query}':")
        for m in matches:
            print(f"- ID: {m['id']} | Khớp: {m['score']:.3f} | Link: {m['metadata']['url']}")

    except Exception as e:
        print(f"❌ Lỗi rồi: {e}")