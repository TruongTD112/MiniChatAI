import os
import requests
import base64
from google.cloud import aiplatform
from google.protobuf import struct_pb2
from pinecone import Pinecone

# --- CẤU HÌNH (Giống file Upsert) ---
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertexAI.json"
PROJECT_ID = "gen-lang-client-0324909658"
LOCATION = "us-central1"
PINECONE_API_KEY = "pcsk_4Yj6Cr_UBfX99wUDDMZ3RynJiNjwojXUeSsvi2vcMUEEcXeFUhfhEN3TpdRjmiqZwLz7kH"
INDEX_NAME = "fashion-shop-02"  # Đảm bảo tên Index này có dimension 1408
NAMESPACE = "image-data"

# Khởi tạo
aiplatform.init(project=PROJECT_ID, location=LOCATION)
client = aiplatform.gapic.PredictionServiceClient(
    client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


def get_embedding(text=None, image_url=None):
    """Tạo vector 1408 chiều cho Text hoặc Image"""
    endpoint = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/multimodalembedding@001"
    instance = struct_pb2.Struct()

    if image_url:
        response = requests.get(image_url)
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        instance.update({"image": {"bytesBase64Encoded": image_base64}})
    elif text:
        instance.update({"text": text})

    parameters = struct_pb2.Struct()
    parameters.update({"dimension": 1408})  # Phải là 1408 để khớp với Index mới

    res = client.predict(endpoint=endpoint, instances=[instance], parameters=parameters)

    if image_url:
        return list(res.predictions[0]['imageEmbedding'])
    else:
        return list(res.predictions[0]['textEmbedding'])


def search_now(query_text=None, query_image_url=None, top_k=3):
    """Tìm kiếm trong Pinecone"""
    # 1. Chuyển câu hỏi (chữ hoặc ảnh) thành vector
    print(f"--- Đang tìm kiếm cho: {query_text if query_text else 'Ảnh đầu vào'} ---")
    query_vector = get_embedding(text=query_text, image_url=query_image_url)

    # 2. Truy vấn Pinecone
    results = index.query(
        namespace=NAMESPACE,
        vector=query_vector,
        top_k=top_k,
        include_metadata=True  # Để lấy lại URL ảnh và thông tin sản phẩm
    )

    return results['matches']


# --- CHẠY THỬ NGHIỆM ---
if __name__ == "__main__":
    # Cách 1: Tìm bằng văn bản (Text-to-Image)
    # print("\n--- TEST: TÌM BẰNG VĂN BẢN ---")
    # matches_text = search_now(query_text="áo sơ mi nam màu trắng")
    # for m in matches_text:
    #     print(f"ID: {m['id']} | Độ khớp: {m['score']:.4f}")
    #     print(f"Link ảnh: {m['metadata']['url']}\n")

    # Cách 2: Tìm bằng ảnh (Image-to-Image)
    print("\n--- TEST: TÌM BẰNG ẢNH TƯƠNG TỰ ---")
    test_img = "https://res.cloudinary.com/dlqkmnlgm/image/upload/v1768662378/2a2b6c81-8459-4029-8e1c-af540ce64f54_ljuk6t.jpg"
    matches_img = search_now(query_image_url=test_img)
    for m in matches_img:
        print(f"ID: {m['id']} | Độ khớp: {m['score']:.4f}")