"""
Giao diện Gradio cho Chatbot
"""
import gradio as gr
from typing import List, Tuple
from services.gemini_service import GeminiService
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Gemini service
try:
    gemini_service = GeminiService()
    logger.info("Đã khởi tạo Gemini service thành công")
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo Gemini service: {str(e)}")
    gemini_service = None


def chat_response(
    message: str,
    history: List[Tuple[str, str]],
    instruction: str,
    product_context: str
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Xử lý tin nhắn và trả về phản hồi từ chatbot
    
    Args:
        message: Tin nhắn hiện tại của người dùng
        history: Lịch sử chat (list of tuples: [(user_msg, bot_msg), ...])
        instruction: Instruction/prompt cho chatbot
        product_context: Context về sản phẩm
        
    Returns:
        Tuple: (updated_history, empty_string)
    """
    if not message or not message.strip():
        return history, ""
    
    if gemini_service is None:
        error_msg = "Lỗi: Gemini service chưa được khởi tạo. Vui lòng kiểm tra GEMINI_API_KEY."
        history.append((message, error_msg))
        return history, ""
    
    try:
        # Chuyển đổi history từ Gradio format sang format cho GeminiService
        # Gradio history: [(user_msg, bot_msg), ...]
        # GeminiService format: [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]
        conversations = []
        for user_msg, bot_msg in history:
            conversations.append({'role': 'user', 'content': user_msg})
            conversations.append({'role': 'assistant', 'content': bot_msg})
        
        # Thêm tin nhắn hiện tại vào conversations
        conversations.append({'role': 'user', 'content': message})
        
        # Gọi Gemini service để tạo phản hồi
        response = gemini_service.generate_chat_response(
            message=message,
            conversations=conversations,
            instruction=instruction,
            product_context=product_context
        )
        
        # Thêm vào history
        history.append((message, response))
        
        return history, ""
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý chat: {str(e)}")
        error_msg = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"
        history.append((message, error_msg))
        return history, ""


def clear_chat():
    """Xóa lịch sử chat"""
    return [], ""


# Tạo giao diện Gradio
with gr.Blocks(title="Chatbot với Gemini", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤖 Chatbot với Gemini LLM
        
        Giao diện chat sử dụng Gemini để trả lời câu hỏi. 
        Bot sẽ sử dụng:
        - **Instruction** (hướng dẫn) do bạn cung cấp
        - **Context sản phẩm** để hiểu về sản phẩm
        - **20 tin nhắn gần nhất** làm context cho cuộc trò chuyện
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Cấu hình Chatbot")
            
            with gr.Tabs():
                with gr.Tab("📝 Instruction"):
                    gr.Markdown(
                        """
                        **Nhập hướng dẫn cho chatbot ở đây:**
                        
                        Đây là phần bạn định nghĩa vai trò và cách chatbot sẽ trả lời.
                        """
                    )
                    instruction_input = gr.Textbox(
                        label="Instruction (Hướng dẫn cho chatbot)",
                        placeholder="Ví dụ:\nBạn là một nhân viên bán hàng thân thiện và chuyên nghiệp.\nHãy trả lời các câu hỏi về sản phẩm một cách chi tiết và nhiệt tình.\nLuôn sử dụng ngôn ngữ lịch sự và thân thiện.",
                        lines=8,
                        value="Bạn là một chatbot hỗ trợ khách hàng. Hãy trả lời các câu hỏi một cách thân thiện và hữu ích."
                    )
                    gr.Markdown(
                        """
                        **💡 Gợi ý:** 
                        - Mô tả vai trò của chatbot
                        - Quy định phong cách trả lời
                        - Hướng dẫn cách xử lý các tình huống
                        """
                    )
                
                with gr.Tab("📦 Context Sản phẩm"):
                    gr.Markdown(
                        """
                        **Nhập thông tin về sản phẩm ở đây:**
                        
                        Bot sẽ sử dụng thông tin này để trả lời các câu hỏi về sản phẩm.
                        """
                    )
                    product_context_input = gr.Textbox(
                        label="Context Sản phẩm",
                        placeholder="Ví dụ:\nTên sản phẩm: Áo thun nam\nGiá: 250.000 VNĐ\nMô tả: Áo thun chất liệu cotton 100%, thoáng mát, phù hợp mùa hè\nMàu sắc: Đen, Trắng, Xanh\nKích thước: S, M, L, XL\nTình trạng: Còn hàng",
                        lines=12,
                        value=""
                    )
                    gr.Markdown(
                        """
                        **💡 Gợi ý:**
                        - Tên sản phẩm
                        - Giá cả
                        - Mô tả chi tiết
                        - Thông số kỹ thuật
                        - Tình trạng hàng hóa
                        - Chính sách bán hàng
                        """
                    )
            
            gr.Markdown("---")
            clear_btn = gr.Button("🗑️ Xóa lịch sử chat", variant="secondary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            
            chatbot = gr.Chatbot(
                label="Cuộc trò chuyện",
                height=500,
                show_copy_button=True
            )
            
            msg_input = gr.Textbox(
                label="Nhập tin nhắn",
                placeholder="Nhập câu hỏi của bạn ở đây...",
                lines=2
            )
            
            with gr.Row():
                send_btn = gr.Button("📤 Gửi", variant="primary", scale=1)
                clear_input_btn = gr.Button("🗑️ Xóa", variant="secondary", scale=1)
    
    # Xử lý sự kiện
    msg_input.submit(
        chat_response,
        inputs=[msg_input, chatbot, instruction_input, product_context_input],
        outputs=[chatbot, msg_input]
    )
    
    send_btn.click(
        chat_response,
        inputs=[msg_input, chatbot, instruction_input, product_context_input],
        outputs=[chatbot, msg_input]
    )
    
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, msg_input]
    )
    
    clear_input_btn.click(
        lambda: "",
        outputs=[msg_input]
    )
    
    gr.Markdown(
        """
        ---
        ### 📌 Hướng dẫn sử dụng:
        
        1. **Cấu hình (Bên trái):**
           - Tab **"📝 Instruction"**: Nhập hướng dẫn cho chatbot (vai trò, phong cách trả lời)
           - Tab **"📦 Context Sản phẩm"**: Nhập thông tin về sản phẩm (tên, giá, mô tả, thông số...)
        
        2. **Chat (Bên phải):**
           - Nhập câu hỏi vào ô "Nhập tin nhắn"
           - Nhấn **Enter** hoặc nút **"📤 Gửi"** để gửi
           - Bot sẽ tự động sử dụng: Instruction + Context sản phẩm + 20 tin nhắn gần nhất
        
        3. **Lưu ý:**
           - Bạn có thể thay đổi Instruction và Context sản phẩm bất cứ lúc nào
           - Bot sẽ tự động lấy 20 tin nhắn gần nhất làm context
           - Nhấn "🗑️ Xóa lịch sử chat" để bắt đầu cuộc trò chuyện mới
        """
    )


if __name__ == "__main__":
    # Chạy Gradio app
    demo.launch(
        server_name="0.0.0.0",  # Cho phép truy cập từ mọi địa chỉ IP
        server_port=7860,       # Port mặc định của Gradio
        share=False,            # Set True nếu muốn tạo public link
        show_error=True
    )

