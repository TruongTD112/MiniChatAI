"""
Service để gọi Gemini LLM API
"""
import os
import time
import logging
from typing import Optional, List, Dict
import google.generativeai as genai
from config import Config

logger = logging.getLogger(__name__)


class GeminiService:
    """Service để tương tác với Gemini LLM"""
    
    def __init__(self):
        """Khởi tạo Gemini client"""
        # Lấy API key từ config
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY không được tìm thấy trong environment variables")
        
        genai.configure(api_key=api_key)
        # Có thể đổi model: 'gemini-pro', 'gemini-1.5-pro', 'gemini-1.5-flash'
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Đã khởi tạo Gemini client với model: {model_name}")
    
    # def classify_intent(
    #     self,
    #     message: str,
    #     conversations: List[Dict],
    #     available_intents: List[Dict]
    # ) -> Dict:
    #     """
    #     Phân loại intent từ message của khách hàng
    #
    #     Args:
    #         message: Tin nhắn hiện tại
    #         conversations: Lịch sử trò chuyện
    #         available_intents: Danh sách intent có sẵn với type và description
    #
    #     Returns:
    #         Dict: {'intent': str, 'confidence': float, 'related_intents': List[str]}
    #     """
    #     try:
    #         # Xây dựng prompt để phân loại intent
    #         intent_descriptions = []
    #         for intent in available_intents:
    #             desc = f"- {intent['type']}: {intent['description'] or 'Không có mô tả'}"
    #             intent_descriptions.append(desc)
    #
    #         # Lấy lịch sử trò chuyện gần đây (tối đa 5 tin nhắn cuối)
    #         recent_conversations = conversations[-5:] if len(conversations) > 5 else conversations
    #         conversation_history = "\n".join([
    #             f"{msg.get('role', 'user')}: {msg.get('content', '')}"
    #             for msg in recent_conversations
    #         ])
    #
    #         prompt = f"""Bạn là một hệ thống phân loại intent cho chatbot bán hàng.
    #
    #                 Danh sách các intent có sẵn:
    #                 {chr(10).join(intent_descriptions)}
    #
    #                 Lịch sử trò chuyện gần đây:
    #                 {conversation_history if conversation_history else 'Chưa có lịch sử'}
    #
    #                 Tin nhắn hiện tại của khách hàng: "{message}"
    #
    #                 Hãy phân loại intent cho tin nhắn này. Trả về theo format JSON:
    #                 {{
    #                     "intent": "tên_intent",
    #                     "confidence": 0.0-1.0,
    #                     "related_intents": ["intent1", "intent2"] (nếu có)
    #                 }}
    #
    #                 Chỉ trả về JSON, không có text thêm."""
    #
    #         # Đo thời gian gọi LLM
    #         start_time = time.perf_counter()
    #         response = self.model.generate_content(prompt)
    #         elapsed_time = time.perf_counter() - start_time
    #
    #         # Parse response
    #         response_text = response.text.strip()
    #         # Loại bỏ markdown code block nếu có
    #         if response_text.startswith("```"):
    #             response_text = response_text.split("```")[1]
    #             if response_text.startswith("json"):
    #                 response_text = response_text[4:]
    #             response_text = response_text.strip()
    #
    #         import json
    #         result = json.loads(response_text)
    #
    #         logger.info(
    #             f"[LLM] Phân loại intent: {result.get('intent')} với confidence: {result.get('confidence')} "
    #             f"- Thời gian xử lý: {elapsed_time:.3f}s"
    #         )
    #         return result
    #
    #     except Exception as e:
    #         logger.error(f"Lỗi khi phân loại intent: {str(e)}")
    #         # Fallback về intent "others"
    #         return {
    #             'intent': 'others',
    #             'confidence': 0.5,
    #             'related_intents': []
    #         }

    def classify_intent(
            self,
            message: str,
            conversations: List[Dict],
            available_intents: List[Dict]
    ) -> Dict:
        try:
            intent_list = "\n".join(
                intent["type"] for intent in available_intents
            )

            recent_conversations = conversations[-2:] if conversations else []
            conversation_text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in recent_conversations
            )

            prompt = f"""
                    You are an intent classifier.
                
                    INTENTS:
                    {intent_list}
                
                    Conversation:
                    {conversation_text}
                
                    User message:
                    {message}
                
                    Return ONLY a valid JSON object in this format:
                    {{"intent": "...", "confidence": 0.0-1.0, "related_intents": []}}
                
                    No markdown. No explanation.
                    """

            start_time = time.perf_counter()

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0
                }
            )

            elapsed = time.perf_counter() - start_time
            text = response.text.strip()

            import json
            result = json.loads(text)

            logger.info(
                f"[LLM][Intent] {result.get('intent')} | "
                f"time={elapsed:.3f}s"
            )

            return {
                "intent": result.get("intent", "others"),
                "confidence": float(result.get("confidence", 0.5)),
                "related_intents": result.get("related_intents", [])
            }

        except Exception as e:
            logger.error(f"Lỗi classify_intent: {str(e)}")
            return {
                "intent": "others",
                "confidence": 0.5,
                "related_intents": []
            }

    # def generate_response(
    #     self,
    #     message: str,
    #     conversations: List[Dict],
    #     context: str,
    #     intent: str
    # ) -> str:
    #     """
    #     Tạo phản hồi từ Gemini dựa trên context
    #
    #     Args:
    #         message: Tin nhắn hiện tại
    #         conversations: Lịch sử trò chuyện
    #         context: Context đã được xây dựng
    #         intent: Intent đã được phân loại
    #
    #     Returns:
    #         str: Phản hồi từ bot
    #     """
    #     try:
    #         # Lấy lịch sử trò chuyện gần đây
    #         recent_conversations = conversations[-10:] if len(conversations) > 10 else conversations
    #         conversation_history = "\n".join([
    #             f"{msg.get('role', 'user')}: {msg.get('content', '')}"
    #             for msg in recent_conversations
    #         ])
    #
    #         prompt = f"""Bạn là một chatbot bán hàng thân thiện và chuyên nghiệp.
    #
    #                 Context và thông tin liên quan:
    #                 {context}
    #
    #                 Lịch sử trò chuyện:
    #                 {conversation_history if conversation_history else 'Đây là tin nhắn đầu tiên'}
    #
    #                 Tin nhắn của khách hàng: "{message}"
    #
    #                 Hãy trả lời một cách tự nhiên, thân thiện và hữu ích. Nếu không có thông tin trong context, hãy nói rõ và đề xuất cách khác để giúp khách hàng."""
    #
    #         # Đo thời gian gọi LLM
    #         start_time = time.perf_counter()
    #         response = self.model.generate_content(prompt)
    #         elapsed_time = time.perf_counter() - start_time
    #
    #         logger.info(
    #             f"[LLM] Tạo phản hồi cho intent '{intent}' - Thời gian xử lý: {elapsed_time:.3f}s"
    #         )
    #         return response.text.strip()
    #
    #     except Exception as e:
    #         logger.error(f"Lỗi khi tạo phản hồi: {str(e)}")
    #         return "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."

    def generate_response(
            self,
            message: str,
            conversations: List[Dict],
            context: str,
            intent: str
    ) -> str:
        try:
            # ===== 1. Chỉ lấy 2–3 turn gần nhất =====
            recent_conversations = conversations[-3:] if conversations else []
            conversation_history = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in recent_conversations
            )

            # ===== 2. Prompt NGẮN + ÉP TIẾNG VIỆT =====
            prompt = f"""
            SYSTEM ROLE:
            Bạn là nhân viên bán hàng của shop.
            NHIỆM VỤ: trả lời đúng theo intent bên dưới.
            CHỈ dùng tiếng Việt.
            KHÔNG giải thích lan man.
            KHÔNG nói những gì không có trong context.

            INTENT:
            {intent}

            THÔNG TIN SHOP / SẢN PHẨM:
            {context}

            HỘI THOẠI GẦN NHẤT:
            {conversation_history}

            KHÁCH HÀNG HỎI:
            {message}

            YÊU CẦU:
            - Trả lời đúng intent
            - Ngắn gọn
            - Lịch sự
            """

            start_time = time.perf_counter()

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0
                }
            )

            elapsed_time = time.perf_counter() - start_time
            reply = response.text.strip()

            logger.info(
                f"[LLM] Generate response | intent={intent} | time={elapsed_time:.3f}s"
            )

            # ===== 3. Guard: đảm bảo tiếng Việt =====
            vietnamese_chars = "ăâđêôơưáàảãạéèẻẽẹíìỉĩịóòỏõọúùủũụýỳỷỹỵ"
            if not any(c in reply.lower() for c in vietnamese_chars):
                reply = (
                    "Dạ bạn chờ shop một chút nhé, "
                    "mình sẽ hỗ trợ bạn ngay ạ 😊"
                )

            return reply

        except Exception as e:
            logger.error(f"Lỗi khi tạo phản hồi: {str(e)}")
            return (
                "Xin lỗi, hệ thống đang gặp sự cố. "
                "Bạn vui lòng liên hệ số 0985006914 để được hỗ trợ nhanh hơn nhé ạ."
            )


