"""
Service để gọi Gemini LLM API
"""
import os
import time
import logging
import hashlib
from typing import Optional, List, Dict, Tuple
import google.generativeai as genai
from config import Config

logger = logging.getLogger(__name__)


def _log_usage(response, label: str = "LLM") -> None:
    """Log số token input/output/total cho mỗi lần gọi (hỗ trợ cả SDK cũ và mới)."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_token_count", None)
    cached_content_prompt_tokens = getattr(usage, "cached_content_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_token_count", None)
    total = getattr(usage, "total_token_count", None)
    if prompt_tokens is not None or output_tokens is not None or total is not None:
        logger.info(
            "[%s] Token usage | input=%s | cached=%s |output=%s | total=%s",
            label,
            prompt_tokens if prompt_tokens is not None else "?",
            cached_content_prompt_tokens if cached_content_prompt_tokens is not None else "?",
            output_tokens if output_tokens is not None else "?",
            total if total is not None else "?",
        )


# SDK mới (google-genai) dùng cho prompt caching
try:
    from google import genai as genai_new
    from google.genai import types as genai_types
    _GENAI_NEW_AVAILABLE = True
except ImportError:
    _GENAI_NEW_AVAILABLE = False


class GeminiService:
    """Service để tương tác với Gemini LLM"""

    CACHE_TTL_SECONDS = 3600  # TTL cache mặc định (1 giờ)

    def __init__(self):
        """Khởi tạo Gemini client"""
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY không được tìm thấy trong environment variables")

        genai.configure(api_key=api_key)
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        self.model = genai.GenerativeModel(model_name)
        self._model_name = model_name
        self._model_name_for_cache = f"models/{model_name}" if not model_name.startswith("models/") else model_name

        self._genai_client: Optional[object] = None
        self._chat_cache: Dict[str, Tuple[str, float]] = {}  # cache_key -> (cache_name, expire_time)

        if _GENAI_NEW_AVAILABLE:
            try:
                self._genai_client = genai_new.Client(api_key=api_key)
                logger.info("Đã bật prompt caching (instruction + product context) với google-genai")
            except Exception as e:
                logger.warning("Không khởi tạo được client caching: %s. Chat sẽ không dùng cache.", e)
        else:
            logger.info("Chưa cài google-genai; chat không dùng prompt caching. Cài: pip install google-genai")

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

            logger.info("[LLM][Intent] Input gửi sang Gemini:\n%s", prompt)

            start_time = time.perf_counter()

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0
                }
            )
            _log_usage(response, "LLM][Intent")

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

            logger.info("[LLM][Generate] Input gửi sang Gemini:\n%s", prompt)

            start_time = time.perf_counter()

            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0
                }
            )
            _log_usage(response, "LLM][Generate")

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

    def _get_or_create_chat_cache(self, instruction: str, product_context: str) -> Optional[str]:
        """
        Lấy hoặc tạo cache cho instruction + product_context (prompt caching).
        Trả về cache.name để dùng với GenerateContentConfig(cached_content=...).
        """
        if not self._genai_client or not _GENAI_NEW_AVAILABLE:
            return None
        cache_key = hashlib.sha256((instruction or "").encode() + (product_context or "").encode()).hexdigest()
        now = time.time()
        if cache_key in self._chat_cache:
            cache_name, expire_time = self._chat_cache[cache_key]
            if now < expire_time:
                logger.info("[LLM][Cache] Đang dùng cache có sẵn (instruction + product context), key=%s...", cache_key[:16])
                return cache_name
            # Cache hết hạn, xóa để tạo mới
            del self._chat_cache[cache_key]
        # Tạo cache mới
        cached_text = ""
        if instruction:
            cached_text += f"INSTRUCTION (Hướng dẫn cho chatbot):\n{instruction}\n\n"
        if product_context:
            cached_text += f"CONTEXT SẢN PHẨM:\n{product_context}\n\n"
        if not cached_text.strip():
            return None
        try:
            cache = self._genai_client.caches.create(
                model=self._model_name_for_cache,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=cached_text.strip(),
                    ttl=f"{self.CACHE_TTL_SECONDS}s",
                ),
            )
            expire_time = now + self.CACHE_TTL_SECONDS
            self._chat_cache[cache_key] = (cache.name, expire_time)
            logger.info("[LLM][Cache] Đã tạo cache mới cho instruction + product context (TTL=%ss), key=%s...", self.CACHE_TTL_SECONDS, cache_key[:16])
            return cache.name
        except Exception as e:
            logger.warning("Không tạo được cache: %s. Gửi full prompt.", e)
            return None

    def generate_chat_response(
        self,
        message: str,
        conversations: List[Dict],
        instruction: str = "",
        product_context: str = ""
    ) -> str:
        """
        Tạo phản hồi chat với instruction tùy chỉnh, product context và lịch sử chat.
        Dùng prompt caching (instruction + product_context) khi có google-genai.
        """
        try:
            recent_conversations = conversations[-6:] if len(conversations) > 6 else conversations
            conversation_history = ""
            if recent_conversations:
                conversation_history = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in recent_conversations
                ])
            else:
                conversation_history = "Đây là tin nhắn đầu tiên trong cuộc trò chuyện."

            # Thử dùng prompt caching (instruction + product_context đã cache)
            cache_name = self._get_or_create_chat_cache(instruction, product_context)
            if cache_name and self._genai_client and _GENAI_NEW_AVAILABLE:
                logger.info("[LLM] Đang dùng prompt cache (instruction + product context) — chỉ gửi lịch sử + tin nhắn")
                dynamic_prompt = (
                    f"LỊCH SỬ TRÒ CHUYỆN (6 tin gần nhất):\n{conversation_history}\n\n"
                    f"TIN NHẮN HIỆN TẠI CỦA NGƯỜI DÙNG: {message}\n\n"
                    "Hãy trả lời một cách tự nhiên, thân thiện và hữu ích dựa trên instruction, context sản phẩm và lịch sử trò chuyện."
                )
                logger.info("[LLM][Chat] Input gửi sang Gemini (dùng cache):\n%s", dynamic_prompt)
                start_time = time.perf_counter()
                response = self._genai_client.models.generate_content(
                    model=self._model_name_for_cache,
                    contents=dynamic_prompt,
                    config=genai_types.GenerateContentConfig(
                        cached_content=cache_name,
                        temperature=0.1,
                    ),
                )
                _log_usage(response, "LLM][Chat-cache")
                elapsed_time = time.perf_counter() - start_time
                reply = (response.text or "").strip()
                logger.info("[LLM] Generate chat response (đã dùng cache) - Thời gian xử lý: %.3fs", elapsed_time)
                return reply

            # Fallback: không cache, gửi full prompt (SDK cũ)
            logger.info("[LLM] Chạy bình thường (không dùng cache) — gửi full prompt (instruction + product context + lịch sử + tin nhắn)")
            prompt_parts = []
            if instruction:
                prompt_parts.append(f"INSTRUCTION (Hướng dẫn cho chatbot):\n{instruction}\n")
            if product_context:
                prompt_parts.append(f"CONTEXT SẢN PHẨM:\n{product_context}\n")
            prompt_parts.append(f"LỊCH SỬ TRÒ CHUYỆN (6 tin gần nhất):\n{conversation_history}\n")
            prompt_parts.append(f"TIN NHẮN HIỆN TẠI CỦA NGƯỜI DÙNG: {message}\n")
            prompt_parts.append("Hãy trả lời một cách tự nhiên, thân thiện và hữu ích dựa trên instruction, context sản phẩm và lịch sử trò chuyện.")
            prompt = "\n".join(prompt_parts)
            logger.info("[LLM][Chat] Input gửi sang Gemini:\n%s", prompt)
            start_time = time.perf_counter()
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": 0.7},
            )
            _log_usage(response, "LLM][Chat")
            elapsed_time = time.perf_counter() - start_time
            reply = response.text.strip()
            logger.info("[LLM] Generate chat response (bình thường, không cache) - Thời gian xử lý: %.3fs", elapsed_time)
            return reply

        except Exception as e:
            logger.error("Lỗi khi tạo phản hồi chat: %s", e)
            return "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau."


