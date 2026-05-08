import cv2
import json
import re
import base64
import numpy as np
from google import genai
from google.genai import types


class GeminiFireAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        try:
            _, buf = cv2.imencode(".jpg", frame_bgr)
            image_bytes = base64.b64encode(buf.tobytes()).decode("utf-8")

            prompt = """Bạn là hệ thống phân tích cháy nổ chuyên nghiệp.
Hãy phân tích ảnh này và trả lời ĐÚNG format JSON sau, không giải thích thêm:

{
  "confirmed": true hoặc false,
  "description": "mô tả ngắn tình huống bằng tiếng Việt, tối đa 2 câu",
  "severity": "low" hoặc "medium" hoặc "high"
}

Quy tắc:
- confirmed = true nếu thấy lửa thật, khói dày, hoặc vật thể đang cháy
- confirmed = false nếu chỉ là ánh sáng đỏ, hoàng hôn, đèn, hoặc không rõ
- severity: low = khói nhẹ/lửa nhỏ, medium = lửa vừa, high = cháy lớn/lan rộng"""

            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(data=base64.b64decode(image_bytes), mime_type="image/jpeg"),
                    types.Part.from_text(text=prompt),
                ]
            )

            text = response.text.strip()
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())

        except Exception as e:
            print(f"[Gemini] Lỗi: {e}")

        return {"confirmed": True, "description": "Không thể phân tích.", "severity": "medium"}