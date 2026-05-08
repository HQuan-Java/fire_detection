import requests
import threading


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self._base   = f"https://api.telegram.org/bot{token}"

    def _send(self, text: str, image_path: str | None = None):
        """Chạy trong thread riêng để không block luồng video."""
        try:
            if image_path:
                with open(image_path, "rb") as img:
                    requests.post(
                        f"{self._base}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": text},
                        files={"photo": img},
                        timeout=10,
                    )
            else:
                requests.post(
                    f"{self._base}/sendMessage",
                    data={"chat_id": self.chat_id, "text": text},
                    timeout=10,
                )
        except Exception as e:
            print(f"[Telegram] Lỗi gửi: {e}")

    def notify(self, text: str, image_path: str | None = None):
        """Gửi bất đồng bộ — không làm trễ video."""
        threading.Thread(
            target=self._send,
            args=(text, image_path),
            daemon=True,
        ).start()