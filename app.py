import cv2
import os
import time
import json
import threading
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template, jsonify

import config
from src.image_processing import FireImageProcessor
from src.yolo_detector import YOLOFireDetector
from src.telegram_notifier import TelegramNotifier
from src.vlm_analyzer import GeminiFireAnalyzer

app = Flask(__name__)

os.makedirs("output/frames", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)

# --- Shared state (thread-safe đủ dùng cho demo) ---
state = {
    "status":         "NORMAL",
    "yolo_detected":  False,
    "detected_labels": [],
    "fire_area":      0,
    "fire_count":     0,
    "frame_id":       0,
    "total_alerts":   0,
    "today_alerts":   0,
    "tg_sent":        0,
    "start_time":     datetime.now().strftime("%H:%M:%S"),
}
alert_log   = deque(maxlen=50)   # 50 bản ghi gần nhất
hourly_data = {str(h): 0 for h in range(24)}
output_frame = None
frame_lock   = threading.Lock()


def detection_loop():
    global output_frame

    processor = FireImageProcessor()
    yolo      = YOLOFireDetector(config.MODEL_PATH, conf=config.YOLO_CONF)
    notifier  = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    cap       = cv2.VideoCapture(config.VIDEO_PATH)
    analyzer = GeminiFireAnalyzer(config.GEMINI_API_KEY)

    if not cap.isOpened():
        print("Không mở được video:", config.VIDEO_PATH)
        return

    fire_count     = 0
    last_save_time = 0
    blink_state    = True
    blink_timer    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            continue

        state["frame_id"] += 1
        frame_id = state["frame_id"]
        display  = frame.copy()
        h, w     = frame.shape[:2]

        fire_mask, fire_area = processor.detect_fire_color(frame)
        yolo_detections      = yolo.detect(frame)

        yolo_detected   = False
        detected_labels = []

        for det in yolo_detections:
            label = det["label"].lower()
            conf  = det["confidence"]
            x1, y1, x2, y2 = det["box"]

            if "fire" in label or "smoke" in label:
                yolo_detected = True
                detected_labels.append(label)
                color = (0, 0, 255) if "fire" in label else (255, 50, 50)
                tag   = f"{'FIRE' if 'fire' in label else 'SMOKE'} {conf:.2f}"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                text_y = max(y1 - 10, 35)
                (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display, (x1, text_y - 28), (x1 + tw + 12, text_y + 8), color, -1)
                cv2.putText(display, tag, (x1 + 6, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if yolo_detected or fire_area > config.MIN_FIRE_AREA:
            fire_count = min(fire_count + 1, config.ALERT_FRAME_COUNT + 2)
        else:
            fire_count = max(0, fire_count - 1)

        status = "FIRE ALERT" if fire_count >= config.ALERT_FRAME_COUNT else "NORMAL"

        # --- Overlay cảnh báo ---
        if status == "FIRE ALERT":
            blink_timer += 1
            if blink_timer % 10 == 0:
                blink_state = not blink_state
            if blink_state:
                cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 220), 8)
                overlay = display.copy()
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
                cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
                text = "CANH BAO CHAY !"
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)
                cv2.putText(display, text, ((w - tw) // 2, 44),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

            current_time = time.time()
            if current_time - last_save_time >= config.ALERT_COOLDOWN:
              last_save_time = current_time
              img_path = f"output/frames/alert_frame_{frame_id}.jpg"
              cv2.imwrite(img_path, display)

              # Gọi Gemini phân tích
              vlm = analyzer.analyze(display)

              hour = datetime.now().hour
              hourly_data[str(hour)] = hourly_data.get(str(hour), 0) + 1
              state["total_alerts"] += 1
              state["today_alerts"] += 1
              state["tg_sent"]      += 1

              log_entry = {
                  "frame_id":       frame_id,
                  "status":         "FIRE ALERT",
                  "yolo":           yolo_detected,
                  "labels":         list(set(detected_labels)),
                  "area":           fire_area,
                  "time":           datetime.now().strftime("%H:%M:%S"),
                  "img_path":       img_path,
                  "vlm_confirmed":  vlm["confirmed"],
                  "vlm_desc":       vlm["description"],
                  "vlm_severity":   vlm["severity"],
              }
              alert_log.appendleft(log_entry)

              with open("output/logs/fire_log.txt", "a", encoding="utf-8") as f:
                  f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

              # Gửi Telegram kèm phân tích VLM
              severity_icon = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(vlm["severity"], "🔴")
              confirmed_text = "XÁC NHẬN" if vlm["confirmed"] else "NGHI NGỜ"
              msg = (
                  f"🔥 CẢNH BÁO CHÁY — {confirmed_text}\n"
                  f"{severity_icon} Mức độ: {vlm['severity'].upper()}\n"
                  f"📋 Phân tích: {vlm['description']}\n\n"
                  f"Frame: {frame_id}\n"
                  f"YOLO: {', '.join(set(detected_labels)) or 'N/A'}\n"
                  f"Fire area: {fire_area} px\n"
                  f"Thời gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
              )
              notifier.notify(msg, img_path)
        else:
            blink_state = True
            blink_timer = 0

        # Cập nhật state
        state["status"]          = status
        state["yolo_detected"]   = yolo_detected
        state["detected_labels"] = list(set(detected_labels))
        state["fire_area"]       = fire_area
        state["fire_count"]      = fire_count

        # Encode JPEG cho stream
        _, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            output_frame = jpeg.tobytes()

        time.sleep(0.025)   # ~40 fps max


# --- Flask routes ---

def gen_frames():
    global output_frame
    while True:
        with frame_lock:
            frame = output_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)   # ~30 fps gửi về browser


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify({**state, "hourly": hourly_data})


@app.route("/api/logs")
def api_logs():
    return jsonify(list(alert_log))


if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("Dashboard: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)