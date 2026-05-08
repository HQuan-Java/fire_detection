import cv2
import os
import time
import numpy as np

from src.image_processing import FireImageProcessor
from src.yolo_detector import YOLOFireDetector


VIDEO_PATH = "video/fire_detec.mp4"
MODEL_PATH = "models/fire_yolo.pt"

INFO_BAR_HEIGHT = 100
MIN_FIRE_AREA = 1200
ALERT_FRAME_COUNT = 5

os.makedirs("output/frames", exist_ok=True)
os.makedirs("output/logs", exist_ok=True)

processor = FireImageProcessor()
yolo = YOLOFireDetector(MODEL_PATH, conf=0.20)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không mở được video:", VIDEO_PATH)
    exit()

fire_count = 0
frame_id = 0
last_save_time = 0
alert_blink_state = True
blink_timer = 0


def draw_info_bar(frame, h, w, status, yolo_detected, detected_labels, fire_area, fire_count):
    """Vẽ thanh thông tin phía dưới với thanh tiến trình fire count."""
    bar = np.zeros((INFO_BAR_HEIGHT, w, 3), dtype=np.uint8)

    # --- Thanh tiến trình fire count ---
    progress_ratio = min(fire_count / ALERT_FRAME_COUNT, 1.0)
    bar_w = int((w - 40) * progress_ratio)
    track_color = (30, 30, 30)
    if progress_ratio < 0.5:
        fill_color = (50, 200, 50)       # xanh lá
    elif progress_ratio < 1.0:
        fill_color = (0, 165, 255)       # cam
    else:
        fill_color = (0, 0, 220)         # đỏ

    cv2.rectangle(bar, (20, 10), (w - 20, 22), track_color, -1)
    if bar_w > 0:
        cv2.rectangle(bar, (20, 10), (20 + bar_w, 22), fill_color, -1)

    cv2.putText(bar, f"Fire count: {fire_count}/{ALERT_FRAME_COUNT}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- Trạng thái ---
    if status == "FIRE ALERT":
        status_color = (0, 0, 220)
        status_text = "!! FIRE ALERT !!"
    else:
        status_color = (50, 200, 50)
        status_text = "NORMAL"

    cv2.putText(bar, f"Status: {status_text}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # --- YOLO ---
    yolo_color = (0, 200, 255) if yolo_detected else (120, 120, 120)
    yolo_text = "YOLO: " + (", ".join(set(detected_labels)).upper() if detected_labels else "NONE")
    cv2.putText(bar, yolo_text,
                (w // 2 - 60, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yolo_color, 2)

    # --- Fire area ---
    cv2.putText(bar, f"Area: {fire_area} px",
                (w - 220, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 2)

    final = cv2.copyMakeBorder(frame, 0, INFO_BAR_HEIGHT, 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    final[h:h + INFO_BAR_HEIGHT] = bar
    return final


def draw_alert_overlay(frame, blink_on):
    """Vẽ viền đỏ nhấp nháy + text cảnh báo to khi phát hiện cháy."""
    h, w = frame.shape[:2]

    if blink_on:
        # Viền đỏ 4 cạnh
        thickness = 8
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), thickness)

        # Banner nền đỏ phía trên
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Chữ cảnh báo to
        text = "CANH BAO CHAY !"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thickness_txt = 3
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness_txt)
        tx = (w - tw) // 2
        cv2.putText(frame, text, (tx, 44), font, scale, (255, 255, 255), thickness_txt)

    return frame


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_id += 1
    display = frame.copy()
    h, w = frame.shape[:2]

    # --- Phát hiện màu lửa ---
    fire_mask, fire_area = processor.detect_fire_color(frame)

    # --- YOLO ---
    yolo_detections = yolo.detect(frame)
    yolo_detected = False
    detected_labels = []

    for det in yolo_detections:
        label = det["label"].lower()
        conf = det["confidence"]
        x1, y1, x2, y2 = det["box"]

        if "fire" in label or "smoke" in label:
            yolo_detected = True
            detected_labels.append(label)
            color = (0, 0, 255) if "fire" in label else (255, 50, 50)
            tag = f"{'FIRE' if 'fire' in label else 'SMOKE'} {conf:.2f}"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

            text_y = max(y1 - 10, 35)
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (x1, text_y - 28), (x1 + tw + 12, text_y + 8), color, -1)
            cv2.putText(display, tag, (x1 + 6, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # --- Cập nhật fire_count ---
    image_fire_detected = fire_area > MIN_FIRE_AREA
    if yolo_detected or image_fire_detected:
        fire_count = min(fire_count + 1, ALERT_FRAME_COUNT + 2)
    else:
        fire_count = max(0, fire_count - 1)

    status = "NORMAL"
    if fire_count >= ALERT_FRAME_COUNT:
        status = "FIRE ALERT"

        # Nhấp nháy overlay mỗi 10 frame
        blink_timer += 1
        if blink_timer % 10 == 0:
            alert_blink_state = not alert_blink_state

        display = draw_alert_overlay(display, alert_blink_state)

        # Lưu frame & log
        current_time = time.time()
        if current_time - last_save_time >= 2:
            cv2.imwrite(f"output/frames/alert_frame_{frame_id}.jpg", display)
            last_save_time = current_time
            with open("output/logs/fire_log.txt", "a", encoding="utf-8") as f:
                f.write(
                    f"Frame: {frame_id} | Status: {status} | "
                    f"YOLO: {yolo_detected} | Labels: {detected_labels} | "
                    f"Fire area: {fire_area} | Time: {time.ctime()}\n"
                )
    else:
        alert_blink_state = True
        blink_timer = 0

    # --- Ghép thanh info ---
    final_frame = draw_info_bar(display, h, w, status, yolo_detected,
                                detected_labels, fire_area, fire_count)

    cv2.imshow("Fire Detection System", final_frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()