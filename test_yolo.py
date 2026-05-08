from ultralytics import YOLO

# load model
model = YOLO("models/fire_yolo.pt")

# chạy trên video
results = model("video/fire.mp4", show=True)