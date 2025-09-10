import os
import cv2
import csv
import re
from ultralytics import YOLO
import pysrt


# ============================
# 1. Configs
# ============================
INPUT_FOLDER = r"InputFolder"        # folder for input videos
UNIQUE_DIR = r"outputFolder/unique_images"                      # folder for unique plants
OUTPUT_VIDEO_DIR = r"outputFolder/video"                # folder for annotated videos
CSV_PATH = r"detections.csv"         # detailed log
COUNT_PATH = r"unique_count.txt"     # summary log
TRACKER_CONFIG = "bytetrack.yaml"                                                                                 # tracker config file
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(UNIQUE_DIR, exist_ok=True)


# ============================
# 2. Load YOLO Model
# ============================
model = YOLO("yolov11n_335th_epoch_patience_50_best_mAP50.pt")   # your trained mango model

# ============================
# 3. Parse SRT Metadata (FIXED)
# ============================
def parse_srt(srt_path):
    subs = pysrt.open(srt_path)
    gps_map = {}
    for i, sub in enumerate(subs):
        frame_num = i + 1
        text = sub.text.strip()

        # Regex to capture latitude and longitude
        lat_match = re.search(r"latitude:\s*([-\d.]+)", text)
        lon_match = re.search(r"longitude:\s*([-\d.]+)", text)

        if lat_match and lon_match:
            lat = float(lat_match.group(1))
            lon = float(lon_match.group(1))
            gps_map[frame_num] = (lat, lon)
        else:
            gps_map[frame_num] = (None, None)

    return gps_map

# gps_data = parse_srt(SRT_PATH)

# ============================
# 4. Process Multiple Videos
# ============================
video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
seen_ids = set()

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video_file", "frame", "track_id", "x1", "y1", "x2", "y2", "lat", "lon", "image_path"])

    for video_file in video_files:
        print(f"Processing video: {video_file}")

        VIDEO_PATH = os.path.join(INPUT_FOLDER, video_file)
        video_name = os.path.splitext(video_file)[0]
        SRT_PATH = os.path.join(INPUT_FOLDER, f"{video_name}.SRT")
        OUTPUT_VIDEO = os.path.join(OUTPUT_VIDEO_DIR, f"{video_name}.mp4")

        # Parse GPS data for this video
        if os.path.exists(SRT_PATH):
            gps_data = parse_srt(SRT_PATH)
        else:
            print(f"Warning: No SRT file found for {video_file}")
            gps_data = {}

        # Prepare Video Writer
        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Run YOLOv8 detection + tracking
            results = model.track(frame, persist=True, tracker=TRACKER_CONFIG)

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []

                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    lat, lon = gps_data.get(frame_num, (None, None))

                    # ============================
                    # Draw box + ID + GPS (Yellow, Thick, Visible)
                    # ============================
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)

                    # Track ID label
                    label = f"ID:{track_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 255), -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    # Lat/Lon label (below bounding box)
                    if lat is not None and lon is not None:
                        gps_label = f"Lat:{lat:.6f}, Lon:{lon:.6f}"
                        (gtw, gth), _ = cv2.getTextSize(gps_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        cv2.rectangle(frame, (x1, y2 + 5), (x1 + gtw + 6, y2 + gth + 10), (0, 255, 255), -1)
                        cv2.putText(frame, gps_label, (x1 + 3, y2 + gth + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # Save first unique crop
                    if track_id not in seen_ids:
                        seen_ids.add(track_id)
                        crop = frame[y1:y2, x1:x2]
                        if lat is not None and lon is not None:
                            # Include longitude and latitude in the filename
                            img_path = os.path.join(UNIQUE_DIR, f"plant_{track_id}_lon_{lon}_lat_{lat}.jpg")
                        else:
                            img_path = os.path.join(UNIQUE_DIR, f"plant_{track_id}.jpg") # Fallback if GPS data is missing
                        cv2.imwrite(img_path, crop)
                    else:
                        img_path = ""

                    # Log to CSV
                    writer.writerow([video_file, frame_num, track_id, x1, y1, x2, y2, lat, lon, img_path])

            # Save annotated frame
            out.write(frame)

cap.release()
out.release()
print(f"✅ Completed processing: {video_file}")

# ============================
# 5. Save Unique Count
# ============================
with open(COUNT_PATH, "w") as f:
    f.write(f"Total unique plants detected: {len(seen_ids)}\n")

print("✅ Processing complete!")
print(f"Annotated videos saved in: {OUTPUT_VIDEO_DIR}")
print(f"Unique Mango Trees saved in: {UNIQUE_DIR}")
print(f"Detections CSV: {CSV_PATH}")
print(f"Unique count summary: {COUNT_PATH}")