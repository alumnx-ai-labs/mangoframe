import os
import cv2
import csv
import re
from ultralytics import YOLO
import pysrt

# ============================
# 1. Configs
# ============================
INPUT_FOLDER = r"InputFolder"                                   # folder for input videos
UNIQUE_DIR = r"outputFolder/unique_images"                      # folder for unique plants
MANGO_DIR = os.path.join(UNIQUE_DIR, "mangoTree")              # folder for mango trees
NOT_MANGO_DIR = os.path.join(UNIQUE_DIR, "notMangoTree")       # folder for non-mango trees
OUTPUT_VIDEO_DIR = r"outputFolder/video"                        # folder for annotated videos
CSV_PATH = r"detections.csv"                                    # detailed log
COUNT_PATH = r"unique_count.txt"                                # summary log
TRACKER_CONFIG = "bytetrack.yaml"                               # tracker config file
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
os.makedirs(UNIQUE_DIR, exist_ok=True)
os.makedirs(MANGO_DIR, exist_ok=True)
os.makedirs(NOT_MANGO_DIR, exist_ok=True)

# ============================
# 2. Load YOLO Model
# ============================
# Update this path to your newly trained single-class model
model = YOLO("yolov8n-custom-v001.pt")   # your trained mango model

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

# ============================
# 4. Process Multiple Videos
# ============================
video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
seen_ids = set()
saved_images = {}  # Track saved images with their areas and paths

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["video_file", "frame", "track_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "lat", "lon", "image_path"])

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
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, class_id, confidence in zip(boxes, ids, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    lat, lon = gps_data.get(frame_num, (None, None))
                    
                    # Get class name and color
                    if class_id == 0:  # mangoTree
                        class_name = "mangoTree"
                        color = (255, 0, 0)  # Blue in BGR
                    else:  # class_id == 1, notMangoTree
                        class_name = "notMangoTree"
                        color = (0, 0, 255)  # Red in BGR

                    # ============================
                    # Draw box + ID + Confidence (Class-specific colors)
                    # ============================
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

                    # Track ID and confidence label
                    label = f"ID:{track_id} {confidence:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    # Lat/Lon label (below bounding box)
                    if lat is not None and lon is not None:
                        gps_label = f"Lat:{lat:.6f}, Lon:{lon:.6f}"
                        (gtw, gth), _ = cv2.getTextSize(gps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y2 + 5), (x1 + gtw + 6, y2 + gth + 10), color, -1)
                        cv2.putText(frame, gps_label, (x1 + 3, y2 + gth + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Calculate current crop area
                    current_area = (x2 - x1) * (y2 - y1)
                    crop = frame[y1:y2, x1:x2]

                    # Determine save directory based on class
                    save_dir = MANGO_DIR if class_name == "mangoTree" else NOT_MANGO_DIR

                    # Generate image path
                    if lat is not None and lon is not None:
                        img_path = os.path.join(save_dir, f"{class_name}_{track_id}_lon_{lon}_lat_{lat}.jpg")
                    else:
                        img_path = os.path.join(save_dir, f"{class_name}_{track_id}.jpg")

                    # Create unique key for class + track_id combination
                    unique_key = f"{class_name}_{track_id}"

                    # Save logic: first time OR 15% larger area
                    if unique_key not in seen_ids:
                        # First time seeing this tree
                        seen_ids.add(unique_key)
                        cv2.imwrite(img_path, crop)
                        saved_images[unique_key] = {'area': current_area, 'path': img_path}
                        print(f"New {class_name} saved: ID {track_id}, area: {current_area}")
                    elif current_area > saved_images[unique_key]['area'] * 1.15:  # 15% larger
                        # Current crop is 15% larger, replace old image
                        old_path = saved_images[unique_key]['path']
                        if os.path.exists(old_path):
                            os.remove(old_path)  # Delete old image
                        cv2.imwrite(img_path, crop)
                        saved_images[unique_key] = {'area': current_area, 'path': img_path}
                        print(f"Better {class_name} image saved: ID {track_id}, new area: {current_area}")
                        img_path = img_path  # Keep path for CSV
                    else:
                        img_path = ""  # No new image saved

                    # Log to CSV
                    writer.writerow([video_file, frame_num, track_id, class_name, confidence, x1, y1, x2, y2, lat, lon, img_path])

            # Save annotated frame
            out.write(frame)

        cap.release()
        out.release()
        print(f"✅ Completed processing: {video_file}")

# ============================
# 5. Save Unique Count
# ============================
mango_count = len([uid for uid in seen_ids if uid.startswith("mangoTree")])
not_mango_count = len([uid for uid in seen_ids if uid.startswith("notMangoTree")])

with open(COUNT_PATH, "w") as f:
    f.write(f"Total unique mangoTree detected: {mango_count}\n")
    f.write(f"Total unique notMangoTree detected: {not_mango_count}\n")
    f.write(f"Total unique objects detected: {len(seen_ids)}\n")

print("✅ Processing complete!")
print(f"Annotated videos saved in: {OUTPUT_VIDEO_DIR}")
print(f"Unique Trees saved in: {UNIQUE_DIR}")
print(f"Detections CSV: {CSV_PATH}")
print(f"Unique count summary: {COUNT_PATH}")
print(f"mangoTree detected: {mango_count}, notMangoTree detected: {not_mango_count}")