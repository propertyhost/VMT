import sys
print("Python executable:", sys.executable)
import os
import cv2
import time
import json
import zipfile
import socket
import subprocess
import requests
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from ultralytics import YOLO
from paddleocr import PaddleOCR

N8N_BASE = "https://n8n.honeyhomes.in/webhook"
PLATE_ENDPOINT = f"{N8N_BASE}/vtm/plate-captured"

WORKER_ID = socket.gethostname()
MODEL_PATH = "/root/best.pt"

REMOTE_PROCESSING = "gdrive:VTM Extractor/Processing Videos"
REMOTE_DONE_BASE = "gdrive:VTM Extractor/Processed"

YOLO_CONF = 0.5
OCR_CONF = 0.6
FRAME_SKIP = 3

def run(cmd):
    print(">>", cmd)
    result = subprocess.run(cmd, shell=True)
    print("Return code:", result.returncode)
    return result.returncode == 0

def get_job():
    try:
        print("Polling for job...")
        r = requests.post(f"{N8N_BASE}/vtm/get-job", json={"worker": WORKER_ID}, timeout=20)
        print("Response:", r.text)
        return r.json()
    except Exception as e:
        print("Error polling job:", e)
        return None

def send_plate(payload):
    print("Sending plate:", payload)
    try:
        r = requests.post(PLATE_ENDPOINT, json=payload, timeout=10)
        print("Webhook status:", r.status_code)
    except Exception as e:
        print("Webhook error:", e)

def find_video(folder):
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                return os.path.join(root, f)
    return None

print("Loading YOLO...")
model = YOLO(MODEL_PATH)
model.to("cuda")
print("YOLO ready.")

print("Loading PaddleOCR...")
ocr_engine = PaddleOCR(use_angle_cls=False, use_gpu=True, lang='en')
print("OCR ready.")

print("Worker started:", WORKER_ID)

while True:

    job = get_job()

    if not job or not job.get("jobId"):
        print("No job. Sleeping 10s...")
        time.sleep(10)
        continue

    print("Job received:", job)

    job_id = job["jobId"]
    file_name = job["fileName"]

    local_dir = f"/workspace/{job_id}"
    os.makedirs(local_dir, exist_ok=True)

    print("Downloading zip...")
    if not run(f'rclone --drive-shared-with-me copy "{REMOTE_PROCESSING}/{file_name}" "{local_dir}"'):
        print("Download failed.")
        continue

    zip_path = os.path.join(local_dir, file_name)
    print("Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(local_dir)

    video_path = find_video(local_dir)
    print("Video found:", video_path)

    if not video_path:
        print("No video found inside zip.")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS:", fps)

    frame_index = 0

    print("Starting frame loop...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % FRAME_SKIP == 0:
            results = model(frame, conf=YOLO_CONF, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                print("Detection found at frame", frame_index)

                for box in boxes.xyxy:
                    x1,y1,x2,y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]

                    ocr_result = ocr_engine.ocr(crop)
                    print("OCR raw:", ocr_result)

                    if ocr_result and ocr_result[0]:
                        text, conf = ocr_result[0][0][1]
                        if conf >= OCR_CONF:
                            plate = ''.join(c for c in text.upper() if c.isalnum())
                            print("Plate detected:", plate)

                            send_plate({
                                "jobId": job_id,
                                "plate": plate
                            })

        frame_index += 1

    cap.release()

    print("Uploading results folder...")
    run(f'rclone --drive-shared-with-me mkdir "{REMOTE_DONE_BASE}/{job_id}"')
    run(f'rclone --drive-shared-with-me moveto "{REMOTE_PROCESSING}/{file_name}" "{REMOTE_DONE_BASE}/{job_id}/{file_name}"')

    print("Job finished.")
