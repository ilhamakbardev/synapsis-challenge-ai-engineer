"""
WARNING: Please ensure that Docker is running before executing this script.

This script requires the vehicle detection, tracking, and counting system to be running inside Docker containers.
Make sure to start the Docker containers first by running the following command in the terminal of root directory:

    docker-compose up --build

Once the containers are up and running, you can execute this script to process the video stream and log detections.

If Docker is not running or the containers are not started, the script will fail to work properly.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse
import cv2
import requests
from datetime import datetime
import os
import json


parser = argparse.ArgumentParser(
    prog='yolov10s',
    description='Track vehicles strictly inside a polygon and count total entries, exits, and current count.'
)

parser.add_argument(
    '-i', '--input',
    default="https://restreamer.kotabogor.go.id/memfs/2a7383a3-78b6-4af9-b7d5-7454ac3924fa_output_0.m3u8?session=LVKygG8x2MLzhJV7vhJmWP",
    help='Input video source URL or path (default: Bogor City CCTV stream)'
)

args = parser.parse_args()

# base url of docker api.py
API_BASE_URL = "http://localhost:8000/api" 

class VehicleCounter:
    def __init__(self, input_video_path):
        self.model = YOLO('yolov10s.pt')
        self.input_video_path = input_video_path
        self.polygon = []
        self.inside = set()
        self.entered = set()
        self.exited = set()
        self.area_id = None


    def save_area(self, area_name):
        coordinates_str = json.dumps([[str(x), str(y)] for x, y in self.polygon])

        response = requests.post(f"{API_BASE_URL}/areas/", json={
            "area_name": area_name,
            "coordinates": str(coordinates_str)
        }, timeout=10)

        print({
            "area_name": area_name,
            "coordinates": str(coordinates_str)
        })

        if response.status_code == 200:
            area_data = response.json()
            self.area_id = area_data.get("id")
        else:
            raise Exception(f"Failed to save area: {response.text}")


    def log_detection(self, vehicle_id, status, area_id):
        detection_data = {
            "vehicle_id": vehicle_id,
            "status": status,
            "area_id": area_id,
            "timestamp": datetime.now().isoformat()
        }
        
        for key, value in detection_data.items():
            if isinstance(value, np.float32): 
                detection_data[key] = float(value) 
        
        response = requests.post(f"http://localhost:8000/api/detections/", json=detection_data, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Failed to log detection: {response.text}")


    def set_polygon(self):
        area_name = input("Enter area name: ")

        def draw(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon.append((x, y))

        cv2.namedWindow('Set Vehicle Zone')
        cv2.setMouseCallback('Set Vehicle Zone', draw)
        cap = cv2.VideoCapture(self.input_video_path)
        ret, frame = cap.read()

        while ret:
            temp_frame = frame.copy()
            if len(self.polygon) > 1:
                cv2.polylines(temp_frame, [np.array(self.polygon)], isClosed=True, color=(255, 0, 0), thickness=2)
            for point in self.polygon:
                cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            cv2.imshow('Set Vehicle Zone', temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        self.polygon = np.array(self.polygon, dtype=np.int32)
        cap.release()

        self.save_area(area_name)


    def is_inside_polygon(self, point):
        return cv2.pointPolygonTest(self.polygon.astype(np.float32), tuple(map(int, point)), False) >= 0


    def process_frame(self, frame):
        results = self.model.track(frame, imgsz=640, persist=True)[0]
        current_inside = set()

        for box, cls, obj_id in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.id.cpu().numpy() if results.boxes.id is not None else []):
            if int(cls) in [2, 5, 7]:  # car, bus, truck only
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if self.is_inside_polygon(center):
                    current_inside.add(obj_id)
                    if obj_id not in self.inside:
                        self.log_detection(obj_id, 'entered', self.area_id)
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID {obj_id}', (int(box[0]), int(box[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for vehicle_id in current_inside - self.inside:
            self.entered.add(vehicle_id)

        for vehicle_id in self.inside - current_inside:
            self.exited.add(vehicle_id)
            self.log_detection(vehicle_id, 'exited', self.area_id)

        self.inside = current_inside

        def draw_text_with_bg(img, text, pos, font_scale=0.7, font_thickness=2, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(img, pos, (pos[0] + w + 5, pos[1] - h - 5), bg_color, -1)
            cv2.putText(img, text, (pos[0], pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        draw_text_with_bg(frame, f'Inside Area: {len(self.inside)}', (10, 40), text_color=(255, 255, 255), bg_color=(255, 165, 0))
        draw_text_with_bg(frame, f'Entered: {len(self.entered)}', (10, 70), text_color=(255, 255, 255), bg_color=(0, 128, 0))
        draw_text_with_bg(frame, f'Exited: {len(self.exited)}', (10, 100), text_color=(255, 255, 255), bg_color=(0, 0, 255))

        cv2.polylines(frame, [self.polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        return frame


    def process_video(self):
        self.set_polygon()

        cap = cv2.VideoCapture(self.input_video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame)
            cv2.imshow('Vehicle Counter with Entries, Exits, and Inside Count', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = VehicleCounter(args.input)
    obj.process_video()
