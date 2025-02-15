import numpy as np
import supervision as sv
from ultralytics import YOLO
import argparse
import cv2
import psycopg2
from datetime import datetime
import os


parser = argparse.ArgumentParser(
    prog='yolov8',
    description='Track vehicles strictly inside a polygon and count total entries, exits, and current count.'
)
parser.add_argument(
    '-i', '--input',
    default="https://restreamer.kotabogor.go.id/memfs/2a7383a3-78b6-4af9-b7d5-7454ac3924fa_output_0.m3u8?session=LVKygG8x2MLzhJV7vhJmWP",
    help='Input video source URL or path (default: Bogor City CCTV stream)'
)
args = parser.parse_args()

DB_CONFIG = {
    'dbname': os.getenv("POSTGRES_DB", "vehicles-counter"),
    'user': os.getenv("POSTGRES_USER", "postgres"),
    'password': os.getenv("POSTGRES_PASSWORD", "admin"),
    'host': os.getenv("POSTGRES_HOST", "db"),  # Docker service name
    'port': os.getenv("POSTGRES_PORT", "5432")
}



class VehicleCounter:
    def __init__(self, input_video_path):
        self.model = YOLO('yolov10s.pt')
        self.input_video_path = input_video_path
        self.polygon = []
        self.inside = set()
        self.entered = set()
        self.exited = set()
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS area_configurations (
                    id SERIAL PRIMARY KEY,
                    area_name VARCHAR(255) UNIQUE,
                    coordinates TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS vehicle_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP,
                    vehicle_id VARCHAR(255),
                    status VARCHAR(50),
                    area_id INT,
                    FOREIGN KEY (area_id) REFERENCES area_configurations(id)
                );
            ''')
            self.conn.commit()

    def save_area(self, area_name):
        coords = ','.join([f'({x},{y})' for x, y in self.polygon])
        with self.conn.cursor() as cur:
            cur.execute('''
                INSERT INTO area_configurations (area_name, coordinates)
                VALUES (%s, %s)
                ON CONFLICT (area_name) DO NOTHING;
            ''', (area_name, coords))
            self.conn.commit()

    def log_detection(self, vehicle_id, status, area_id):
        with self.conn.cursor() as cur:
            cur.execute('''
                INSERT INTO vehicle_logs (timestamp, vehicle_id, status, area_id)
                VALUES (%s, %s, %s, %s);
            ''', (datetime.now(), str(vehicle_id), status, area_id))
            self.conn.commit()

    def get_area_id(self, area_name):
        with self.conn.cursor() as cur:
            cur.execute(
                'SELECT id FROM area_configurations WHERE area_name = %s;', (area_name,))
            result = cur.fetchone()
            return result[0] if result else None

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
                cv2.polylines(temp_frame, [np.array(self.polygon)], isClosed=True, color=(
                    255, 0, 0), thickness=2)
            for point in self.polygon:
                cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            cv2.imshow('Set Vehicle Zone', temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        self.polygon = np.array(self.polygon, dtype=np.int32)
        cap.release()

        self.save_area(area_name)
        self.area_id = self.get_area_id(area_name)

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
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(
                        box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID {obj_id}', (int(box[0]), int(box[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for vehicle_id in current_inside - self.inside:
            self.entered.add(vehicle_id)
        for vehicle_id in self.inside - current_inside:
            self.exited.add(vehicle_id)
            self.log_detection(vehicle_id, 'exited', self.area_id)

        self.inside = current_inside

        def draw_text_with_bg(img, text, pos, font_scale=0.7, font_thickness=2, text_color=(0, 0, 0), bg_color=(255, 255, 255)):
            (w, h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(
                img, pos, (pos[0] + w + 5, pos[1] - h - 5), bg_color, -1)
            cv2.putText(img, text, (pos[0], pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

        draw_text_with_bg(frame, f'Inside Area: {len(self.inside)}', (10, 40), text_color=(
            255, 255, 255), bg_color=(255, 165, 0))   # Blue bg
        draw_text_with_bg(frame, f'Entered: {len(self.entered)}', (10, 70), text_color=(
            255, 255, 255), bg_color=(0, 128, 0))  # Green bg
        draw_text_with_bg(frame, f'Exited: {len(self.exited)}', (10, 100), text_color=(
            255, 255, 255), bg_color=(0, 0, 255))  # Red bg

        cv2.polylines(frame, [self.polygon], isClosed=True,
                      color=(255, 0, 0), thickness=2)

        return frame

    def process_video(self):
        self.set_polygon()
        cap = cv2.VideoCapture(self.input_video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            cv2.imshow(
                'Vehicle Counter with Entries, Exits, and Inside Count', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    obj = VehicleCounter(args.input)
    obj.process_video()
