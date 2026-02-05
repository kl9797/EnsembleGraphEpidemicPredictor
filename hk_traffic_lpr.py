import cv2
import requests
import numpy as np
import easyocr
import os
import time
from xml.dom.minidom import parse
from ultralytics import YOLO
import matplotlib.pyplot as plt

class TrafficCamLPR:
    def __init__(self, xml_path, gpu=False):
        """
        Initialize the LPR System.
        :param xml_path: Path to the Traffic_Camera_Locations_En.xml file.
        :param gpu: Set to True if you have an NVIDIA GPU (runs OCR faster).
        """
        print("--- Initializing Traffic Cam LPR ---")
        self.cameras = self.load_cameras_from_xml(xml_path)
        
        # 1. Load YOLOv8 Model (Auto-downloads on first run)
        # 'yolov8n.pt' is the 'Nano' model (fastest). 
        print("Loading YOLOv8 Detector...")
        self.detector = YOLO('yolov8n.pt') 
        
        # 2. Load EasyOCR Model (Auto-downloads on first run)
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        print("Initialization Complete.\n")

    def load_cameras_from_xml(self, xml_path):
        """Parses the HK TD XML to get camera IDs and URLs."""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        dom = parse(xml_path)
        root = dom.documentElement
        images = root.getElementsByTagName("image")
        
        cam_list = []
        for img in images:
            try:
                key = img.getElementsByTagName("key")[0].childNodes[0].data
                url = img.getElementsByTagName("url")[0].childNodes[0].data
                desc = img.getElementsByTagName("description")[0].childNodes[0].data
                cam_list.append({"id": key, "url": url, "desc": desc})
            except IndexError:
                continue 
        
        print(f"Loaded {len(cam_list)} cameras from XML.")
        return cam_list

    def fetch_image(self, url):
        """Downloads the image directly into memory (no file save)."""
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                image_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
                return img
            else:
                print(f"Failed to fetch {url} (Status: {resp.status_code})")
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    def preprocess_plate(self, img_crop):
        """Enhances a cropped image for better OCR accuracy."""
        # Convert to Grayscale
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        
        # Upscale (Resize) to make text larger
        scale = 3
        width = int(gray.shape[1] * scale)
        height = int(gray.shape[0] * scale)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
        
        # Otsu's thresholding for high contrast
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def detect_and_read(self, cam_id, save_image=False):
        """
        Main pipeline: Fetch -> Detect Vehicle -> Crop -> OCR
        """
        # Find camera by ID
        cam = next((c for c in self.cameras if c["id"] == cam_id), None)
        if not cam:
            print(f"Camera ID {cam_id} not found.")
            return

        print(f"Processing Camera: {cam_id} ({cam['desc']})...")
        img = self.fetch_image(cam['url'])
        if img is None: return

        # Optional: Save the raw image if you are building a dataset
        if save_image:
            filename = f"captured_{cam_id}_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            print(f"   [Saved raw image to {filename}]")

        # 1. Run Object Detection
        # Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        results = self.detector(img, classes=[2, 3, 5, 7], verbose=False)

        detections = []
        original_img = img.copy()

        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < 0.5: continue

                # Draw vehicle box (Blue)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # --- PLATE LOCALIZATION HEURISTIC ---
                # Focus on bottom 30% of the vehicle for the plate
                v_h = y2 - y1
                v_w = x2 - x1
                
                p_y1 = int(y1 + (v_h * 0.70)) 
                p_y2 = y2
                p_x1 = int(x1 + (v_w * 0.10))
                p_x2 = int(x2 - (v_w * 0.10))

                if p_y1 >= p_y2 or p_x1 >= p_x2: continue
                
                plate_crop = original_img[p_y1:p_y2, p_x1:p_x2]
                
                # Skip if crop is too tiny
                if plate_crop.shape[0] < 10 or plate_crop.shape[1] < 20:
                    continue

                processed_plate = self.preprocess_plate(plate_crop)

                # 2. Run OCR
                ocr_results = self.reader.readtext(processed_plate, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

                for (bbox, text, prob) in ocr_results:
                    if prob > 0.4 and len(text) > 2:
                        print(f"   -> Detected Plate: {text} (Conf: {prob:.2f})")
                        
                        # Draw text on image (Green)
                        cv2.rectangle(img, (p_x1, p_y1), (p_x2, p_y2), (0, 255, 0), 2)
                        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        detections.append(text)

        if not detections:
            print("   No legible plates detected.")

        # Display result
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"{cam_id} - {len(detections)} Plates Found")
        plt.axis('off')
        plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup
    xml_file = "Traffic_Camera_Locations_En.xml" 
    
    # Check if XML exists
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} is missing. Please upload it to the same folder.")
    else:
        # 2. Run
        app = TrafficCamLPR(xml_file)
        
        # 3. Test on a specific camera (Aberdeen Praya Road usually has cars)
        app.detect_and_read("H429F", save_image=False)