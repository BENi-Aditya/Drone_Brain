#!/usr/bin/env python3
"""
Fixed version of test_01.py with proper indentation
Raspberry Pi Drone Brain - Live Camera Feed Version
"""

import cv2
import numpy as np
import requests
import base64
import math
import datetime
import time
import os
import threading

# Fix Qt display issues on RPi
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA_AVAILABLE = True
except ImportError:
    print("[Warning] PiCamera2 not available. Install with: sudo apt install python3-picamera2")
    PICAMERA_AVAILABLE = False

# --- CONFIG (optimized for RPi) ---
API_KEY = "hjCFUfsJBCSxi8KzP8yC"
MODEL_ENDPOINT = "vegetation-gvb0s/7"
SEED_MASS = 10.0  # grams (fixed)
DRONE_SPEED = 12.0  # m/s
ALTITUDE_DEFAULT = 30.0  # meters
WIND_SPEED_DEFAULT = 0.0  # m/s
WIND_DIR_DEFAULT = 0.0  # degrees
FOV_DEG = 62.0  # PiCam v2/v3
DROP_RADIUS_M = 1.0  # meters
OVERLAY_ALPHA = 0.4
CIRCLE_THICKNESS = 3  # Reduced for RPi performance
LINE_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6  # Reduced for RPi
FONT_THICKNESS = 2
FONT_COLOR = (0, 0, 0)
G = 9.81

# --- RPi Camera Settings (Optimized for speed) ---
CAMERA_WIDTH = 480  # Further reduced for speed
CAMERA_HEIGHT = 360
CAMERA_FPS = 30  # Higher FPS for smoother feed
INFERENCE_SKIP_FRAMES = 10  # Process every 10th frame for inference
CAMERA_BUFFER_SIZE = 1  # Minimal buffer for real-time

# --- Hardcoded thrust values for typical DIY quadcopter ---
THRUST_PER_MOTOR_N = 7.85  # Newtons (approx 800g per motor)
NUM_MOTORS = 4
TOTAL_THRUST_N = THRUST_PER_MOTOR_N * NUM_MOTORS  # 31.4 N

# --- UI State (shared between threads) ---
ui_state = {
    'wind_speed': WIND_SPEED_DEFAULT,
    'wind_dir': WIND_DIR_DEFAULT,
    'altitude': ALTITUDE_DEFAULT,
    'update': False
}

# --- Global slider state ---
slider_values = {
    'wind_speed': WIND_SPEED_DEFAULT,
    'wind_dir': WIND_DIR_DEFAULT,
    'altitude': ALTITUDE_DEFAULT
}

def wind_speed_cb(val):
    slider_values['wind_speed'] = max(0, min(val / 10.0, 20.0))

def wind_dir_cb(val):
    slider_values['wind_dir'] = max(0, min(val, 360))

def altitude_cb(val):
    slider_values['altitude'] = max(5, min(val, 100))

# --- Shared state for inference result ---
inference_result = {
    'vis': None,
    'frame_id': -1,
    'processing': False
}

# --- Frame queue for inference (optimized) ---
frame_queue = []
frame_queue_lock = threading.Lock()
MAX_QUEUE_SIZE = 2  # Smaller queue for faster processing

# --- Latest inference result cache ---
latest_inference = {
    'frame': None,
    'mask': None,
    'class_map': None,
    'timestamp': 0
}

# --- Helper: Encode image as base64 ---
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Send image to Roboflow Segmentation API ---
def get_segmentation_mask_and_classmap(image_bgr, temp_path="_frame.jpg"):
    try:
        cv2.imwrite(temp_path, image_bgr)
        url = f"https://segment.roboflow.com/{MODEL_ENDPOINT}?api_key={API_KEY}&name={os.path.basename(temp_path)}"
        img_b64 = encode_image_base64(temp_path)
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        response = requests.post(url, data=img_b64, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print("Error from Roboflow API:", response.text)
            return None, None
            
        data = response.json()
        mask_b64 = data.get("segmentation_mask")
        class_map = data.get("class_map")
        
        if not mask_b64 or not class_map:
            print("No segmentation mask or class_map in response.")
            return None, None
            
        mask_bytes = base64.b64decode(mask_b64)
        mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return mask_arr, class_map
        
    except Exception as e:
        print(f"[Error] API request failed: {e}")
        return None, None

# --- Pixel-to-meter calculation ---
def meters_per_pixel(image_width, altitude, fov_deg):
    fov_rad = math.radians(fov_deg)
    ground_width = 2 * altitude * math.tan(fov_rad / 2)
    return ground_width / image_width

# --- Initialize USB Camera ---
def init_usb_camera():
    """Initialize USB camera with optimized settings for speed"""
    # Try different backends for better performance
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
        except:
            continue
    else:
        print("[Error] Could not open USB camera with any backend")
        return None
    
    # Set camera properties for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    
    # Additional optimizations
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Faster exposure
    
    # Test capture
    ret, test_frame = cap.read()
    if not ret:
        print("[Error] Could not read from USB camera")
        cap.release()
        return None
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[Camera] USB camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    return cap

# --- Simple test function ---
def test_camera():
    print("Testing camera initialization...")
    camera = init_usb_camera()
    
    if camera is None:
        print("❌ Camera initialization failed")
        return False
    
    print("✅ Camera initialized successfully")
    
    # Test a few frames
    for i in range(5):
        ret, frame = camera.read()
        if not ret:
            print(f"❌ Failed to read frame {i+1}")
            camera.release()
            return False
        print(f"✅ Frame {i+1}: {frame.shape}")
    
    camera.release()
    print("✅ Camera test completed successfully")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Camera Test - Fixed Indentation Version")
    print("=" * 50)
    
    if test_camera():
        print("\n🎉 Camera is working properly!")
        print("You can now run the main application.")
    else:
        print("\n❌ Camera test failed!")
        print("Please check your camera connection.")
