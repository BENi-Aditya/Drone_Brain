"""yea but not the best 
think more
more easier and faster

think and harcode fixed values like what should be the thrust value at that altitude take an average seed weight 
and anything you can take average that doesnt change that uch approxiamte that 

then 

add two windows and one panel 
for the ui 

on the one window it will show the live feed which for now is a video in mp4 file that keeps looping

below shows the models inference with fps count and also the drop location being tracked witht the location
so example i run themodel it will find out the location to drop but as the drone moves that location also movees right so that lovcatiion should be tracked so the circle remains on that exact spot no matter it the drone moves

and if the circle goesout of frame it will analyse again for a new drop spot and track that spot to the ground 

and for the 3rd tab it should show a panel where i can directly add the values and change the values of parametres like the wind and altitude directly on the ui and not needed to be done in the teminal 

make the ui interactive"""

import cv2
import numpy as np
import requests
import base64
import math
import datetime
import time
import os
import threading
import queue  # <-- Added for frame sharing

# --- CONFIG (hardcoded/approximate) ---
API_KEY = "hjCFUfsJBCSxi8KzP8yC"
MODEL_ENDPOINT = "vegetation-gvb0s/7"
VIDEO_PATH = "video.mov"  # Use the available video file
SEED_MASS = 10.0  # grams (fixed)
DRONE_SPEED = 12.0  # m/s
ALTITUDE_DEFAULT = 30.0  # meters
WIND_SPEED_DEFAULT = 0.0  # m/s
WIND_DIR_DEFAULT = 0.0  # degrees
FOV_DEG = 62.0  # PiCam v2/v3
DROP_RADIUS_M = 1.0  # meters
OVERLAY_ALPHA = 0.4
CIRCLE_THICKNESS = 4
LINE_THICKNESS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 3
FONT_COLOR = (0, 0, 0)
G = 9.81

# --- NEW: Hardcoded thrust values for typical DIY quadcopter ---
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

# --- Shared frame queue for webcam frames ---
frame_queue = queue.Queue(maxsize=3)

# --- Shared state for inference result ---
inference_result = {
    'vis': None,
    'frame_id': -1
}

# --- Helper: Encode image as base64 ---
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Send image to Roboflow Segmentation API ---
def get_segmentation_mask_and_classmap(image_bgr, temp_path="_frame.jpg"):
    cv2.imwrite(temp_path, image_bgr)
    url = f"https://segment.roboflow.com/{MODEL_ENDPOINT}?api_key={API_KEY}&name={os.path.basename(temp_path)}"
    img_b64 = encode_image_base64(temp_path)
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=img_b64, headers=headers)
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
    return mask_arr, class_map

# --- Pixel-to-meter calculation ---
def meters_per_pixel(image_width, altitude, fov_deg):
    fov_rad = math.radians(fov_deg)
    ground_width = 2 * altitude * math.tan(fov_rad / 2)
    return ground_width / image_width

# --- OpenCV Trackbar UI (robust with callbacks) ---
def setup_opencv_trackbars():
    cv2.namedWindow('Horn-Bill Inference')
    cv2.createTrackbar('Wind Speed (m/s)', 'Horn-Bill Inference', int(WIND_SPEED_DEFAULT * 10), 200, wind_speed_cb)
    cv2.createTrackbar('Wind Dir (deg)', 'Horn-Bill Inference', int(WIND_DIR_DEFAULT), 360, wind_dir_cb)
    cv2.createTrackbar('Altitude (m)', 'Horn-Bill Inference', int(ALTITUDE_DEFAULT), 100, altitude_cb)

# --- Manual parameter entry via OpenCV overlay ---
def opencv_text_input(window_name, prompt, default_value):
    # Draw prompt and get input from keyboard
    input_str = str(default_value)
    while True:
        img = 255 * np.ones((120, 400, 3), dtype=np.uint8)
        cv2.putText(img, prompt, (10, 40), FONT, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, input_str, (10, 90), FONT, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(0)
        if key == 13 or key == 10:  # Enter
            break
        elif key == 27:  # ESC
            return default_value
        elif key == 8 or key == 127:  # Backspace
            input_str = input_str[:-1]
        elif 48 <= key <= 57 or key == 46:  # 0-9 or .
            input_str += chr(key)
    try:
        return float(input_str)
    except:
        return default_value

# --- Thread-safe parameter state ---
param_lock = threading.Lock()
shared_params = {
    'wind_speed': WIND_SPEED_DEFAULT,
    'wind_dir': WIND_DIR_DEFAULT,
    'altitude': ALTITUDE_DEFAULT
}

# --- Inference worker thread ---
def inference_worker():
    frame_id = 0
    INFER_EVERY_N = 50  # Increased to skip more frames for speed
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        frame_id += 1
        if frame_id % INFER_EVERY_N != 1:
            continue
        # Downscale frame for faster API inference
        small_frame = cv2.resize(frame, (320, 240))
        # Get current slider values
        wind_speed = cv2.getTrackbarPos('Wind Speed (m/s)', 'Horn-Bill Inference') / 10.0
        wind_dir = cv2.getTrackbarPos('Wind Dir (deg)', 'Horn-Bill Inference')
        altitude = cv2.getTrackbarPos('Altitude (m)', 'Horn-Bill Inference')
        # Run inference
        mask_img, class_map = get_segmentation_mask_and_classmap(small_frame)
        if mask_img is None:
            vis = frame.copy()
            cv2.putText(vis, "Model error", (10, 60), FONT, 1, (0,0,255), 3, cv2.LINE_AA)
        else:
            mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            vegetation_id = None
            soil_id = None
            for k, v in class_map.items():
                if "vegetation" in v.lower():
                    vegetation_id = int(k)
                if "soil" in v.lower() or "bare" in v.lower() or "background" in v.lower():
                    soil_id = int(k)
            if vegetation_id is None:
                vegetation_id = max([int(k) for k in class_map.keys()])
            if soil_id is None:
                soil_id = min([int(k) for k in class_map.keys()])
            vegetation_mask = (mask_img == vegetation_id).astype(np.uint8) * 255
            soil_mask = (mask_img == soil_id).astype(np.uint8) * 255
            red_overlay = np.zeros_like(frame, dtype=np.uint8)
            green_overlay = np.zeros_like(frame, dtype=np.uint8)
            red_overlay[vegetation_mask > 128] = [0, 0, 255]
            green_overlay[soil_mask > 128] = [0, 255, 0]
            blend = frame.copy().astype(np.float32)
            blend = cv2.addWeighted(blend, 1.0, red_overlay.astype(np.float32), OVERLAY_ALPHA, 0)
            blend = cv2.addWeighted(blend, 1.0, green_overlay.astype(np.float32), OVERLAY_ALPHA, 0)
            blend = np.clip(blend, 0, 255).astype(np.uint8)
            
            # --- IMPROVED DROP LOCATION CALCULATION ---
            h, w = mask_img.shape
            center_x, center_y = w // 2, h // 2
            
            # Calculate meters per pixel for this altitude
            m_per_px = meters_per_pixel(frame.shape[1], altitude, FOV_DEG)
            drop_radius_px = max(10, int(DROP_RADIUS_M / m_per_px))  # Minimum 10px radius
            
            # Physics: Calculate drop time and drift
            drop_time = math.sqrt(2 * altitude / G)
            
            # Separate wind and drone movement effects
            wind_drift_m = wind_speed * drop_time
            drone_drift_m = DRONE_SPEED * drop_time * 0.3  # Reduced drone effect (30%)
            
            # Convert to pixels
            wind_drift_px = wind_drift_m / m_per_px
            drone_drift_px = drone_drift_m / m_per_px
            
            # Calculate wind direction offset
            wind_angle_rad = math.radians(wind_dir)
            wind_dx = wind_drift_px * math.sin(wind_angle_rad)
            wind_dy = -wind_drift_px * math.cos(wind_angle_rad)
            
            # Add drone forward movement (assume drone moves in positive x direction)
            drone_dx = drone_drift_px
            drone_dy = 0
            
            # Total drift
            total_dx = wind_dx + drone_dx
            total_dy = wind_dy + drone_dy
            
            # Calculate initial drop spot
            initial_drop_x = center_x + total_dx
            initial_drop_y = center_y + total_dy
            
            # --- BOUNDS CHECKING AND ADJUSTMENT ---
            margin = max(drop_radius_px + 20, 50)  # Margin from edge
            
            # Clamp to frame bounds with margin
            adj_x = max(margin, min(w - margin, int(initial_drop_x)))
            adj_y = max(margin, min(h - margin, int(initial_drop_y)))
            
            # If the calculated spot is too far off-screen, scale it back
            if abs(initial_drop_x - center_x) > w * 0.4 or abs(initial_drop_y - center_y) > h * 0.4:
                # Scale back the drift to keep it reasonable
                scale_factor = min(w * 0.4 / abs(initial_drop_x - center_x) if abs(initial_drop_x - center_x) > 0 else 1,
                                 h * 0.4 / abs(initial_drop_y - center_y) if abs(initial_drop_y - center_y) > 0 else 1)
                adj_x = int(center_x + total_dx * scale_factor)
                adj_y = int(center_y + total_dy * scale_factor)
                # Re-clamp after scaling
                adj_x = max(margin, min(w - margin, adj_x))
                adj_y = max(margin, min(h - margin, adj_y))
            
            drop_spot_px = (adj_x, adj_y)
            
            # --- VISUALIZATION ---
            vis = blend
            # Draw drift vector from center to drop spot
            color = (0, 255, 0)  # Green for normal drop
            if abs(initial_drop_x - adj_x) > 5 or abs(initial_drop_y - adj_y) > 5:
                color = (0, 255, 255)  # Yellow if position was adjusted
            
            # Draw line from drone center to drop spot
            cv2.line(vis, (center_x, center_y), drop_spot_px, color, LINE_THICKNESS)
            cv2.circle(vis, drop_spot_px, drop_radius_px, color, CIRCLE_THICKNESS)
            cv2.circle(vis, (center_x, center_y), 8, (255, 0, 0), -1)
            drift_distance_m = math.sqrt(total_dx**2 + total_dy**2) * m_per_px
            text_y_offset = max(drop_spot_px[1] - drop_radius_px - 10, 20)
            cv2.putText(vis, f"Drop Zone", (drop_spot_px[0]-40, text_y_offset), FONT, 0.6, FONT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Drift: {drift_distance_m:.1f}m", (drop_spot_px[0]-40, text_y_offset + 20), FONT, 0.5, FONT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(vis, f"Time: {drop_time:.1f}s", (drop_spot_px[0]-40, text_y_offset + 35), FONT, 0.5, FONT_COLOR, 2, cv2.LINE_AA)
            if wind_speed > 0.1:
                wind_arrow_start = (center_x + 60, center_y - 60)
                wind_arrow_end = (int(wind_arrow_start[0] + 30 * math.sin(wind_angle_rad)), 
                                int(wind_arrow_start[1] - 30 * math.cos(wind_angle_rad)))
                cv2.arrowedLine(vis, wind_arrow_start, wind_arrow_end, (255, 255, 0), 3)
                cv2.putText(vis, f"Wind {wind_speed:.1f}m/s", (wind_arrow_start[0]-20, wind_arrow_start[1]-10), 
                          FONT, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
        inference_result['vis'] = vis
        inference_result['frame_id'] = frame_id

# --- Main Video Processing Loop (UI always smooth) ---
def main_video_loop():
    print("[Main] Starting Horn-Bill video simulation...")
    setup_opencv_trackbars()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not access the webcam. Please ensure your webcam is connected and not in use.")
        return
    frame_count = 0
    fps_time = time.time()
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0 or video_fps > 120:
        video_fps = 30
    print(f"[Main] Video FPS: {video_fps}")
    # Start inference thread
    t = threading.Thread(target=inference_worker, daemon=True)
    t.start()
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        # Feed frame to inference queue (non-blocking, drop if full)
        try:
            if not frame_queue.full():
                frame_queue.put_nowait(frame.copy())
        except Exception:
            pass
        frame_count += 1
        # Read slider values directly
        wind_speed = cv2.getTrackbarPos('Wind Speed (m/s)', 'Horn-Bill Inference') / 10.0
        wind_dir = cv2.getTrackbarPos('Wind Dir (deg)', 'Horn-Bill Inference')
        altitude = cv2.getTrackbarPos('Altitude (m)', 'Horn-Bill Inference')
        # Show live feed immediately
        cv2.imshow("Horn-Bill Live Feed", frame)
        # Show latest inference result (even if from previous frame)
        vis = inference_result['vis'] if inference_result['vis'] is not None else frame.copy()
        now = time.time()
        elapsed = now - fps_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(vis, f"FPS: {fps:.2f}", (10, 30), FONT, 1, (0,0,0), 3, cv2.LINE_AA)
        cv2.imshow("Horn-Bill Inference", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        frame_time = time.time() - start_time
        wait_time = max(1, int(1000 / video_fps - frame_time * 1000))
        cv2.waitKey(wait_time)
    cap.release()
    cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    print("[System] Launching Horn-Bill simulation...")
    main_video_loop() 