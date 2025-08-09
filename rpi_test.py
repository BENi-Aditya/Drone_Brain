"""
Raspberry Pi Drone Brain - Live Camera Feed Version
Optimized for RPi with PiCamera module for real-time vegetation analysis and seed dropping simulation.

Features:
- Live PiCamera feed instead of video file
- Optimized for RPi hardware performance
- Real-time inference with threading
- Interactive UI with parameter controls
- Wind drift and altitude compensation
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

# --- Apply latest inference overlay to any frame ---
def apply_latest_inference_overlay(frame, wind_speed, wind_dir, altitude):
    """Apply the latest inference results to the current frame for real-time display"""
    if latest_inference['mask'] is None or latest_inference['class_map'] is None:
        # No inference available, return frame with status
        vis = frame.copy()
        cv2.putText(vis, "Waiting for inference...", (10, 30), FONT, FONT_SCALE, 
                   (0, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        return vis
    
    # Check if inference is too old (more than 5 seconds)
    if time.time() - latest_inference['timestamp'] > 5.0:
        vis = frame.copy()
        cv2.putText(vis, "Inference outdated...", (10, 30), FONT, FONT_SCALE, 
                   (0, 165, 255), FONT_THICKNESS, cv2.LINE_AA)
        return vis
    
    try:
        mask_img = latest_inference['mask']
        class_map = latest_inference['class_map']
        
        # Find vegetation and soil classes
        vegetation_id = None
        soil_id = None
        
        for class_id, class_name in class_map.items():
            if 'vegetation' in class_name.lower() or 'plant' in class_name.lower():
                vegetation_id = int(class_id)
            elif 'soil' in class_name.lower() or 'ground' in class_name.lower():
                soil_id = int(class_id)
        
        if vegetation_id is None:
            vegetation_id = max([int(k) for k in class_map.keys()])
        if soil_id is None:
            soil_id = min([int(k) for k in class_map.keys()])
        
        # Create masks
        vegetation_mask = (mask_img == vegetation_id).astype(np.uint8) * 255
        soil_mask = (mask_img == soil_id).astype(np.uint8) * 255
        
        # Resize masks to match current frame
        if mask_img.shape != (frame.shape[0], frame.shape[1]):
            vegetation_mask = cv2.resize(vegetation_mask, (frame.shape[1], frame.shape[0]))
            soil_mask = cv2.resize(soil_mask, (frame.shape[1], frame.shape[0]))
        
        # Create overlay
        vis = frame.copy().astype(np.float32)
        
        # Apply vegetation overlay (red)
        veg_indices = vegetation_mask > 128
        if np.any(veg_indices):
            red_overlay = np.zeros_like(vis)
            red_overlay[veg_indices] = [0, 0, 255]
            vis = cv2.addWeighted(vis, 1.0 - OVERLAY_ALPHA, red_overlay, OVERLAY_ALPHA, 0)
        
        # Apply soil overlay (green)
        soil_indices = soil_mask > 128
        if np.any(soil_indices):
            green_overlay = np.zeros_like(vis)
            green_overlay[soil_indices] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 1.0 - OVERLAY_ALPHA, green_overlay, OVERLAY_ALPHA, 0)
        
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        
        # Calculate and draw drop zone
        m_per_px = meters_per_pixel(frame.shape[1], altitude, FOV_DEG)
        drop_radius_px = max(5, int(DROP_RADIUS_M / m_per_px))
        drop_time = math.sqrt(2 * altitude / G)
        total_horizontal_speed = wind_speed + DRONE_SPEED
        drift_offset_m = total_horizontal_speed * drop_time
        drift_offset_px = drift_offset_m / m_per_px
        
        # Wind drift calculation
        angle_rad = math.radians(wind_dir)
        dx = drift_offset_px * math.sin(angle_rad)
        dy = -drift_offset_px * math.cos(angle_rad)
        
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        adj_x = int(center_x + dx)
        adj_y = int(center_y + dy)
        drop_spot_px = (adj_x, adj_y)
        
        # Draw drop indicators
        color = (0, 255, 0)
        cv2.line(vis, (center_x, center_y), drop_spot_px, color, LINE_THICKNESS)
        cv2.circle(vis, drop_spot_px, drop_radius_px, color, CIRCLE_THICKNESS)
        cv2.putText(vis, f"Drop Zone", 
                   (max(10, drop_spot_px[0]-30), max(30, drop_spot_px[1]-40)), 
                   FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
        
        # Add info text
        age = time.time() - latest_inference['timestamp']
        cv2.putText(vis, f"Inference: {age:.1f}s ago", (10, 30), FONT, FONT_SCALE-0.1, 
                   (255, 255, 255), FONT_THICKNESS-1, cv2.LINE_AA)
        
        return vis
        
    except Exception as e:
        print(f"[Error] Overlay application failed: {e}")
        vis = frame.copy()
        cv2.putText(vis, "Overlay error", (10, 30), FONT, FONT_SCALE, 
                   (0, 0, 255), FONT_THICKNESS, cv2.LINE_AA)
        return vis

# --- OpenCV Trackbar UI (optimized for RPi) ---
def setup_opencv_trackbars():
    cv2.namedWindow('Horn-Bill Control Panel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Horn-Bill Control Panel', 400, 200)
    
    # Create trackbars with reasonable ranges
    cv2.createTrackbar('Wind Speed (x0.1 m/s)', 'Horn-Bill Control Panel', 
                       int(WIND_SPEED_DEFAULT * 10), 200, wind_speed_cb)
    cv2.createTrackbar('Wind Dir (deg)', 'Horn-Bill Control Panel', 
                       int(WIND_DIR_DEFAULT), 360, wind_dir_cb)
    cv2.createTrackbar('Altitude (m)', 'Horn-Bill Control Panel', 
                       int(ALTITUDE_DEFAULT), 100, altitude_cb)

# --- Initialize Camera (USB/PiCamera) ---
def init_camera():
    if not PICAMERA_AVAILABLE:
        print("[Info] PiCamera2 not available. Using USB camera.")
        return init_usb_camera()
    
    try:
        picam2 = Picamera2()
        
        # Configure camera for optimal performance
        config = picam2.create_preview_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        picam2.configure(config)
        
        # Set camera controls for better image quality
        picam2.set_controls({
            "AwbEnable": True,
            "AeEnable": True,
            "FrameRate": CAMERA_FPS
        })
        
        picam2.start()
        time.sleep(2)  # Allow camera to warm up
        
        print(f"[Camera] PiCamera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
        return picam2
        
    except Exception as e:
        print(f"[Error] Failed to initialize PiCamera: {e}")
        print("[Fallback] Using USB camera...")
        return init_usb_camera()

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

# --- Inference worker thread (optimized for RPi) ---
def inference_worker():
    print("[Inference] Worker thread started")
    
    while True:
        try:
            # Get frame from queue
            with frame_queue_lock:
                if not frame_queue:
                    time.sleep(0.1)
                    continue
                frame, frame_id, wind_speed, wind_dir, altitude = frame_queue.pop(0)
            
            inference_result['processing'] = True
            
            # Get segmentation
            mask_img, class_map = get_segmentation_mask_and_classmap(frame)
            
            if mask_img is not None and class_map is not None:
                # Store latest inference results
                latest_inference['mask'] = mask_img
                latest_inference['class_map'] = class_map
                latest_inference['frame'] = frame.copy()
                latest_inference['timestamp'] = time.time()
                
                # Process segmentation results
                vegetation_id = None
                soil_id = None
                
                for class_id, class_name in class_map.items():
                    if 'vegetation' in class_name.lower() or 'plant' in class_name.lower():
                        vegetation_id = int(class_id)
                    elif 'soil' in class_name.lower() or 'ground' in class_name.lower():
                        soil_id = int(class_id)
                
                # Default fallback
                if vegetation_id is None:
                    vegetation_id = max([int(k) for k in class_map.keys()])
                if soil_id is None:
                    soil_id = min([int(k) for k in class_map.keys()])
                
                # Create masks
                vegetation_mask = (mask_img == vegetation_id).astype(np.uint8) * 255
                soil_mask = (mask_img == soil_id).astype(np.uint8) * 255
                
                # Resize mask to match frame dimensions if needed
                if mask_img.shape != (frame.shape[0], frame.shape[1]):
                    vegetation_mask = cv2.resize(vegetation_mask, (frame.shape[1], frame.shape[0]))
                    soil_mask = cv2.resize(soil_mask, (frame.shape[1], frame.shape[0]))
                
                # Create overlays with better blending
                vis = frame.copy().astype(np.float32)
                
                # Apply vegetation overlay (red)
                veg_indices = vegetation_mask > 128
                if np.any(veg_indices):
                    red_overlay = np.zeros_like(vis)
                    red_overlay[veg_indices] = [0, 0, 255]  # BGR format
                    vis = cv2.addWeighted(vis, 1.0 - OVERLAY_ALPHA, red_overlay, OVERLAY_ALPHA, 0)
                
                # Apply soil overlay (green)
                soil_indices = soil_mask > 128
                if np.any(soil_indices):
                    green_overlay = np.zeros_like(vis)
                    green_overlay[soil_indices] = [0, 255, 0]  # BGR format
                    vis = cv2.addWeighted(vis, 1.0 - OVERLAY_ALPHA, green_overlay, OVERLAY_ALPHA, 0)
                
                vis = np.clip(vis, 0, 255).astype(np.uint8)
                
                # Calculate drop parameters
                m_per_px = meters_per_pixel(frame.shape[1], altitude, FOV_DEG)
                drop_radius_px = max(5, int(DROP_RADIUS_M / m_per_px))
                drop_time = math.sqrt(2 * altitude / G)
                total_horizontal_speed = wind_speed + DRONE_SPEED
                drift_offset_m = total_horizontal_speed * drop_time
                drift_offset_px = drift_offset_m / m_per_px
                
                # Calculate wind drift
                angle_rad = math.radians(wind_dir)
                dx = drift_offset_px * math.sin(angle_rad)
                dy = -drift_offset_px * math.cos(angle_rad)
                
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                adj_x = int(center_x + dx)
                adj_y = int(center_y + dy)
                drop_spot_px = (adj_x, adj_y)
                
                # Draw drop indicators
                color = (0, 255, 0)
                cv2.line(vis, (center_x, center_y), drop_spot_px, color, LINE_THICKNESS)
                cv2.circle(vis, drop_spot_px, drop_radius_px, color, CIRCLE_THICKNESS)
                cv2.putText(vis, f"Drop @ {drop_spot_px}", 
                           (max(10, drop_spot_px[0]-40), max(30, drop_spot_px[1]-40)), 
                           FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
                
                # Add parameter info
                cv2.putText(vis, f"Alt: {altitude:.1f}m Wind: {wind_speed:.1f}m/s @ {wind_dir:.0f}deg", 
                           (10, vis.shape[0] - 30), FONT, FONT_SCALE, (255, 255, 255), 
                           FONT_THICKNESS, cv2.LINE_AA)
                
                # Add class info
                cv2.putText(vis, f"Classes: Veg({vegetation_id}) Soil({soil_id})", 
                           (10, vis.shape[0] - 10), FONT, FONT_SCALE-0.1, (255, 255, 255), 
                           FONT_THICKNESS-1, cv2.LINE_AA)
            else:
                vis = frame.copy()
                cv2.putText(vis, "Waiting for inference...", (10, 30), FONT, FONT_SCALE, 
                           (0, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
            
            # Update shared result
            inference_result['vis'] = vis
            inference_result['frame_id'] = frame_id
            inference_result['processing'] = False
            
        except Exception as e:
            print(f"[Error] Inference worker error: {e}")
            inference_result['processing'] = False
            time.sleep(1)

# --- Main Camera Loop (optimized for RPi) ---
def main_camera_loop():
    print("[Main] Starting Horn-Bill RPi camera simulation...")
    
    # Initialize camera
    camera = init_camera()
    if camera is None:
        print("[Error] Failed to initialize camera")
        return
    
    # Determine camera type
    is_picamera = PICAMERA_AVAILABLE and hasattr(camera, 'capture_array')
    print(f"[Camera] Using {'PiCamera' if is_picamera else 'USB Camera'}")
    
    # Setup UI
    setup_opencv_trackbars()
    
    # Create windows
    cv2.namedWindow("Horn-Bill Live Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Horn-Bill Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Horn-Bill Live Feed", CAMERA_WIDTH, CAMERA_HEIGHT)
    cv2.resizeWindow("Horn-Bill Inference", CAMERA_WIDTH, CAMERA_HEIGHT)
    
    frame_count = 0
    fps_time = time.time()
    
    # Start inference thread
    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()
    
    print("[Main] Camera loop started. Press 'q' to quit.")
    
    try:
        while True:
            start_time = time.time()
            
            # Capture frame
            if is_picamera:
                # PiCamera2
                try:
                    frame = camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[Error] PiCamera capture failed: {e}")
                    break
            else:
                # USB Camera
                ret, frame = camera.read()
                if not ret:
                    print("[Error] Failed to capture frame from USB camera")
                    time.sleep(0.1)
                    continue
                    
            # Validate frame
            if frame is None or frame.size == 0:
                print("[Warning] Empty frame received")
                continue
                    
            frame_count += 1
            
            # Resize frame if needed
            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            
            # Read slider values
            wind_speed = cv2.getTrackbarPos('Wind Speed (x0.1 m/s)', 'Horn-Bill Control Panel') / 10.0
            wind_dir = cv2.getTrackbarPos('Wind Dir (deg)', 'Horn-Bill Control Panel')
            altitude = cv2.getTrackbarPos('Altitude (m)', 'Horn-Bill Control Panel')
            
            # Show live feed immediately
            live_frame = frame.copy()
            cv2.putText(live_frame, f"Live Feed - Frame: {frame_count}", 
                       (10, 30), FONT, FONT_SCALE, (0, 255, 0), FONT_THICKNESS, cv2.LINE_AA)
            cv2.imshow("Horn-Bill Live Feed", live_frame)
            
            # Add frame to inference queue (skip frames for performance)
            if frame_count % INFERENCE_SKIP_FRAMES == 0:
                with frame_queue_lock:
                    if len(frame_queue) < MAX_QUEUE_SIZE:
                        frame_queue.append((frame.copy(), frame_count, wind_speed, wind_dir, altitude))
            
            # Apply latest inference overlay to current frame for real-time display
            vis = apply_latest_inference_overlay(frame, wind_speed, wind_dir, altitude)
            
            # Calculate and display FPS
            now = time.time()
            elapsed = now - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS and status info
            status_text = "Processing..." if inference_result['processing'] else f"Live FPS: {fps:.1f}"
            cv2.putText(vis, status_text, (10, 60), FONT, FONT_SCALE, (255, 255, 0), 
                       FONT_THICKNESS, cv2.LINE_AA)
            
            # Add parameter display
            cv2.putText(vis, f"Alt: {altitude:.1f}m Wind: {wind_speed:.1f}m/s @ {wind_dir:.0f}deg", 
                       (10, vis.shape[0] - 30), FONT, FONT_SCALE, (255, 255, 255), 
                       FONT_THICKNESS, cv2.LINE_AA)
            
            cv2.imshow("Horn-Bill Inference", vis)
            
            # Handle key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[Main] Quit requested")
                break
            elif key == ord('r'):
                # Reset parameters
                cv2.setTrackbarPos('Wind Speed (x0.1 m/s)', 'Horn-Bill Control Panel', 0)
                cv2.setTrackbarPos('Wind Dir (deg)', 'Horn-Bill Control Panel', 0)
                cv2.setTrackbarPos('Altitude (m)', 'Horn-Bill Control Panel', 30)
                print("[Main] Parameters reset")
            
            # Optimized frame rate control
            frame_time = time.time() - start_time
            target_frame_time = 1.0 / CAMERA_FPS
            
            # Only sleep if we're significantly ahead of schedule
            if frame_time < target_frame_time * 0.8:
                sleep_time = target_frame_time - frame_time
                if sleep_time > 0.001:  # Only sleep if > 1ms
                    time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    except Exception as e:
        print(f"[Error] Main loop error: {e}")
    finally:
        # Cleanup
        print("[Main] Cleaning up...")
        try:
            if is_picamera and hasattr(camera, 'stop'):
                camera.stop()
            elif hasattr(camera, 'release'):
                camera.release()
        except Exception as e:
            print(f"[Warning] Camera cleanup error: {e}")
        finally:
            cv2.destroyAllWindows()

# --- Main ---
if __name__ == "__main__":
    print("=" * 50)
    print("Horn-Bill Drone Brain - Raspberry Pi Version")
    print("=" * 50)
    print("Controls:")
    print("- 'q': Quit application")
    print("- 'r': Reset parameters")
    print("- Use trackbars to adjust wind speed, direction, and altitude")
    print("=" * 50)
    
    # Check OpenCV build info
    print(f"[System] OpenCV version: {cv2.__version__}")
    print(f"[System] OpenCV build info: {cv2.getBuildInformation()[:200]}...")
    
    try:
        main_camera_loop()
    except Exception as e:
        print(f"[Fatal Error] Application crashed: {e}")
        import traceback
        traceback.print_exc()
