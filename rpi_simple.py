#!/usr/bin/env python3
"""
Simplified Raspberry Pi Drone Brain - Camera Only Version
No API calls, just camera feed with basic UI for testing
"""

import cv2
import numpy as np
import time
import os

# Fix Qt display issues on RPi
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

# --- CONFIG ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# --- UI State ---
wind_speed = 0.0
wind_dir = 0.0
altitude = 30.0

def wind_speed_cb(val):
    global wind_speed
    wind_speed = max(0, min(val / 10.0, 20.0))

def wind_dir_cb(val):
    global wind_dir
    wind_dir = max(0, min(val, 360))

def altitude_cb(val):
    global altitude
    altitude = max(5, min(val, 100))

def setup_trackbars():
    """Setup control trackbars"""
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 200)
    
    cv2.createTrackbar('Wind Speed (x0.1 m/s)', 'Controls', 0, 200, wind_speed_cb)
    cv2.createTrackbar('Wind Dir (deg)', 'Controls', 0, 360, wind_dir_cb)
    cv2.createTrackbar('Altitude (m)', 'Controls', 30, 100, altitude_cb)

def init_camera():
    """Initialize USB camera"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Error] Could not open camera")
        return None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Test capture
    ret, test_frame = cap.read()
    if not ret:
        print("[Error] Could not read from camera")
        cap.release()
        return None
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[Camera] Initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
    return cap

def draw_overlay(frame):
    """Draw simple overlay with parameters"""
    global wind_speed, wind_dir, altitude
    
    # Create overlay
    overlay = frame.copy()
    
    # Draw center crosshair
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Draw crosshair
    cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
    
    # Draw simulated drop zone
    drop_radius = max(20, int(100 / altitude))  # Larger radius for lower altitude
    cv2.circle(overlay, (center_x, center_y), drop_radius, (0, 255, 255), 2)
    
    # Draw wind direction indicator
    if wind_speed > 0:
        angle_rad = np.radians(wind_dir)
        wind_length = int(wind_speed * 10)  # Scale wind vector
        end_x = int(center_x + wind_length * np.sin(angle_rad))
        end_y = int(center_y - wind_length * np.cos(angle_rad))
        cv2.arrowedLine(overlay, (center_x, center_y), (end_x, end_y), (255, 0, 0), 3)
    
    # Add parameter text
    cv2.putText(overlay, f"Altitude: {altitude:.1f}m", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Wind: {wind_speed:.1f}m/s @ {wind_dir:.0f}deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay

def main():
    print("=" * 50)
    print("Simplified Horn-Bill Camera Test")
    print("=" * 50)
    print("Controls:")
    print("- 'q': Quit")
    print("- 'r': Reset parameters")
    print("- Use trackbars to adjust parameters")
    print("=" * 50)
    
    # Initialize camera
    camera = init_camera()
    if camera is None:
        return
    
    # Setup UI
    setup_trackbars()
    
    # Create windows
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    print("[Main] Starting camera loop. Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("[Error] Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Validate frame
            if frame is None or frame.size == 0:
                print("[Warning] Empty frame")
                continue
            
            frame_count += 1
            
            # Resize if needed
            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            
            # Read trackbar values
            wind_speed = cv2.getTrackbarPos('Wind Speed (x0.1 m/s)', 'Controls') / 10.0
            wind_dir = cv2.getTrackbarPos('Wind Dir (deg)', 'Controls')
            altitude = cv2.getTrackbarPos('Altitude (m)', 'Controls')
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS to frame
            live_frame = frame.copy()
            cv2.putText(live_frame, f"FPS: {fps:.1f} | Frame: {frame_count}", 
                       (10, live_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
            
            # Create overlay version
            overlay_frame = draw_overlay(frame)
            cv2.putText(overlay_frame, f"FPS: {fps:.1f}", (10, overlay_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frames
            cv2.imshow("Live Feed", live_frame)
            cv2.imshow("Overlay", overlay_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset parameters
                cv2.setTrackbarPos('Wind Speed (x0.1 m/s)', 'Controls', 0)
                cv2.setTrackbarPos('Wind Dir (deg)', 'Controls', 0)
                cv2.setTrackbarPos('Altitude (m)', 'Controls', 30)
                print("[Reset] Parameters reset")
            
            # Frame rate control
            time.sleep(1.0 / CAMERA_FPS)
            
    except KeyboardInterrupt:
        print("\n[Interrupted] Stopping...")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("[Cleanup] Releasing camera...")
        camera.release()
        cv2.destroyAllWindows()
        
    print(f"[Results] Processed {frame_count} frames in {elapsed:.1f} seconds")
    print(f"[Results] Average FPS: {fps:.1f}")

if __name__ == "__main__":
    main()
