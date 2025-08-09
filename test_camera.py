#!/usr/bin/env python3
"""
Simple USB Camera Test Script
Use this to verify your USB camera is working before running the main application.
"""

import cv2
import numpy as np
import time

def test_usb_camera():
    print("=" * 50)
    print("USB Camera Test")
    print("=" * 50)
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Error] Could not open camera")
        print("Try these commands:")
        print("1. lsusb  # Check if camera is detected")
        print("2. ls /dev/video*  # Check video devices")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[Camera] Default resolution: {width}x{height}")
    print(f"[Camera] Default FPS: {fps}")
    
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Check actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"[Camera] Set resolution: {actual_width}x{actual_height}")
    print(f"[Camera] Set FPS: {actual_fps}")
    
    # Test frame capture
    print("\n[Test] Capturing frames...")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[Error] Failed to capture frame")
                break
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info to frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("USB Camera Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[Saved] {filename}")
                
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    print(f"\n[Results] Captured {frame_count} frames in {elapsed:.1f} seconds")
    print(f"[Results] Average FPS: {current_fps:.1f}")
    
    return True

def check_system():
    """Check system information"""
    print("\n" + "=" * 50)
    print("System Information")
    print("=" * 50)
    
    # OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # Available cameras
    print("\nChecking available cameras...")
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera {i}: Available ({width}x{height})")
            cap.release()
        else:
            print(f"Camera {i}: Not available")

if __name__ == "__main__":
    check_system()
    
    if test_usb_camera():
        print("\n[Success] Camera test completed!")
        print("You can now run the main application: python3 rpi_test.py")
    else:
        print("\n[Failed] Camera test failed!")
        print("Please check your camera connection and try again.")
