#!/usr/bin/env python3
"""
Camera Fix Script - Try different methods to detect and use camera
"""

import cv2
import numpy as np
import time
import os

def try_camera_method_1():
    """Standard OpenCV method"""
    print("Method 1: Standard cv2.VideoCapture(0)")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("  ✅ Method 1 works!")
                return True
        print("  ❌ Method 1 failed")
        return False
    except Exception as e:
        print(f"  ❌ Method 1 error: {e}")
        return False

def try_camera_method_2():
    """Try with V4L2 backend"""
    print("Method 2: V4L2 backend")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("  ✅ Method 2 works!")
                return True
        print("  ❌ Method 2 failed")
        return False
    except Exception as e:
        print(f"  ❌ Method 2 error: {e}")
        return False

def try_camera_method_3():
    """Try different camera indices"""
    print("Method 3: Different camera indices")
    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    print(f"  ✅ Method 3 works with index {i}!")
                    return i
                cap.release()
        except Exception as e:
            pass
    print("  ❌ Method 3 failed")
    return False

def try_camera_method_4():
    """Try GStreamer pipeline"""
    print("Method 4: GStreamer pipeline")
    try:
        pipeline = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("  ✅ Method 4 works!")
                return True
        print("  ❌ Method 4 failed")
        return False
    except Exception as e:
        print(f"  ❌ Method 4 error: {e}")
        return False

def create_working_camera_script(method, index=0):
    """Create a working camera script based on successful method"""
    
    if method == 1:
        camera_code = "cap = cv2.VideoCapture(0)"
    elif method == 2:
        camera_code = "cap = cv2.VideoCapture(0, cv2.CAP_V4L2)"
    elif method == 3:
        camera_code = f"cap = cv2.VideoCapture({index})"
    elif method == 4:
        camera_code = '''pipeline = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)'''
    else:
        return
    
    script_content = f'''#!/usr/bin/env python3
"""
Working Camera Script - Generated automatically
"""

import cv2
import time

def main():
    print("Starting camera with working method...")
    
    # Working camera initialization
    {camera_code}
    
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Add info to frame
        cv2.putText(frame, f"FPS: {{fps:.1f}} Frame: {{frame_count}}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Working Camera", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {{frame_count}} frames in {{elapsed:.1f}} seconds")

if __name__ == "__main__":
    main()
'''
    
    with open("working_camera.py", "w") as f:
        f.write(script_content)
    
    os.chmod("working_camera.py", 0o755)
    print(f"✅ Created working_camera.py using method {method}")

def main():
    print("🔧 CAMERA FIX TOOL")
    print("Trying different camera detection methods...")
    print()
    
    # Try different methods
    methods = [
        (try_camera_method_1, 1),
        (try_camera_method_2, 2),
        (try_camera_method_3, 3),
        (try_camera_method_4, 4)
    ]
    
    working_method = None
    camera_index = 0
    
    for method_func, method_num in methods:
        result = method_func()
        if result:
            working_method = method_num
            if isinstance(result, int):  # Method 3 returns index
                camera_index = result
            break
    
    print()
    if working_method:
        print(f"🎉 Found working method: {working_method}")
        create_working_camera_script(working_method, camera_index)
        print()
        print("Next steps:")
        print("1. Run: python3 working_camera.py")
        print("2. If it works, update your main script with the working method")
    else:
        print("❌ No working camera method found")
        print()
        print("Troubleshooting steps:")
        print("1. Run: python3 diagnose_camera.py")
        print("2. Check USB connection")
        print("3. Install camera drivers: sudo apt install v4l-utils")
        print("4. Add user to video group: sudo usermod -a -G video $USER")
        print("5. Reboot system")

if __name__ == "__main__":
    main()
