#!/usr/bin/env python3
"""
Camera Diagnostic Script for Raspberry Pi
Comprehensive camera detection and troubleshooting
"""

import cv2
import os
import subprocess
import sys
import time

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_system_info():
    """Check basic system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # OpenCV version
    print(f"OpenCV version: {cv2.__version__}")
    
    # OS info
    code, out, err = run_command("uname -a")
    if code == 0:
        print(f"System: {out.strip()}")
    
    # Check if running on RPi
    code, out, err = run_command("cat /proc/device-tree/model 2>/dev/null")
    if code == 0:
        print(f"Device: {out.strip()}")

def check_usb_devices():
    """Check USB devices"""
    print("\n" + "=" * 60)
    print("USB DEVICES")
    print("=" * 60)
    
    # List USB devices
    code, out, err = run_command("lsusb")
    if code == 0:
        print("USB devices found:")
        for line in out.split('\n'):
            if line.strip():
                print(f"  {line}")
                # Look for common camera keywords
                if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'logitech', 'microsoft', 'creative']):
                    print(f"    ^^^ POTENTIAL CAMERA DEVICE ^^^")
    else:
        print("Could not list USB devices")

def check_video_devices():
    """Check video devices"""
    print("\n" + "=" * 60)
    print("VIDEO DEVICES")
    print("=" * 60)
    
    # Check /dev/video* devices
    code, out, err = run_command("ls -la /dev/video* 2>/dev/null")
    if code == 0:
        print("Video devices found:")
        print(out)
    else:
        print("No /dev/video* devices found")
    
    # Check v4l2 info if available
    for i in range(5):
        code, out, err = run_command(f"v4l2-ctl --device=/dev/video{i} --info 2>/dev/null")
        if code == 0:
            print(f"\nVideo{i} info:")
            print(out)

def check_camera_permissions():
    """Check camera permissions"""
    print("\n" + "=" * 60)
    print("PERMISSIONS CHECK")
    print("=" * 60)
    
    # Check user groups
    code, out, err = run_command("groups")
    if code == 0:
        groups = out.strip().split()
        print(f"User groups: {groups}")
        
        required_groups = ['video', 'dialout']
        missing_groups = [g for g in required_groups if g not in groups]
        
        if missing_groups:
            print(f"⚠️  Missing groups: {missing_groups}")
            print("Add to groups with:")
            for group in missing_groups:
                print(f"  sudo usermod -a -G {group} $USER")
        else:
            print("✅ User has required permissions")

def test_opencv_backends():
    """Test different OpenCV backends"""
    print("\n" + "=" * 60)
    print("OPENCV BACKEND TEST")
    print("=" * 60)
    
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_ANY, "Any")
    ]
    
    for backend_id, backend_name in backends:
        print(f"\nTesting {backend_name} backend...")
        try:
            cap = cv2.VideoCapture(0, backend_id)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"  ✅ {backend_name}: Working ({w}x{h})")
                else:
                    print(f"  ❌ {backend_name}: Opened but no frame")
                cap.release()
            else:
                print(f"  ❌ {backend_name}: Failed to open")
        except Exception as e:
            print(f"  ❌ {backend_name}: Error - {e}")

def test_multiple_indices():
    """Test multiple camera indices"""
    print("\n" + "=" * 60)
    print("CAMERA INDEX TEST")
    print("=" * 60)
    
    working_cameras = []
    
    for i in range(10):  # Test indices 0-9
        print(f"Testing camera index {i}...")
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"  ✅ Camera {i}: Working ({w}x{h} @ {fps}fps)")
                    working_cameras.append(i)
                else:
                    print(f"  ⚠️  Camera {i}: Opened but no frame")
                cap.release()
            else:
                print(f"  ❌ Camera {i}: Not available")
        except Exception as e:
            print(f"  ❌ Camera {i}: Error - {e}")
        
        time.sleep(0.1)  # Small delay between tests
    
    return working_cameras

def test_gstreamer_pipeline():
    """Test GStreamer pipeline"""
    print("\n" + "=" * 60)
    print("GSTREAMER PIPELINE TEST")
    print("=" * 60)
    
    # Test if GStreamer is available
    code, out, err = run_command("gst-launch-1.0 --version 2>/dev/null")
    if code != 0:
        print("GStreamer not available")
        return
    
    print("GStreamer available, testing pipeline...")
    
    # Try GStreamer pipeline
    pipeline = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"
    
    try:
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ GStreamer pipeline: Working ({w}x{h})")
            else:
                print(f"  ❌ GStreamer pipeline: No frame")
            cap.release()
        else:
            print(f"  ❌ GStreamer pipeline: Failed to open")
    except Exception as e:
        print(f"  ❌ GStreamer pipeline: Error - {e}")

def generate_solutions(working_cameras):
    """Generate solutions based on findings"""
    print("\n" + "=" * 60)
    print("SOLUTIONS & RECOMMENDATIONS")
    print("=" * 60)
    
    if working_cameras:
        print(f"✅ Found working cameras at indices: {working_cameras}")
        print(f"🔧 Update your code to use index {working_cameras[0]} instead of 0")
        print(f"   Change: cv2.VideoCapture({working_cameras[0]})")
    else:
        print("❌ No working cameras found. Try these solutions:")
        print()
        print("1. Check physical connection:")
        print("   - Unplug and reconnect USB camera")
        print("   - Try different USB port")
        print("   - Check USB cable")
        print()
        print("2. Install camera drivers:")
        print("   sudo apt update")
        print("   sudo apt install v4l-utils")
        print("   sudo apt install uvcdynctrl")
        print()
        print("3. Check camera permissions:")
        print("   sudo usermod -a -G video $USER")
        print("   sudo usermod -a -G dialout $USER")
        print("   # Then logout and login again")
        print()
        print("4. Test camera manually:")
        print("   v4l2-ctl --list-devices")
        print("   v4l2-ctl --device=/dev/video0 --info")
        print("   cheese  # GUI camera app")
        print()
        print("5. Reboot the system:")
        print("   sudo reboot")

def main():
    print("🔍 CAMERA DIAGNOSTIC TOOL")
    print("This tool will help diagnose camera detection issues")
    print()
    
    check_system_info()
    check_usb_devices()
    check_video_devices()
    check_camera_permissions()
    test_opencv_backends()
    working_cameras = test_multiple_indices()
    test_gstreamer_pipeline()
    generate_solutions(working_cameras)
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
