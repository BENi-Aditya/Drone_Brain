#!/usr/bin/env python3
"""
Performance Monitor for Horn-Bill Drone Brain
Monitor FPS, CPU usage, memory usage, and inference timing
"""

import cv2
import time
import psutil
import threading
import numpy as np
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=10)
        
        self.frame_count = 0
        self.start_time = time.time()
        self.last_update = time.time()
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_system(self):
        """Background thread to monitor system resources"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_history.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_history.append(memory.percent)
                
                time.sleep(0.5)  # Update every 500ms
            except Exception as e:
                print(f"[Monitor] Error: {e}")
                time.sleep(1)
    
    def update_frame(self):
        """Call this for each frame processed"""
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS every second
        if current_time - self.last_update >= 1.0:
            elapsed = current_time - self.last_update
            fps = (self.frame_count - len(self.fps_history)) / elapsed if elapsed > 0 else 0
            self.fps_history.append(fps)
            self.last_update = current_time
    
    def record_inference_time(self, inference_time):
        """Record time taken for inference"""
        self.inference_times.append(inference_time)
    
    def get_stats(self):
        """Get current performance statistics"""
        current_time = time.time()
        total_elapsed = current_time - self.start_time
        overall_fps = self.frame_count / total_elapsed if total_elapsed > 0 else 0
        
        stats = {
            'overall_fps': overall_fps,
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'cpu_percent': self.cpu_history[-1] if self.cpu_history else 0,
            'avg_cpu': np.mean(self.cpu_history) if self.cpu_history else 0,
            'memory_percent': self.memory_history[-1] if self.memory_history else 0,
            'avg_memory': np.mean(self.memory_history) if self.memory_history else 0,
            'avg_inference_time': np.mean(self.inference_times) if self.inference_times else 0,
            'total_frames': self.frame_count,
            'uptime': total_elapsed
        }
        
        return stats
    
    def draw_overlay(self, frame):
        """Draw performance overlay on frame"""
        stats = self.get_stats()
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Background rectangle
        cv2.rectangle(overlay, (w-250, 10), (w-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        
        # Performance text
        texts = [
            f"FPS: {stats['current_fps']:.1f} (avg: {stats['avg_fps']:.1f})",
            f"CPU: {stats['cpu_percent']:.1f}% (avg: {stats['avg_cpu']:.1f}%)",
            f"RAM: {stats['memory_percent']:.1f}% (avg: {stats['avg_memory']:.1f}%)",
            f"Inference: {stats['avg_inference_time']:.2f}s",
            f"Frames: {stats['total_frames']}",
            f"Uptime: {stats['uptime']:.1f}s"
        ]
        
        y_start = 30
        for i, text in enumerate(texts):
            y = y_start + i * 20
            cv2.putText(frame, text, (w-240, y), font, font_scale, color, thickness)
        
        # FPS graph (mini)
        if len(self.fps_history) > 1:
            graph_width = 200
            graph_height = 40
            graph_x = w - 240
            graph_y = 140
            
            # Background
            cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (50, 50, 50), -1)
            
            # FPS line
            fps_data = list(self.fps_history)
            max_fps = max(fps_data) if fps_data else 30
            
            for i in range(1, len(fps_data)):
                x1 = graph_x + int((i-1) * graph_width / len(fps_data))
                y1 = graph_y + graph_height - int(fps_data[i-1] * graph_height / max_fps)
                x2 = graph_x + int(i * graph_width / len(fps_data))
                y2 = graph_y + graph_height - int(fps_data[i] * graph_height / max_fps)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        return frame
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.get_stats()
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Overall FPS: {stats['overall_fps']:.2f}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Average CPU: {stats['avg_cpu']:.1f}%")
        print(f"Average Memory: {stats['avg_memory']:.1f}%")
        print(f"Average Inference Time: {stats['avg_inference_time']:.3f}s")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Uptime: {stats['uptime']:.1f}s")
        print("="*50)
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)

# Example usage
if __name__ == "__main__":
    print("Performance Monitor Test")
    
    monitor = PerformanceMonitor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Could not open camera")
        exit()
    
    print("Press 'q' to quit, 's' to show summary")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            monitor.update_frame()
            
            # Add performance overlay
            frame_with_overlay = monitor.draw_overlay(frame)
            
            cv2.imshow("Performance Monitor Test", frame_with_overlay)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                monitor.print_summary()
    
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        monitor.stop()
        cap.release()
        cv2.destroyAllWindows()
        monitor.print_summary()
