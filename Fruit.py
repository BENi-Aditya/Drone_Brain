import cv2
import numpy as np
import requests

# Initialize constants for the Roboflow API
API_URL = "https://detect.roboflow.com/fruit-ripeness-detection-1ht7j/4"
API_KEY = "hjCFUfsJBCSxi8KzP8yC"

# Function to convert OpenCV frame to bytes suitable for the API
def preprocess_frame(frame):
    _, encoded_image = cv2.imencode('.jpg', frame)
    return encoded_image.tobytes()

# Function to send image to Roboflow API and print predictions
def send_to_model(frame):
    image_bytes = preprocess_frame(frame)
    files = {"file": image_bytes}
    try:
        response = requests.post(f"{API_URL}?api_key={API_KEY}", files=files)
        response_data = response.json()

        if "predictions" in response_data:
            predictions = response_data["predictions"]
            print("Predictions:")
            for prediction in predictions:
                print(f"Class: {prediction['class']}, Confidence: {prediction['confidence'] * 100:.2f}%")
        else:
            print("No predictions received.")
    except Exception as e:
        print(f"Error: {e}")

# Start webcam feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam. Press 's' to send a frame to the model, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Show the live webcam feed
    cv2.imshow('Fruit Quality Detection', frame)

    # Handle key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("Sending frame to model...")
        send_to_model(frame)

    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()