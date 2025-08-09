import argparse
import cv2
import numpy as np
import time

# Try to import ONNX and TFLite runtime
try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None


def run_onnx_inference(model_path, frame):
    # Preprocess: assume model expects 320x320, 3ch RGB, float32
    img = cv2.resize(frame, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred = sess.run([output_name], {input_name: img})[0]
    # Assume output is (1, H, W) or (1, C, H, W)
    mask = pred.squeeze()
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def run_tflite_inference(model_path, frame):
    # Preprocess: assume model expects 320x320, 3ch RGB, float32
    img = cv2.resize(frame, (320, 320))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    mask = interpreter.get_tensor(output_details[0]['index'])
    mask = mask.squeeze()
    if mask.ndim == 3:
        mask = mask[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to ONNX or TFLite model')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite runtime')
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default 0)')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Error: Could not open webcam')
        return
    print('Press q to quit')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        if args.tflite:
            if tflite is None:
                print('TFLite runtime not installed!')
                break
            mask = run_tflite_inference(args.model, frame)
        else:
            if ort is None:
                print('ONNX Runtime not installed!')
                break
            mask = run_onnx_inference(args.model, frame)
        t1 = time.time()
        # Overlay mask
        overlay = frame.copy()
        overlay[mask > 128] = [0, 255, 0]
        vis = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        fps = 1.0 / (t1 - t0)
        cv2.putText(vis, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Local Model Inference', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
