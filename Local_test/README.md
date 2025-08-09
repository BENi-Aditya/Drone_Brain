# Local_test: Local Roboflow Model Inference

This folder demonstrates how to run your Roboflow segmentation model locally for fast, real-time inference (no cloud API required).

## Steps:

1. **Export Your Model from Roboflow**
   - Go to your Roboflow project: [Roboflow Vegetation Model](https://app.roboflow.com/beni-a0itp/vegetation-gvb0s-86ebu/annotate)
   - Click "Export Model"
   - Select `ONNX` (recommended for cross-platform) or `TFLite` (for Raspberry Pi)
   - Download the exported model file (e.g., `model.onnx` or `model.tflite`)
   - Place the model file in this `Local_test` directory

2. **Install Python dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the demo**
   ```sh
   python local_infer_webcam.py --model model.onnx
   # or for TFLite
   python local_infer_webcam.py --model model.tflite --tflite
   ```

## Files
- `local_infer_webcam.py`: Main script for webcam/video inference
- `requirements.txt`: Python dependencies
- `model.onnx` or `model.tflite`: Your exported model (add this yourself)

---

If you need help exporting your model from Roboflow or want to use a different runtime, let me know!
