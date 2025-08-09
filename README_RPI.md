# Horn-Bill Drone Brain - Raspberry Pi Version

A real-time vegetation analysis and seed dropping simulation system optimized for Raspberry Pi with PiCamera support.

## Features

- **Live PiCamera Feed**: Real-time camera input using Raspberry Pi Camera Module
- **AI-Powered Analysis**: Vegetation segmentation using Roboflow API
- **Wind Drift Compensation**: Real-time calculation of seed drop locations
- **Interactive Controls**: Trackbar-based parameter adjustment
- **Optimized Performance**: Reduced resolution and frame skipping for RPi hardware
- **Threaded Processing**: Separate threads for camera feed and AI inference

## Hardware Requirements

- Raspberry Pi 4 (recommended) or Pi 3B+
- Raspberry Pi Camera Module v2 or v3
- MicroSD card (32GB+ recommended)
- Internet connection for AI inference

## Software Requirements

- Raspberry Pi OS (Bullseye or newer)
- Python 3.9+
- PiCamera2 library
- OpenCV
- Internet connection for Roboflow API

## Installation

### Quick Setup
```bash
# Make setup script executable
chmod +x setup_rpi.sh

# Run setup script
./setup_rpi.sh
```

### Manual Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-opencv python3-picamera2

# Install Python dependencies
pip3 install -r requirements_rpi.txt

# Enable camera interface
sudo raspi-config nonint do_camera 0

# Reboot to apply camera settings
sudo reboot
```

## Configuration

### Camera Settings
The application is configured for optimal RPi performance:
- Resolution: 640x480 (adjustable in code)
- Frame Rate: 15 FPS
- Inference: Every 5th frame processed

### API Configuration
Update the API credentials in `rpi_test.py`:
```python
API_KEY = "your_roboflow_api_key"
MODEL_ENDPOINT = "your_model_endpoint"
```

## Usage

### Running the Application
```bash
python3 rpi_test.py
```

### Controls
- **Live Feed Window**: Shows real-time camera feed
- **Inference Window**: Shows AI analysis with drop predictions
- **Control Panel**: Trackbars for parameter adjustment
  - Wind Speed: 0-20 m/s (×0.1 precision)
  - Wind Direction: 0-360 degrees
  - Altitude: 5-100 meters

### Keyboard Shortcuts
- `q`: Quit application
- `r`: Reset all parameters to defaults

## Performance Optimization

### For Better Performance
1. **Reduce Resolution**: Lower `CAMERA_WIDTH` and `CAMERA_HEIGHT`
2. **Increase Frame Skip**: Higher `INFERENCE_SKIP_FRAMES` value
3. **Lower FPS**: Reduce `CAMERA_FPS` for stability
4. **GPU Memory Split**: Increase GPU memory split to 128MB or 256MB

### Memory Split Configuration
```bash
sudo raspi-config
# Advanced Options > Memory Split > 256
```

### Overclocking (Optional)
```bash
# Add to /boot/config.txt (use with caution)
arm_freq=1750
gpu_freq=600
over_voltage=4
```

## Troubleshooting

### Camera Issues
```bash
# Check camera detection
vcgencmd get_camera

# Test camera manually
libcamera-hello --timeout 5000
```

### Performance Issues
- Ensure adequate power supply (3A+ recommended)
- Use fast MicroSD card (Class 10 or better)
- Close unnecessary applications
- Monitor temperature: `vcgencmd measure_temp`

### Common Errors

**ImportError: No module named 'picamera2'**
```bash
sudo apt install python3-picamera2
```

**Camera not detected**
```bash
sudo raspi-config
# Interface Options > Camera > Enable
sudo reboot
```

**Low FPS or lag**
- Reduce resolution in code
- Increase `INFERENCE_SKIP_FRAMES`
- Check system temperature

## File Structure

```
Drone_Brain/
├── rpi_test.py              # Main RPi application
├── requirements_rpi.txt     # Python dependencies
├── setup_rpi.sh            # Automated setup script
├── README_RPI.md           # This file
└── drone_simulation.py     # Original simulation (for reference)
```

## Technical Details

### Differences from Original
- **Camera Input**: PiCamera2 instead of video file
- **Performance**: Optimized for RPi hardware limitations
- **Threading**: Improved thread management for stability
- **Error Handling**: Enhanced error handling for hardware issues
- **Memory Management**: Reduced memory usage and cleanup

### Processing Pipeline
1. **Camera Capture**: Real-time frame capture from PiCamera
2. **Frame Queue**: Buffered frames for inference processing
3. **AI Inference**: Vegetation segmentation via Roboflow API
4. **Wind Calculation**: Real-time drift compensation
5. **Visualization**: Overlay results on live feed

## API Integration

The system uses Roboflow's segmentation API for vegetation analysis. Ensure you have:
- Valid Roboflow account
- Trained vegetation segmentation model
- Sufficient API credits

## Future Enhancements

- [ ] Local inference using TensorFlow Lite
- [ ] GPS integration for real coordinates
- [ ] Drone control integration
- [ ] Data logging and analytics
- [ ] Mobile app interface

## Support

For issues specific to Raspberry Pi deployment, check:
1. System logs: `journalctl -f`
2. Camera status: `vcgencmd get_camera`
3. Temperature: `vcgencmd measure_temp`
4. Memory usage: `free -h`

## License

This project is part of the Horn-Bill Drone Brain system for precision agriculture applications.
