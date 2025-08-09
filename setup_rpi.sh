#!/bin/bash

# Horn-Bill Drone Brain - Raspberry Pi Setup Script
# Run this script on your Raspberry Pi to set up the environment

echo "=========================================="
echo "Horn-Bill Drone Brain - RPi Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-dev python3-setuptools
sudo apt install -y python3-opencv
sudo apt install -y python3-picamera2
sudo apt install -y libatlas-base-dev
sudo apt install -y libjasper-dev
sudo apt install -y libqtgui4
sudo apt install -y libqt4-test
sudo apt install -y libhdf5-dev
sudo apt install -y libhdf5-serial-dev
sudo apt install -y libatlas-base-dev
sudo apt install -y libjasper-dev
sudo apt install -y libqtgui4
sudo apt install -y libqt4-test

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements_rpi.txt

# Enable camera interface
echo "Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Create desktop shortcut
echo "Creating desktop shortcut..."
cat > ~/Desktop/horn-bill-drone.desktop << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Horn-Bill Drone Brain
Comment=Drone vegetation analysis system
Exec=python3 $(pwd)/rpi_test.py
Icon=applications-science
Terminal=true
Categories=Science;Education;
EOF

chmod +x ~/Desktop/horn-bill-drone.desktop

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "To run the application:"
echo "1. python3 rpi_test.py"
echo "2. Or double-click the desktop shortcut"
echo ""
echo "Controls:"
echo "- 'q': Quit application"
echo "- 'r': Reset parameters"
echo "- Use trackbars to adjust parameters"
echo "=========================================="
