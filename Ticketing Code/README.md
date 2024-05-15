# Ticketing Queue Monitoring System

## Overview

This Python script is designed to monitor a ticketing queue in a real-time video feed. When a certain number of people crosses the threshold of the ticketing queue, an alert is raised.

## Dependencies

- OpenCV (cv2)
- Ultralytics YOLO library
- NumPy
- PyTorch

## Setup Instructions

1. Ensure Python is installed on your system.
2. Install the required dependencies using pip:
   ```bash
   pip install opencv-python numpy torch torchvision
   pip install 'git+https://github.com/ultralytics/yolov5.git'  # Install Ultralytics YOLO library
