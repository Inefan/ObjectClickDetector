# ObjectClickDetector

C++ application using LibTorch + OpenCV.
Detects objects with YOLOv5 and allows clicking on them to get class name and confidence.

## Features
- YOLOv5 TorchScript inference
- Mouse click → object identification
- Class names loaded from file
- Real-time webcam support

## Tech stack
- C++
- LibTorch
- OpenCV

## How it works
1. Frame preprocessing
2. YOLO inference
3. NMS
4. Mouse hit-test inside bounding boxes

## Build
### Requirements
- Windows 10+
- C++17
- OpenCV
- LibTorch (CPU)

### Build steps
1. Clone repository
2. Configure OpenCV and LibTorch paths
3. Build with Visual Studio (x64 Release)
4. Place model and classes file in `model/` directory

### Run
- Image mode: load single image and click on detected objects

