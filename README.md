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
...

