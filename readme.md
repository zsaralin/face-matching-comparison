# Face Matching Comparison
![0](0.png)
![1](1.png)
![2](2.png)
This project is a full-stack web application designed to compare the performance of different face recognition libraries—**DeepFace**, **InsightFace**, and **FaceAPI.js**—specifically in the context of real-time face matching with webcam input.

## Overview

The application uses a **Vite React** frontend to capture and process webcam video, running real-time face detection and landmark extraction. The backend, built with **Node.js**, performs face recognition and matching using both **DeepFace** (Python) and **FaceAPI.js** (JavaScript) to compare their accuracy and efficiency in practical scenarios.

## Features

- Real-time face detection and landmark drawing using MediaPipe in the frontend.
- Face descriptor generation and matching performed by multiple backends (DeepFace, InsightFace, FaceAPI.js).
- Comparative evaluation of the face matching accuracy and performance.
- Demonstrates that for webcam input, FaceAPI.js outperforms DeepFace in speed and responsiveness despite DeepFace’s reputation for higher accuracy in other contexts.

## Tech Stack

- Frontend: Vite + React, MediaPipe for real-time face detection
- Backend: Node.js with FaceAPI.js and Python integration for DeepFace and InsightFace
- Face Recognition Libraries: DeepFace, InsightFace, FaceAPI.js

## Findings

While DeepFace is widely regarded as a top-tier face recognition library, this project shows that **FaceAPI.js provides better real-time performance with webcam input**, making it more suitable for interactive web applications where responsiveness is critical.

## How to Run

python -m venv venv-insight
python -m venv venv-deepface
# Activate and install InsightFace
.\venv-insight\Scripts\activate
pip install insightface onnxruntime-gpu opencv-python
deactivate

# Activate and install DeepFace
.\venv-deepface\Scripts\activate
pip install "tensorflow<2.14" "opencv-python<4.9"
pip install deepface
deactivate