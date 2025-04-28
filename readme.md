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