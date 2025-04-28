# match_deepface.py
import sys
import json
import base64
import numpy as np
import cv2
import os

# Silence TensorFlow/DeepFace logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = fatal only

NDJSON_PATH = 'descriptors_facenet.ndjson'
DB_PREFIX = 'db/small'

from deepface import DeepFace

def get_descriptor(base64_image):
    try:
        if not base64_image:
            raise ValueError("Empty image data")

        if ',' in base64_image:
            base64_data = base64_image.split(',')[1]
        else:
            base64_data = base64_image

        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        if np_arr.size == 0:
            raise ValueError("Decoded image bytes are empty")

        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 could not decode image")

        # ✅ Save debug image
        debug_path = "debug_crash.jpg"
        cv2.imwrite(debug_path, img)
        print(f"💾 Saved debug image to {debug_path}", file=sys.stderr)

        h, w = img.shape[:2]
        if h < 20 or w < 20:
            raise ValueError(f"Image too small ({w}x{h})")

        # 🟠 Extract FaceNet embeddings via DeepFace
        obj = DeepFace.represent(img_path=img, model_name="Facenet", enforce_detection=False)

        if not obj or len(obj) == 0:
            print("⚠️ No face embedding found", file=sys.stderr)
            return None

        return obj[0]["embedding"]

    except Exception as e:
        print(f"❌ Error in get_descriptor: {e}", file=sys.stderr)
        return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)

def compare_descriptor(input_descriptor, top_n=26):
    results = []
    with open(NDJSON_PATH, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj['key']
            embedding = obj['embedding']
            distance = cosine_similarity(input_descriptor, embedding)
            file_path = os.path.join(DB_PREFIX, key, 'images', f'{key}_cmp.png').replace('\\', '/')
            results.append({'key': key, 'file': file_path, 'similarity': distance})
    results.sort(key=lambda x: -x['similarity'])  # Descending similarity
    return results[:top_n]

def main():
    try:
        if len(sys.argv) < 2:
            print("❌ No JSON file path provided", file=sys.stderr)
            print(json.dumps([]))
            return

        json_path = sys.argv[1]
        with open(json_path, 'r', encoding='utf-8') as f:
            payload = f.read()

        print(f"📥 Loaded payload from {json_path}", file=sys.stderr)

        data = json.loads(payload)
        image = data.get("imageData")
        print(f"📏 Base64 string length: {len(image)}", file=sys.stderr)

        descriptor = get_descriptor(image)
        if descriptor:
            print("✅ Descriptor generated", file=sys.stderr)
            matches = compare_descriptor(descriptor)
            print(f"🔎 Found {len(matches)} matches", file=sys.stderr)
            print(json.dumps(matches))
        else:
            print("⚠️ No face detected", file=sys.stderr)
            print(json.dumps([]))
    except Exception as e:
        print(f"Exception in main: {e}", file=sys.stderr)
        print(json.dumps([]))

if __name__ == '__main__':
    main()
