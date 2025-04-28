# match_insight.py
import sys
import json
import base64
import numpy as np
import cv2
import os

# Redirect unwanted stdout to stderr (e.g. InsightFace logs)
import contextlib
import io

NDJSON_PATH = 'descriptors_insight.ndjson'
DB_PREFIX = 'db/small'

# Completely silence InsightFace stdout logs
with contextlib.redirect_stdout(io.StringIO()):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(256, 256))

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

        # ✅ Write to debug file just before InsightFace call
        debug_path = "debug_crash.jpg"
        cv2.imwrite(debug_path, img)
        print(f"💾 Saved debug image to {debug_path}", file=sys.stderr)

        # ✅ Sanity check: Is image shape valid?
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            raise ValueError(f"Image too small ({w}x{h})")

        faces = app.get(img)  # <-- likely crash point
        if not faces:
            print("⚠️ No faces found", file=sys.stderr)
            return None

        return faces[0].embedding.tolist()

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
    results.sort(key=lambda x: -x['similarity'])  # Use - for descending
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
            print(json.dumps(matches))  # Final result to stdout
        else:
            print("⚠️ No face detected", file=sys.stderr)
            print(json.dumps([]))
    except Exception as e:
        print(f"Exception in main: {e}", file=sys.stderr)
        print(json.dumps([]))

if __name__ == '__main__':
    main()
