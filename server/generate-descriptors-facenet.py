import os
import json
from deepface import DeepFace
import cv2
import numpy as np

OUTPUT_FILE = 'descriptors_facenet.ndjson'
ROOT_DIR = '../db/small'  # adjust as needed

def init_model():
    print('🔄 Loading DeepFace Facenet model...')
    # This triggers the model download and preparation internally
    model = DeepFace.build_model("Facenet")
    return model

def generate_descriptor(model, image_path):
    print('📸 Processing:', image_path)
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image {image_path}")
        return None

    # Convert BGR to RGB (DeepFace expects RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        # Only pass model_name and input image — not model directly
        embedding_objs = DeepFace.represent(
            img_path=img_rgb,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )
        return embedding_objs[0]["embedding"]
    except Exception as e:
        print(f"❌ DeepFace error: {e}")
        return None


def process_folder(root, model):
    with open(OUTPUT_FILE, 'w') as output:
        for folder in os.listdir(root):
            folder_path = os.path.join(root, folder, 'images')
            if not os.path.isdir(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.endswith('cmp.png')]
            if not files:
                print(f'⚠️  No cmp.png in {folder}')
                continue

            cmp_path = os.path.join(folder_path, files[0])
            descriptor = generate_descriptor(model, cmp_path)

            if descriptor is not None:
                record = {'key': folder, 'embedding': descriptor}
                output.write(json.dumps(record) + '\n')
                print(f'✅ Processed {folder}')
            else:
                print(f'❌ No face found in {folder}')

if __name__ == '__main__':
    model = init_model()
    print('🚀 Starting processing...')
    process_folder(ROOT_DIR, model)
    print(f'✅ Done. Output saved to {OUTPUT_FILE}')
