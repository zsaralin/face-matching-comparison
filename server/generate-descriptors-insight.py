import os
import json
from insightface.app import FaceAnalysis
import cv2

OUTPUT_FILE = 'descriptors_insight.ndjson'
ROOT_DIR = '../db/small'  # adjust as needed

def init_model():
    app = FaceAnalysis(name='buffalo_l')  # you can use 'antelopev2' for lightweight
    app.prepare(ctx_id=0, det_size=(256, 256))
    return app

def generate_descriptor(app, image_path):
    print('here')
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        return None
    return faces[0].embedding.tolist()  # return first face only

def process_folder(root, app):
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
            descriptor = generate_descriptor(app, cmp_path)

            if descriptor:
                record = {'key': folder, 'embedding': descriptor}
                output.write(json.dumps(record) + '\n')
                print(f'✅ Processed {folder}')
            else:
                print(f'❌ No face found in {folder}')

if __name__ == '__main__':
    print('🔄 Loading InsightFace models...')
    app = init_model()
    print('🚀 Starting processing...')
    process_folder(ROOT_DIR, app)
    print(f'✅ Done. Output saved to {OUTPUT_FILE}')
