import os
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from datetime import datetime

DERIVATIONS_DIR = "data/derivations"
ANCHOR_PATH = "reference/identity_embedding.npy"

THRESHOLD = 0.93

print("Loading anchor embedding...")

anchor = np.load(ANCHOR_PATH)

print("Loading face model...")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

images = sorted([f for f in os.listdir(DERIVATIONS_DIR) if f.endswith(".png")])

print("Found", len(images), "images")

for img_name in images:

    path = os.path.join(DERIVATIONS_DIR, img_name)

    img = cv2.imread(path)

    faces = app.get(img)

    if len(faces) == 0:
        print("No face detected:", img_name)
        continue

    embedding = faces[0].embedding

    embedding = embedding / np.linalg.norm(embedding)

    score = float(cosine_similarity(anchor, embedding))

    result = "PASS" if score >= THRESHOLD else "FAIL"

    report = {
        "image_name": img_name,
        "similarity_score": score,
        "threshold": THRESHOLD,
        "result": result,
        "timestamp": datetime.utcnow().isoformat()
    }

    json_name = img_name.replace(".png", "_validation.json")

    json_path = os.path.join(DERIVATIONS_DIR, json_name)

    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"{img_name} → {result} (score {score:.4f})")

print("Validation complete.")