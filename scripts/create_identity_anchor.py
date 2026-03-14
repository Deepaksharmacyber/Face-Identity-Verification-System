import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis

IMAGE_PATH = "data/reference/reference_normalized.png"

print("Loading face model...")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise Exception("Failed to load normalized image")

faces = app.get(img)

if len(faces) == 0:
    raise Exception("No face detected in normalized image")

face = faces[0]

embedding = face.embedding

# normalize embedding
embedding = embedding / np.linalg.norm(embedding)

np.save("data/reference/identity_embedding.npy", embedding)

metadata = {
    "model_name": "InsightFace Buffalo_L",
    "embedding_dimension": int(embedding.shape[0]),
    "normalization": "L2",
    "similarity_threshold": 0.93
}

with open("data/reference/embedding_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Identity embedding created")
print("Embedding dimension:", embedding.shape)