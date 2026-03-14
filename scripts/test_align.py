import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import INPUT_DIR, REFERENCE_DIR
import cv2
import numpy as np

from config import REFERENCE_DIR, DERIVATIONS_DIR

import cv2
import insightface
from insightface.app import FaceAnalysis
import os

# INPUT_IMAGE = "reference/viratkohli.avif"
# OUTPUT_IMAGE = "reference/reference_normalized.png"

INPUT_IMAGE = f"{INPUT_DIR}/viratkohli.avif"
OUTPUT_IMAGE = f"{REFERENCE_DIR}/reference_normalized.png"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

img = cv2.imread(INPUT_IMAGE)

faces = app.get(img)

if len(faces) == 0:
    raise Exception("No face detected")

face = faces[0]

# aligned face
aligned = face.normed_embedding  # not image
bbox = face.bbox.astype(int)

x1,y1,x2,y2 = bbox

# expand bounding box
margin = 0.35

h, w = img.shape[:2]

dx = int((x2-x1)*margin)
dy = int((y2-y1)*margin)

x1 = max(0, x1-dx)
y1 = max(0, y1-dy)
x2 = min(w, x2+dx)
y2 = min(h, y2+dy)

face_crop = img[y1:y2, x1:x2]

# resize to required resolution
target_size = 2048

h2, w2 = face_crop.shape[:2]

scale = target_size / max(h2, w2)

new_w = int(w2*scale)
new_h = int(h2*scale)

face_crop = cv2.resize(face_crop, (new_w,new_h))

os.makedirs("reference", exist_ok=True)

cv2.imwrite(OUTPUT_IMAGE, face_crop)

print("Normalized image saved:", OUTPUT_IMAGE)
print("Resolution:", new_w, "x", new_h)