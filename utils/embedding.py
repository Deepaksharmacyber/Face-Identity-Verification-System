# # utils/embedding.py

# import numpy as np
# import cv2
# from insightface.app import FaceAnalysis

# app = FaceAnalysis()
# app.prepare(ctx_id=0)
# # 0 → GPU
# # -1 → CPU

# def extract_embedding(image_path):
#     img = cv2.imread(image_path)
#     faces = app.get(img)

#     if len(faces) == 0:
#         print("No face detected.")
#         return None

#     embedding = faces[0].embedding

#     # L2 normalize
#     norm = np.linalg.norm(embedding)
#     embedding = embedding / norm
    
#     # normalize
#     embedding = embedding / np.linalg.norm(embedding)
    
#     return embedding


# utils/embedding.py

import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    print("Faces detected:", len(faces))

    if len(faces) == 0:
        print("No face detected.")
        return None

    embedding = faces[0].embedding

    # L2 normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding