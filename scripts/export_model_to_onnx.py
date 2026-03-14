import insightface
from insightface.app import FaceAnalysis
import os

MODEL_DIR = "model"
ONNX_OUTPUT = "model/face_embedding_model.onnx"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading InsightFace model...")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

model = app.models["recognition"]

print("Exporting model to ONNX...")

model.export_onnx(ONNX_OUTPUT)

print("Model exported successfully!")
print("Saved at:", ONNX_OUTPUT)