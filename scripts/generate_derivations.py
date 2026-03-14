import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import REFERENCE_DIR, DERIVATIONS_DIR

INPUT_IMAGE = f"{REFERENCE_DIR}/reference_normalized.png"
OUTPUT_DIR = DERIVATIONS_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(INPUT_IMAGE)

if img is None:
    raise Exception("Failed to load normalized image")

h, w = img.shape[:2]

print("Generating identity derivations...")

for i in range(16):

    new_img = img.copy()

    # slight rotation
    angle = np.random.uniform(-2, 2)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    new_img = cv2.warpAffine(new_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # brightness change
    brightness = np.random.uniform(0.95, 1.05)
    new_img = np.clip(new_img * brightness, 0, 255).astype(np.uint8)

    output_path = f"{OUTPUT_DIR}/img_{i+1:02d}.png"

    cv2.imwrite(output_path, new_img)

    print("Created:", output_path)

print("16 derivations generated successfully.")
