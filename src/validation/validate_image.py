import sys
import numpy as np

from src.utils.embedding import extract_embedding
from src.utils.similarity import cosine_similarity

THRESHOLD = 0.93


def validate_image(reference_image, test_image):

    print("\nExtracting reference embedding...")
    anchor_embedding = extract_embedding(reference_image)

    if anchor_embedding is None:
        print("❌ No face detected in reference image")
        return

    print("Extracting test image embedding...")
    test_embedding = extract_embedding(test_image)

    if test_embedding is None:
        print("❌ No face detected in test image")
        return

    similarity = cosine_similarity(anchor_embedding, test_embedding)

    print("\nSimilarity Score:", similarity)

    if similarity >= THRESHOLD:
        print("\n✅ RESULT: SAME PERSON")
    else:
        print("\n❌ RESULT: DIFFERENT PERSON")


if __name__ == "__main__":

    reference_image = sys.argv[1]
    test_image = sys.argv[2]

    validate_image(reference_image, test_image)