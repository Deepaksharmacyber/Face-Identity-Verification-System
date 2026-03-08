# validation/validate_image.py

import sys
import numpy as np
from utils.embedding import extract_embedding
from utils.similarity import cosine_similarity

anchor_path = "reference/identity_embedding_viratkohli.npy"
threshold = 0.75


def validate_image(image_path):

    # load anchor
    anchor_embedding = np.load(anchor_path)

    # extract embedding of input image
    test_embedding = extract_embedding(image_path)

    if test_embedding is None:
        print("Face not detected.")
        return

    # compute similarity
    similarity = cosine_similarity(anchor_embedding, test_embedding)

    print("Similarity Score:", similarity)

    if similarity >= threshold:
        print("PASS — Same Identity")
    else:
        print("FAIL — Different Identity")

    print("Anchor embedding shape:", anchor_embedding.shape)
    print("Test embedding shape:", test_embedding.shape)

    print("Anchor norm:", np.linalg.norm(anchor_embedding))
    print("Test norm:", np.linalg.norm(test_embedding))


if __name__ == "__main__":
    if len(sys.argv) < 2 :
        print("Usage: python -m validation.validate_image <image_path>")
        exit()

    image_path = sys.argv[1]
    validate_image(image_path)
    # image_path = "validation/virat_kohli_test.webp"
    # validate_image(image_path)

#when you will run this file you need to run using file name 
# python3 -m validation.validate_image validation/virat_kohli_test.webp