# reference/create_identity_anchor.py

import numpy as np
import os
from utils.embedding import extract_embedding

# Paths
input_image = "reference/viratkohli.avif"
output_embedding_path = "reference/identity_embedding_viratkohli.npy"

def create_identity_anchor():
    print("Extracting embedding...")

    embedding = extract_embedding(input_image)

    if embedding is None:
        print("Embedding failed.")
        return

    np.save(output_embedding_path, embedding)

    print("Identity embedding saved at:", output_embedding_path)

if __name__ == "__main__":
    create_identity_anchor()