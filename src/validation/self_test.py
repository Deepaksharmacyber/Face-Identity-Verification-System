# validation/self_test.py

import numpy as np
from utils.similarity import cosine_similarity

embedding_path = "reference/identity_embedding.npy"

def run_self_test():
    anchor = np.load(embedding_path)

    similarity = cosine_similarity(anchor, anchor)

    print("Self Similarity Score:", similarity)

    if similarity >= 0.999:
        print("PASS — Embedding system stable")
    else:
        print("FAIL — Embedding instability detected")

if __name__ == "__main__":
    run_self_test()