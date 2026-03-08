Face Identity Verification System
Project Overview

This project is a Face Identity Verification System built using Python and InsightFace.

The goal of this project is to check whether two face images belong to the same person or not.

Instead of comparing images directly, the system converts faces into numerical vectors called embeddings and then compares those vectors using cosine similarity.

If the similarity score is higher than a defined threshold, the system says the images belong to the same person. Otherwise, it says they are different people.

This technique is commonly used in:

Face recognition systems

Security verification

Identity authentication

Biometric systems

What I Implemented in This Project

In this project, I built a complete pipeline for face verification.

The main steps are:

Face Detection

Face Embedding Extraction

Embedding Storage

Face Comparison using Cosine Similarity

Identity Verification

How the System Works
Step 1 — Reference Face Embedding

First, we take a reference image of a person (for example Virat Kohli).

The system:

detects the face

extracts features

converts them into a 512-dimensional embedding vector

This embedding is saved as a file:

identity_embedding_viratkohli.npy

This is called the anchor embedding.

Step 2 — Test Image

Next, we give a test image to the system.

Example:

virat_kohli_test.webp

The system again:

detects the face

extracts its embedding

Step 3 — Similarity Calculation

Now the system compares:

reference embedding
vs
test image embedding

using Cosine Similarity.

Cosine similarity measures how close two vectors are.

The score ranges between:

-1 to 1

But in face recognition it usually falls between:

0 to 1

Example:

0.90 → very similar (same person)
0.40 → different person
Step 4 — Identity Decision

We define a threshold value:

threshold = 0.75

If:

similarity >= 0.75

Result:

PASS — Same Identity

If:

similarity < 0.75

Result:

FAIL — Different Identity
Project Structure
identity_system/
│
├── model/                 # Face recognition model files
│
├── reference/             # Stored reference embeddings
│   └── identity_embedding_viratkohli.npy
│
├── validation/            # Image validation scripts
│   └── validate_image.py
│
├── utils/                 # Utility modules
│   ├── embedding.py
│   └── similarity.py
│
├── requirements.txt
├── test_align.py
└── README.md
Technologies Used

This project uses the following technologies:

Python

InsightFace

ONNX Runtime

NumPy

OpenCV

These libraries help in:

detecting faces

extracting facial embeddings

comparing identities

Example Output

When running the verification script:

python -m validation.validate_image validation/virat_kohli_test.webp

Example output:

Similarity Score: 0.83
PASS — Same Identity

Anchor embedding shape: (512,)
Test embedding shape: (512,)
Anchor norm: 21.33
Test norm: 21.01

This means the test image belongs to the same person as the reference image.

Key Concepts Used

This project demonstrates important concepts in Computer Vision and Face Recognition:

Face Detection

Face Embeddings

Cosine Similarity

Identity Verification

Deep Learning Models

Future Improvements

Possible improvements for this project:

Real-time webcam verification

Multi-person database

Face recognition instead of verification

API for identity verification

Web interface for uploading images

Author

Deepak Sharma

This project was built as part of learning Face Recognition and Computer Vision systems using Python.