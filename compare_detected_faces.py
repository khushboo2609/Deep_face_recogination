from deepface.models.facial_recognition.Facenet import FaceNet128dClient
import cv2
import numpy as np
import os
from itertools import combinations


# ==========================================
# 1. LOAD FACENET
# ==========================================

print("Loading FaceNet model...")

model = FaceNet128dClient()

print("FaceNet loaded successfully!")


# ==========================================
# 2. LOAD HAAR FACE DETECTOR
# ==========================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

print("Haar face detector loaded!")


# ==========================================
# 3. DATASET
# ==========================================

dataset_path = r"dataset\Khushboo"


# ==========================================
# 4. DETECT FACE + GENERATE EMBEDDING
# ==========================================

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"WARNING: Could not load {image_path}")
        return None

    # Convert to grayscale for Haar detector
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print(f"WARNING: No face detected in {image_path}")
        return None

    # Select largest detected face
    x, y, w, h = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    # Crop face
    face = image[y:y+h, x:x+w]

    # Convert BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize for FaceNet
    face = cv2.resize(face, (160, 160))

    # Normalize
    face = face.astype("float32") / 255.0

    # Add batch dimension
    face = np.expand_dims(face, axis=0)

    # Generate embedding
    embedding = model.forward(face)

    return np.array(embedding)


# ==========================================
# 5. PROCESS UNIQUE IMAGES ONLY
# ==========================================

duplicate_images = {
    "18.jpg",
    "31.jpg",
    "45.jpg"
}

image_files = [
    os.path.join(dataset_path, f"{i}.jpg")
    for i in range(1, 51)
    if f"{i}.jpg" not in duplicate_images
]

print(f"\nUnique images used: {len(image_files)}")


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

embeddings = {}

print("\nGenerating face embeddings...")

for image_path in image_files:

    filename = os.path.basename(image_path)

    embedding = get_embedding(image_path)

    if embedding is not None:

        embeddings[filename] = embedding

        print(f"Processed: {filename}")


print(
    f"\nSuccessfully generated embeddings: "
    f"{len(embeddings)}"
)


# ==========================================
# 7. COMPARE PAIRS
# ==========================================

results = []

image_names = list(embeddings.keys())

print("\nComparing image pairs...")

for img1, img2 in combinations(image_names, 2):

    embedding1 = embeddings[img1]
    embedding2 = embeddings[img2]

    # Euclidean distance
    distance = np.linalg.norm(
        embedding1 - embedding2
    )

    # Cosine similarity
    similarity = np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        * np.linalg.norm(embedding2)
    )

    results.append({
        "image1": img1,
        "image2": img2,
        "distance": distance,
        "similarity": similarity
    })


# ==========================================
# 8. STATISTICS
# ==========================================

distances = np.array([
    r["distance"] for r in results
])

similarities = np.array([
    r["similarity"] for r in results
])


print("\n========================================")
print("   FACENET + HAAR RESULTS")
print("========================================")

print(f"Images used: {len(embeddings)}")
print(f"Pairs compared: {len(results)}")

print("\n----- Euclidean Distance -----")

print(f"Minimum : {distances.min():.6f}")
print(f"Maximum : {distances.max():.6f}")
print(f"Average : {distances.mean():.6f}")
print(f"Std Dev : {distances.std():.6f}")

print("\n----- Cosine Similarity -----")

print(f"Minimum : {similarities.min():.6f}")
print(f"Maximum : {similarities.max():.6f}")
print(f"Average : {similarities.mean():.6f}")
print(f"Std Dev : {similarities.std():.6f}")


# ==========================================
# 9. MOST SIMILAR
# ==========================================

most_similar = max(
    results,
    key=lambda x: x["similarity"]
)

print("\n----- Most Similar Pair -----")

print("Image 1:", most_similar["image1"])
print("Image 2:", most_similar["image2"])
print(
    "Cosine Similarity:",
    f"{most_similar['similarity']:.6f}"
)


# ==========================================
# 10. LEAST SIMILAR
# ==========================================

least_similar = min(
    results,
    key=lambda x: x["similarity"]
)

print("\n----- Least Similar Pair -----")

print("Image 1:", least_similar["image1"])
print("Image 2:", least_similar["image2"])
print(
    "Cosine Similarity:",
    f"{least_similar['similarity']:.6f}"
)