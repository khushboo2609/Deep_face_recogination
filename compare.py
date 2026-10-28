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
# 2. DATASET PATH
# ==========================================

dataset_path = r"dataset\Khushboo"


# ==========================================
# 3. GET IMAGE FILES
# ==========================================

image_files = [
    os.path.join(dataset_path, f"{i}.jpg")
    for i in range(1, 51)
]

print(f"\nImages found: {len(image_files)}")


# ==========================================
# 4. FUNCTION TO GENERATE EMBEDDING
# ==========================================

def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"WARNING: Could not load {image_path}")
        return None

    # OpenCV uses BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # FaceNet input size
    image = cv2.resize(image, (160, 160))

    # Normalize
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Generate embedding
    embedding = model.forward(image)

    return np.array(embedding)


# ==========================================
# 5. GENERATE ALL EMBEDDINGS
# ==========================================

embeddings = {}

print("\nGenerating embeddings...")

for image_path in image_files:

    filename = os.path.basename(image_path)

    embedding = get_embedding(image_path)

    if embedding is not None:
        embeddings[filename] = embedding

        print(f"Processed: {filename}")


print(f"\nSuccessfully generated embeddings: {len(embeddings)}")


# ==========================================
# 6. COMPARE EVERY IMAGE PAIR
# ==========================================

results = []

image_names = list(embeddings.keys())

print("\nComparing image pairs...")

for img1, img2 in combinations(image_names, 2):

    embedding1 = embeddings[img1]
    embedding2 = embeddings[img2]

    # Euclidean distance
    euclidean_distance = np.linalg.norm(
        embedding1 - embedding2
    )

    # Cosine similarity
    cosine_similarity = np.dot(
        embedding1,
        embedding2
    ) / (
        np.linalg.norm(embedding1)
        * np.linalg.norm(embedding2)
    )

    results.append({
        "image1": img1,
        "image2": img2,
        "euclidean_distance": euclidean_distance,
        "cosine_similarity": cosine_similarity
    })


# ==========================================
# 7. CONVERT RESULTS TO NUMPY ARRAYS
# ==========================================

distances = np.array([
    result["euclidean_distance"]
    for result in results
])

similarities = np.array([
    result["cosine_similarity"]
    for result in results
])


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print("       FACENET DATASET RESULTS")
print("========================================")

print(f"Total images: {len(embeddings)}")
print(f"Total pairs compared: {len(results)}")

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
# 9. MOST SIMILAR PAIR
# ==========================================

most_similar = max(
    results,
    key=lambda x: x["cosine_similarity"]
)

print("\n----- Most Similar Pair -----")

print("Image 1:", most_similar["image1"])
print("Image 2:", most_similar["image2"])
print(
    "Cosine Similarity:",
    f"{most_similar['cosine_similarity']:.6f}"
)


# ==========================================
# 10. LEAST SIMILAR PAIR
# ==========================================

least_similar = min(
    results,
    key=lambda x: x["cosine_similarity"]
)

print("\n----- Least Similar Pair -----")

print("Image 1:", least_similar["image1"])
print("Image 2:", least_similar["image2"])
print(
    "Cosine Similarity:",
    f"{least_similar['cosine_similarity']:.6f}"
)