from deepface.models.facial_recognition.Facenet import FaceNet128dClient
import cv2
import numpy as np

print("Loading FaceNet model...")

model = FaceNet128dClient()

print("FaceNet loaded successfully!")


def get_embedding(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # OpenCV loads images as BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # FaceNet input size
    image = cv2.resize(image, (160, 160))

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Generate embedding
    embedding = model.forward(image)

    return np.array(embedding)


# Image paths
img1 = r"dataset\Khushboo\1.jpg"
img2 = r"dataset\Khushboo\2.jpg"

print("\nGenerating embedding for Image 1...")
embedding1 = get_embedding(img1)

print("Generating embedding for Image 2...")
embedding2 = get_embedding(img2)

print("\n===== EMBEDDING INFORMATION =====")
print("Image 1 shape:", embedding1.shape)
print("Image 2 shape:", embedding2.shape)


# Euclidean distance
euclidean_distance = np.linalg.norm(embedding1 - embedding2)

# Cosine similarity
cosine_similarity = np.dot(embedding1, embedding2) / (
    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
)

print("\n===== FACENET COMPARISON =====")
print("Euclidean Distance:", euclidean_distance)
print("Cosine Similarity:", cosine_similarity)