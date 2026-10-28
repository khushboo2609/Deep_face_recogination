from deepface.models.facial_recognition.Facenet import FaceNet128dClient
import cv2
import numpy as np

print("Loading FaceNet model...")

model = FaceNet128dClient()

print("FaceNet loaded successfully!")

image_path = r"dataset\Khushboo\1.jpg"

image = cv2.imread(image_path)

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to FaceNet input size
image = cv2.resize(image, (160, 160))

# Normalize
image = image.astype("float32") / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

print("Generating embedding...")

embedding = model.forward(image)

# Convert list to NumPy array
embedding = np.array(embedding)

print("Embedding generated successfully!")
print("Embedding type:", type(embedding))
print("Embedding shape:", embedding.shape)
print("Embedding length:", len(embedding))

print("\nFirst 10 embedding values:")
print(embedding[:10])