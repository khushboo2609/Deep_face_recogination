import cv2
import numpy as np
import os
from itertools import combinations

dataset_path = r"dataset\Khushboo"

images = {}

# Load all 50 images
for i in range(1, 51):

    filename = f"{i}.jpg"
    path = os.path.join(dataset_path, filename)

    image = cv2.imread(path)

    if image is not None:
        images[filename] = image

print(f"Images loaded: {len(images)}")

print("\n===== DUPLICATE IMAGE CHECK =====")

duplicates = []

for img1, img2 in combinations(images.keys(), 2):

    image1 = images[img1]
    image2 = images[img2]

    if image1.shape == image2.shape:

        identical = np.array_equal(image1, image2)

        if identical:
            duplicates.append((img1, img2))
            print(f"Duplicate: {img1} == {img2}")

print("\n===== SUMMARY =====")

print("Total duplicate pairs:", len(duplicates))

if len(duplicates) == 0:
    print("No duplicate images found.")
else:
    print("\nDuplicate pairs:")
    for pair in duplicates:
        print(pair)