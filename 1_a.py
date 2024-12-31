import numpy as np
import cv2
import matplotlib.pyplot as plt

def reduce_resolution(image, iterations):
   
    images = [image]
    for _ in range(iterations):
        # Reduce the resolution by half
        height, width = images[-1].shape
        resized = cv2.resize(images[-1], (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        images.append(resized)
    return images

# Load a 512x512 grayscale image
image_path = "img.jpg"  # Replace with your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if original_image.shape != (512, 512):
    original_image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_AREA)

# Reduce resolution iteratively
num_iterations = 4  # Number of resolution reductions
reduced_images = reduce_resolution(original_image, num_iterations)

# Display the images
plt.figure(figsize=(15, 10))
for i, img in enumerate(reduced_images):
    plt.subplot(1, len(reduced_images), i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Resolution {img.shape[1]}x{img.shape[0]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
