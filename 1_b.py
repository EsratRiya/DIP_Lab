import numpy as np
import cv2
import matplotlib.pyplot as plt

# Display the images
plt.figure(figsize=(12, 6))

def reduce_intensity_levels(image, bits):

    i = 1
    
    for b in range(bits, 0, -1):
        # Calculate the factor to reduce intensity levels
        max_value = 2 ** b - 1
        reduced_image = (image * (max_value / 255)).astype(np.uint8)
        plt.subplot(2,4,i)
        i = i + 1
        plt.imshow(reduced_image, cmap='gray')
        plt.title(f"Bits: {b}")
        plt.axis('off')



# Load a 512x512 grayscale image
image_path = "img.jpg"  # Replace with your image path
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



if original_image.shape != (512, 512):
    original_image = cv2.resize(original_image, (512, 512), interpolation=cv2.INTER_AREA)

# Reduce intensity levels iteratively
num_bits = 8  # Reduce intensity level by 1 bit at a time down to binary
reduce_intensity_levels(original_image, num_bits)


plt.tight_layout()
plt.show()
