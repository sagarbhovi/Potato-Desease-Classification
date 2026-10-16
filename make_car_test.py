
import pickle
import numpy as np
from PIL import Image

# Path to the extracted CIFAR-10 dataset
file_path = "cifar-10-batches-py/test_batch"

# Load the test dataset
with open(file_path, "rb") as file:
    data = pickle.load(file, encoding="bytes")

images = data[b"data"]
labels = data[b"labels"]

# CIFAR-10 label 1 represents a car (automobile)
car_index = labels.index(1)

# Extract the car image
car_image = images[car_index]

# Convert the image from array format to RGB format
car_image = car_image.reshape(3, 32, 32).transpose(1, 2, 0)
car_image = car_image.astype(np.uint8)

# Save the image in your project folder
Image.fromarray(car_image).save("cifar_car_test.png")

print("Car test image saved as cifar_car_test.png")
