
import os
import pickle

dataset_path = "cifar-10-batches-py"

required_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
    "batches.meta",
]

# Check that all required files exist
for filename in required_files:
    path = os.path.join(dataset_path, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

print("All required CIFAR-10 files are present.")

# Read one training batch
with open(
    os.path.join(dataset_path, "data_batch_1"), "rb"
) as file:
    batch = pickle.load(file, encoding="bytes")

images = batch[b"data"]
labels = batch[b"labels"]

print("Images in first batch:", images.shape)
print("Labels in first batch:", len(labels))
print("Dataset verification successful!")
