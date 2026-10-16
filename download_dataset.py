
import tensorflow as tf

print("Starting CIFAR-10 download...")

try:
    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.cifar10.load_data()
    )

    print("Download successful!")
    print("Training images:", x_train.shape)
    print("Testing images:", x_test.shape)

except Exception as e:
    print("Download failed:", e)
