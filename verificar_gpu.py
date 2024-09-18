import tensorflow as tf
print("GPUs detectadas:", len(tf.config.experimental.list_physical_devices('GPU')))
