import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import pathlib
import deeplake
import numpy as np
from sklearn.utils import shuffle

print("Loading dataset...")
ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')

print("Dataset loaded...")

# idxs = np.arange(len(ds.labels))
# np.random.shuffle(idxs)

# small_batch = int(idxs[:100])
# print(small_batch)

print("Converting to numpy arrays...")
img = ds.images.numpy()
labels = ds.labels.numpy()

print("Shuffling data...")
img, labels = shuffle(img, labels)

img = img[0:100]

labels = labels[0:100]
# batch_labels = []
# for i in small_batch:
#     batch_labels += [label[i]]

labels = labels.flatten()
labels = tf.convert_to_tensor(labels)
img = tf.convert_to_tensor(img)
# print("label", label)
print("labels", labels)



model = keras.applications.ResNet152(include_top=False, input_shape=(256, 256, 3))


tuned_model = models.Sequential([
    model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(39)
])

for layer in model.layers:
    layer.trainable = False

tuned_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

small_train_images = img
small_train_labels = labels

print(small_train_images.shape)
print(small_train_labels.shape)

history = tuned_model.fit(small_train_images, small_train_labels, epochs=5, verbose=2)

