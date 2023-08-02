import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import pathlib
import deeplake
import numpy as np

ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')

img = ds.images[0:1000].numpy()
label = ds.labels[0:1000].numpy(aslist=True)
labels = np.array(label)
labels = labels.flatten()
labels = tf.convert_to_tensor(labels)
img = tf.convert_to_tensor(img)
print("label", label)
print("labels", labels)



model = keras.applications.ResNet152(include_top=False, input_shape=(256, 256, 3))


tuned_model = models.Sequential([
    model,
    layers.GlobalAveragePooling2D(),
    #layers.Dense(1024, activation='relu'),
    layers.Dense(38)
])

for layer in model.layers:
    layer.trainable = False

tuned_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

small_train_images = img[0:10]
small_train_labels = labels[0:10]

print(small_train_images.shape)
print(small_train_labels.shape)

tuned_model.fit(small_train_images, small_train_labels, epochs=10, verbose=2)