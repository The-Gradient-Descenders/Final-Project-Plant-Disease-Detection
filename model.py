import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import pathlib
import deeplake
import numpy as np

ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')

img = ds.images[0:640].numpy()
label = ds.labels[0:640].numpy(aslist=True)
labels = np.array(label)
labels = labels.flatten()
labels = tf.convert_to_tensor(labels)
img = tf.convert_to_tensor(img)
print("label", label)
print("labels", labels)


model = keras.applications.ResNet152(include_top=False, input_shape=(256, 256, 3))
model.trainable = False
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_outputs = layers.Dense(38, activation="softmax")(base_outputs)

tuned_model = keras.Model(inputs=base_inputs, outputs=final_outputs)


tuned_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

small_train_images = img
small_train_labels = labels

print(small_train_images.shape)
print(small_train_labels.shape)

tuned_model.fit(small_train_images, small_train_labels, epochs=10, verbose=2)