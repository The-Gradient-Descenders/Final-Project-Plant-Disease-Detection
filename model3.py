import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow_hub as hub
import pathlib
import deeplake
import numpy as np
from sklearn.utils import shuffle
from load_data import Database
import time
import processing as p
# print("Loading dataset...")
# ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')
start = time.time()

test_data_images, train_data_images, test_data_labels, train_data_labels = p.loading_data()

print("Finished loading data.")

train_dataset = tf.data.Dataset.from_tensor_slices((train_data_images, train_data_labels))


test_dataset = tf.data.Dataset.from_tensor_slices((test_data_images, test_data_labels))

lr = 0.001
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = len(train_data_images)

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
print("Shuffled and batched datasets.")

# add a classfier on top of a base model as our first model

#instructor suggesion: maybe try downgrading the model from resnet152 to resnet50?  the second one's smaller
#also that additional dense layer could be causing trouble if it's too large
base_model = keras.applications.ResNet152(include_top=False, input_shape=(256, 256, 3))

base_model.trainable = False

inputs = keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False) # we must specify training=False so that BatchNormalization layers stay in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(100, activation='relu')(x)
outputs = layers.Dense(39)(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("Model compiled.")

initial_epochs = 10
history = model.fit(train_dataset, epochs=initial_epochs, verbose=2)

model.evaluate(test_dataset)

# now we un-freeze the top layers of the base model and do additional training
print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True

fine_tune_at = 100 # change this number

# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False

# model.compile(
#     optimizer = keras.optimizers.Adam(learning_rate = lr/10),
#     loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics = ["accuracy"],
# )

# fine_tune_epochs = 10
# total_epochs = initial_epochs + fine_tune_epochs
# history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch = history.epoch[-1], verbose=2)

# model.evaluate(test_dataset)

end = time.time()
print("The execution time of the program is ", (end-start), " seconds")