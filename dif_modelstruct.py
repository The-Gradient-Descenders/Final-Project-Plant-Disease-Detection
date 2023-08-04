import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_hub as hub
import pathlib
import deeplake
import numpy as np
from sklearn.utils import shuffle
from load_data import Database
import time
import processing as p
from collections import Counter
import math

# print("Loading dataset...")
# ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')
start = time.time()



train_data_images = np.memmap("./village_train_img.dat", mode="r", shape=(40727, 256, 256, 3))
test_data_images = np.memmap("./village_test_img.dat", mode="r", shape=(13576, 256, 256, 3))
train_data_labels = np.memmap("./village_train_labels.dat", mode="r", shape=(40727,))
test_data_labels = np.memmap("./village_test_labels.dat", mode="r", shape=(13576,))

print("Finished loading data.")

# train_dataset = tf.data.Dataset.from_tensor_slices((train_data_images, train_data_labels))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_data_images, test_data_labels))

# del train_data_images
# del test_data_images
# del train_data_labels
# del test_data_labels

# BATCH_SIZE = 3
# SHUFFLE_BUFFER_SIZE = 100

# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# print("Shuffled and batched datasets.")

# print("Shuffling data...")
img, labels = shuffle(train_data_images, train_data_labels)



# img = img[0:10000]

# labels = labels[0:10000]

# labels = tf.convert_to_tensor(labels)
# img = tf.convert_to_tensor(img)
#print("labels", labels)

base_model = keras.applications.ResNet50(include_top=False, input_shape=(256, 256, 3))
#instructor suggesion: maybe try downgrading the model from resnet152 to resnet50?  the second one's smaller
#also that additional dense layer could be causing trouble if it's too large

base_model.trainable = False
# for layer in model.layers:
#     layer.trainable = False



tuned_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    # layers.Dense(1024, activation='relu'),
    layers.Dense(39)
]) # see https://www.tensorflow.org/guide/keras/transfer_learning, this part might be wrong

print(tuned_model.summary())

tuned_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print("Model compiled.")
# small_train_images = img
# small_train_labels = labels

# print(small_train_images.shape)
# print(small_train_labels.shape)



# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class PlantVillageSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.x.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        low = idx * self.batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.x.shape[0])
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        return batch_x, batch_y

pvs = PlantVillageSequence(img, labels, 10)


print("Training model...")
tuned_model.fit(pvs, epochs=5, verbose=1)

tuned_model.evaluate(test_data_images, test_data_labels)

end = time.time()
print("The execution time of the program is ", (end-start), " seconds")