from load_data import Database
import numpy as np
db = Database()
from collections import Counter

def loading_data():
    
    train_images = np.memmap("./village_train_img.dat", dtype="uint8", mode="w+", shape=(40727, 256, 256, 3))
    train_data_images = db.load("./data/train_images_plantVillage.pkl")
    train_images = train_data_images
    del train_data_images
    
    test_images = np.memmap("./village_test_img.dat", dtype="uint8", mode="w+", shape=(13576, 256, 256, 3))
    test_data_images = db.load("./data/test_images_plantVillage.pkl")
    test_images = test_data_images
    del test_data_images
    
    train_labels = np.memmap("./village_train_labels.dat", dtype="uint8", mode="w+", shape=(40727,))
    train_data_labels = db.load("./data/train_labels_plantVillage.pkl")
    train_labels = train_data_labels

    test_labels = np.memmap("./village_test_labels.dat", dtype="uint8", mode="w+", shape=(13576,))
    test_data_labels = db.load("./data/test_labels_plantVillage.pkl")
    test_labels = test_data_labels
    del test_data_labels
    # print(Counter(train_data_labels))
    return

# def preprocessing():
#     train_images = np.memmap("./village_train_img.dat", dtype="uint8", mode="w+", shape=(42601, 256, 256, 3))
#     data = db.load("./data/train_images_plantVillage.pkl")
#     train_images[:data.shape[0]] = data
#     del data
#     data2 = db.load("./data/train_images_plantDocs.pkl")
#     train_images[40727:] = data2.astype("uint8")
#     del data2
#     train_images = np.memmap("./village_test_img.dat", dtype="uint8", mode="w+", shape=(42601, 256, 256, 3))
#     data = db.load("./data/test_images_plantVillage.pkl")
#     train_images[:data.shape[0]] = data
#     del data
#     data2 = db.load("./data/train_images_plantDocs.pkl")
#     train_images[40727:] = data2.astype("uint8")
#     del data2
    