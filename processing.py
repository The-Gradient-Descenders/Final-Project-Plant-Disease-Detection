from load_data import Database
import numpy as np
db = Database()
from collections import Counter
import deeplake
from sklearn.utils import shuffle

def loading_data():
    
    train_images = np.memmap("./village_train_img.dat", dtype="uint8", mode="w+", shape=(42243, 256, 256, 3))
    train_data_images = db.load("./data2/train_images_plantVillage.pkl")
    train_images = train_data_images
    del train_data_images
    
    test_images = np.memmap("./village_test_img.dat", dtype="uint8", mode="w+", shape=(5280, 256, 256, 3))
    test_data_images = db.load("./data2/test_images_plantVillage.pkl")
    test_images = test_data_images
    del test_data_images
    
    val_images = np.memmap("./village_val_images.dat", dtype="uint8", mode="w+", shape=(5280, 256, 256, 3))
    val_data_images = db.load("./data2/validation_images_plantVillage.pkl")
    val_images = val_data_images
    del val_data_images

    train_labels = np.memmap("./village_train_labels.dat", dtype="uint8", mode="w+", shape=(42243,))
    train_data_labels = db.load("./data2/train_labels_plantVillage.pkl")
    train_labels = train_data_labels

    test_labels = np.memmap("./village_test_labels.dat", dtype="uint8", mode="w+", shape=(5280,))
    test_data_labels = db.load("./data2/test_labels_plantVillage.pkl")
    test_labels = test_data_labels
    del test_data_labels

    val_labels = np.memmap("./village_val_labels.dat", dtype="uint8", mode="w+", shape=(5280,))
    val_data_labels = db.load("./data2/validation_labels_plantVillage.pkl")
    val_labels = val_data_labels
    del val_data_labels

    # print(Counter(train_data_labels))
    return

def load_data():
    
    ds = deeplake.load('hub://activeloop/plantvillage-without-augmentation')
    imgs = ds.images.numpy()   
    labels = ds.labels.numpy()
    labels = labels.flatten()
    print(imgs.shape)
    print(labels.shape)

    print("Data Loaded")

    print("Shuffling Data")

    imgs, labels = shuffle(imgs, labels)

    print(imgs.shape)
    print(labels.shape)

    print("Shuffled Data")
    
    return imgs[0:42243], labels[0:42243], imgs[42243:47523], labels[42243:47523], imgs[47523:52803], labels[47523:52803]


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
    