# to get a cli tool
import click
# essentials
import numpy as np
import requests
import cv2
# For downloading and extracting dataset
from urllib.request import urlretrieve
import os
import tarfile
# for splitting into training & validation sets
from sklearn.model_selection import train_test_split
# for making labels one-hot encoded
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# for CNN and NN models
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, Dropout, Activation, Dense, MaxPooling2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
# For transfer learning
from keras.applications.inception_v3 import InceptionV3
# for saving model
import json
# for pretty printing
import pprint
# for sorting dictionary by value
import operator


URL_CALTECH_101_DATA = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz'

########################################
# Download & Extract dataset
########################################

def download_dataset(url):
    current_directory = os.path.dirname(os.path.realpath('__file__'))
    dataset_file_path = current_directory+"/dataset.tgz"
    if os.path.exists(dataset_file_path):
        print("Already downloaded.")
    else:
        filename, headers = urlretrieve(url, dataset_file_path)
    print("Done")

def extract_dataset(dataset_file_path, extraction_directory):
    if (not os.path.exists(extraction_directory)):
        os.makedirs(extraction_directory)
    if (dataset_file_path.endswith("tar.gz") or dataset_file_path.endswith(".tgz")):
        tar = tarfile.open(dataset_file_path, "r:gz")
        tar.extractall(path=extraction_directory)
        tar.close()
    elif (dataset_file_path.endswith("tar")):
        tar = tarfile.open(dataset_file_path, "r:")
        tar.extractall(path=extraction_directory)
        tar.close()
    print("Extraction of dataset done")

def get_current_directory():
    return os.path.dirname(os.path.realpath(__file__))


########################################
# Creating Dataset
########################################

def get_images(object_category, data_directory):
    if (not os.path.exists(data_directory)):
        print("Data directory not found. Are you sure you downloaded and extracted dataset properly?")
        return
    obj_category_dir = os.path.join(os.path.join(data_directory,"101_ObjectCategories"),object_category)
    images = [os.path.join(obj_category_dir,img) for img in os.listdir(obj_category_dir)]
    return images

def read_image(image_path):
    """Read and resize individual images - Caltech 101 avg size of image is 300x200, so we resize accordingly"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300,200), interpolation=cv2.INTER_CUBIC)
    return img

def return_images_per_category(data_directory):
    categories = os.listdir(data_directory+"/101_ObjectCategories/")
    object_images_count_dict = {}
    for category in categories:
        object_images_count_dict[category] = len(os.listdir(data_directory+"/101_ObjectCategories/"+category))
    object_images_count_dict = sorted(object_images_count_dict.items(), key=operator.itemgetter(1), reverse=True)
    return object_images_count_dict


def create_training_data(data_directory):
    i = 0
    X = np.ndarray((8677, 200, 300, 3), dtype=np.uint8)
    Y = []
    print("Preparing X and Y for dataset...")
    for category,_ in return_images_per_category(data_directory):
        if category == 'BACKGROUND_Google':
            continue
        print("Processing images of ",category)
        for image in get_images(category, data_directory):
            if not image.endswith('.jpg'):
                # to escape hidden ipynb checkpoints and other unnecessary files
                continue
            X[i] = read_image(image)
            Y.insert(i,category)
            i += 1
        print("Images processed : ",i+1," of 8678")
    print("Datasets constructed")
    return X,Y


def create_and_save_label_encoder(Y):
    label_encoder = LabelEncoder()
    Y_integer_encoded = label_encoder.fit_transform(Y)
    np.save('label_encoder.npy', label_encoder.classes_)
    return label_encoder, Y_integer_encoded

########################################
# Finding per category accuracy
########################################

def find_accuracy_per_category(data_directory, nn_model, label_encoder):
    category_accuracy_dict = {}
    for category,count in return_images_per_category(data_directory):
        correctly_classified = 0
        if category == 'BACKGROUND_Google':
            continue
        Y_category = [category for _ in range(count)]
        encoded = label_encoder.transform(Y_category)
        Y_category = to_categorical(encoded, num_classes=101)
        assert Y_category.shape == (count, 101)
        X_category = np.ndarray((count, 200, 300, 3), dtype=np.float64)
        for i,img in enumerate(get_images(category, data_directory)):
            if not img.endswith('.jpg'):
                # to escape hidden ipynb checkpoints and other unnecessary files
                continue
            X_category[i] = read_image(img) / 255
        score = nn_model.evaluate(x=X_category, y=Y_category, verbose=1)
        del X_category
        category_accuracy_dict[category] = score
    print("Accuracy found for each class")
    return category_accuracy_dict

def find_average_accuracy_for_model(data_directory, nn_model, label_encoder):
    category_accuracy_dict = find_accuracy_per_category('./data', nn_model, label_encoder)
    average_accuracy = 0
    for category, scores in category_accuracy_dict.items():
        print(category,":",scores[1])
        average_accuracy += scores[1]
    average_accuracy /= 101
    print("Average accuracy : ",average_accuracy)
    return average_accuracy, category_accuracy_dict

########################################
# COMMON STEPS
########################################

def perform_common_steps():
    print("Downloading dataset...")
    download_dataset(URL_CALTECH_101_DATA)
    data_directory = os.path.join(get_current_directory(),'data')
    extract_dataset(get_current_directory()+'/dataset.tgz',data_directory)
    X, Y = create_training_data(data_directory)
    label_encoder, Y_integer_encoded = create_and_save_label_encoder(Y)
    X_normalized = X.astype(np.float64) / 255
    del X
    Y_one_hot = to_categorical(Y_integer_encoded)
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_normalized, Y_one_hot, test_size=0.25, random_state=42)
    del X_normalized
    return X_train, X_validation, Y_train, Y_validation, label_encoder, data_directory


########################################
# MODELS
########################################

def get_transfer_learning_model_architecture():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    transfer_learning_arch = base_model.output
    transfer_learning_arch = GlobalAveragePooling2D()(transfer_learning_arch)
    transfer_learning_arch = Dense(1024, activation='relu')(transfer_learning_arch)
    transfer_learning_arch = Dropout(0.4)(transfer_learning_arch)
    transfer_learning_arch = Dense(512, activation='relu')(transfer_learning_arch)
    transfer_learning_arch = Dropout(0.4)(transfer_learning_arch)
    predictions = Dense(101, activation='softmax')(transfer_learning_arch)
    transfer_learning_model = Model(inputs=base_model.input, outputs=predictions)
    return transfer_learning_model

def get_simple_cnn_model_architecture():
    model_cnn = Sequential()
    model_cnn.add(Conv2D(16, (3,3), activation='relu', input_shape=(200,300,3)))
    model_cnn.add(Conv2D(32, (3,3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=2, strides=2))
    model_cnn.add(Conv2D(64, (3,3), activation='relu'))
    model_cnn.add(Conv2D(128, (3,3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=2, strides=2))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(101, activation='softmax'))
    return model_cnn

def get_cnn_model_with_dropouts_architecture():
    model_cnn_dropout = Sequential()
    model_cnn_dropout.add(Conv2D(32, (3,3), activation='relu', input_shape=(200,300,3)))
    model_cnn_dropout.add(Conv2D(32, (3,3), activation='relu'))
    model_cnn_dropout.add(MaxPooling2D(pool_size=2, strides=2))
    model_cnn_dropout.add(Conv2D(64, (3,3), activation='relu'))
    model_cnn_dropout.add(Conv2D(64, (3,3), activation='relu'))
    model_cnn_dropout.add(MaxPooling2D(pool_size=2, strides=2))
    model_cnn_dropout.add(Flatten())
    model_cnn_dropout.add(Dense(512, activation='relu'))
    model_cnn_dropout.add(Dropout(0.5))
    model_cnn_dropout.add(Dense(101, activation='softmax'))
    return model_cnn_dropout



########################################
# COMMANDS
########################################


@click.group()
def cli():
    """ This is a simple cli tool to train different neural network models for caltech 101 problem. """
    pass


@cli.command()
@click.option('--print-average-accuracy/--no-print-average-accuracy', default=False, help='Depending upon which you '
                                                                                          'pass, model can print '
                                                                                          'average accuracy after it '
                                                                                          'has trained the model')
def train_transfer_learning_model(print_average_accuracy):
    """This trains the transfer learning model."""
    X_train, X_validation, Y_train, Y_validation, label_encoder, data_directory = perform_common_steps()
    # Get transfer learning model
    transfer_learning_model = get_transfer_learning_model_architecture()

    #We freeze the model excepted the added layers
    #279 is number of mixed 9 layer
    for layer in transfer_learning_model.layers[:280]:
        layer.trainable = False
    for layer in transfer_learning_model.layers[280:]:
        layer.trainable = True

    # Compile and fit the model
    opt=Adadelta(lr=1.0, rho=0.9, epsilon=1e-08, decay=0.0)
    transfer_learning_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    callbacks = [ModelCheckpoint('model_transfer_learning_weights.h5', monitor='val_acc', save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')]
    transfer_learning_model.fit(X_train, Y_train, batch_size=32, epochs=15, verbose=1, validation_data=(X_validation,Y_validation), callbacks=callbacks)
    print(">> Model trained.")

    json_model = transfer_learning_model.to_json()
    with open("model_transfer_learning.json", "w") as f:
        json.dump(json.loads(json_model), f, indent=4)
    print(">> Model saved to file.")

    if print_average_accuracy:
        print(">> Finding average accuracy per category")
        total_avg_accuracy, category_accuracy_dict =  find_average_accuracy_for_model(data_directory,
                                                                      transfer_learning_model, label_encoder)
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(category_accuracy_dict)
        print(">> Total average accuracy across categories : ",str(total_avg_accuracy))
    print("Done. Exiting.")


@cli.command()
@click.option('--print-average-accuracy/--no-print-average-accuracy', default=False, help='Depending upon which you '
                                                                                          'pass, model can print '
                                                                                          'average accuracy after it '
                                                                                          'has trained the model')
def train_simple_cnn_model(print_average_accuracy):
    """This trains a simple CNN model with 5 layers."""
    X_train, X_validation, Y_train, Y_validation, label_encoder, data_directory = perform_common_steps()
    # Get transfer learning model
    model_cnn = get_simple_cnn_model_architecture()
    # Compile and fit the model
    model_cnn.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    callbacks = [ModelCheckpoint('model_cnn_weights.h5', monitor='val_acc', save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')]
    model_cnn.fit(X_train, Y_train, batch_size=64, epochs=30, verbose=1, validation_data=(X_validation,Y_validation), callbacks=callbacks)
    print(">> Model trained.")

    json_model = model_cnn.to_json()
    with open("model_transfer_learning.json", "w") as f:
        json.dump(json.loads(json_model), f, indent=4)
    print(">> Model saved to file.")

    if print_average_accuracy:
        print(">> Finding average accuracy per category")
        total_avg_accuracy, category_accuracy_dict =  find_average_accuracy_for_model(data_directory,
                                                                                      model_cnn, label_encoder)
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(category_accuracy_dict)
        print(">> Total average accuracy across categories : ",str(total_avg_accuracy))
    print("Done. Exiting.")


@cli.command()
@click.option('--print-average-accuracy/--no-print-average-accuracy', default=False, help='Depending upon which you '
                                                                                          'pass, model can print '
                                                                                          'average accuracy after it '
                                                                                          'has trained the model')
def train_cnn_model_with_dropouts(print_average_accuracy):
    """This trains an improvement over above simple CNN model which uses dropouts to reduce overfitting."""
    X_train, X_validation, Y_train, Y_validation, label_encoder, data_directory = perform_common_steps()
    # Get transfer learning model
    model_cnn_dropout = get_cnn_model_with_dropouts_architecture()
    # loss and optimizer
    model_cnn_dropout.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['accuracy'])

    # training
    callbacks = [ModelCheckpoint('model_cnn_dropout_weights.h5', monitor='val_acc', save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')]
    model_cnn_dropout.fit(X_train, Y_train, batch_size=16, epochs=10, verbose=1, validation_data=(X_validation,Y_validation), callbacks=callbacks)
    print(">> Model trained.")

    json_model = model_cnn_dropout.to_json()
    with open("model_transfer_learning.json", "w") as f:
        json.dump(json.loads(json_model), f, indent=4)
    print(">> Model saved to file.")

    if print_average_accuracy:
        print(">> Finding average accuracy per category")
        total_avg_accuracy, category_accuracy_dict =  find_average_accuracy_for_model(data_directory,
                                                                                      model_cnn_dropout, label_encoder)
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(category_accuracy_dict)
        print(">> Total average accuracy across categories : ",str(total_avg_accuracy))
    print("Done. Exiting.")


if __name__ == "__main__":
    cli()
