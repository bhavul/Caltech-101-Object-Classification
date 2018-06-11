import click
from keras.models import model_from_json
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder

model_weights_map = {
    'model_transfer_learning.json':'model_transfer_learning_weights.h5',
    'model_cnn.json':'model_cnn_weights.h5',
    'model_cnn_dropout.json':'model_cnn_dropout_weights.h5'
}

def load_nn_model(model_file_path, weights_file_path):
    """This loads the required neural network model."""
    json_file = open(model_file_path,'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # load model
    model = model_from_json(loaded_model_json)
    print("Model loaded - ",str(model_file_path))
    # load weights
    model.load_weights(weights_file_path)
    print("Weights loaded for the model - ",str(weights_file_path))
    return model

def get_current_directory():
    return os.path.dirname(os.path.realpath(__file__))

def read_image(image_path):
    """Read and resize individual images - Caltech 101 avg size of image is 300x200, so we resize accordingly"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300,200), interpolation=cv2.INTER_CUBIC)
    return img

def validations_for_image_path(image_path):
    if image_path is None:
        print("No image provided! Please use --image-path argument to provide an image to predict for.")
        raise ValueError('No image provided! Please use --image-path argument to provide an image to predict for.')
    if not os.path.exists(image_path):
        print("Not a valid image. Please make sure you've given correct path to an image!")
        raise ValueError('The image does not exist on the path!')
    if not (image_path.endswith('.jpg') or image_path.endswith('.jpeg') or image_path.endswith('.png')):
        print("Image format not supported!")
        raise ValueError('Image format is not supported. Please use a jpg/jpeg/png!')

def check_if_weights_exists(model_file):
    """A weights file must be there for predictions. This method checks for its existence."""
    model_file_name = os.path.basename(model_file)
    if model_file_name not in model_weights_map:
        print("Incorrect model file given. Mapping does not exist for weights! The program will exit!")
        raise ValueError('Incorrect model file given. This script only supports one of the three models which are '
                         'part of Git.')
    weights_file_name = model_weights_map[model_file_name]
    if not os.path.exists(os.path.join(get_current_directory(),weights_file_name)):
        print("Weights have not been downloaded. Model wouldn't be able to load properly, or will need re-training!")
        raise ValueError('Weights file does not exist. Either use train.py to train the model again, or download the '
                         'pre-trained weights from the Google drive URLs shared in Readme.md file.')
    return weights_file_name, os.path.join(get_current_directory(),weights_file_name)


def get_label_encoder():
    """This returns the label encoder which contains label to integer mapping"""
    encoder = LabelEncoder()
    encoder.classes_ = np.load('label_encoder.npy')
    return encoder


@click.command()
@click.option('--model-file-path', help='Give the path of model json file. By default, transfer learning model is used '
                                   'since it performs the best out of the three models.')
@click.option('--image-path', help="The image which needs to be classified")
def find_object_in_image(model_file_path, image_path):
    """This prints out the label of the object it classifies in the image. Since the models have been trained on
    caltech 101, it can only print out one of the 101 object categories from the dataset."""
    validations_for_image_path(image_path)
    if model_file_path is None:
        print("Choosing transfer learning model for prediction since no model specified in arguments.")
        model_file_path = os.path.join(get_current_directory(),'model_transfer_learning.json')
    weights_file_name, weights_file_path = check_if_weights_exists(model_file_path)
    model = load_nn_model(model_file_path, weights_file_path)
    np_image = np.ndarray((1, 200, 300, 3), dtype=np.float64)
    np_image[0] = read_image(image_path) / 255
    label_encoder = get_label_encoder()
    prediction = model.predict(np_image)
    label_predicted = label_encoder.inverse_transform(np.argmax(prediction))
    print("Prediction : ",label_predicted)


if __name__ == "__main__":
    """This prints out the label of the object it classifies in the image. Since the models have been trained on
    caltech 101, it can only print out one of the 101 object categories from the dataset."""
    find_object_in_image()
