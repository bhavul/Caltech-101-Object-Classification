# Caltech101-Object-Classification

In this experiment, we try to do object classification on Caltech 101 dataset to identify the 101 object categories. Info about the dataset is available [here](http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html).  

While the dataset could be used for object detection as well, I've only done object classification here. The whole experiment from downloading the dataset to training three different neural network based models and doing predictions is all written in the jupyter notebook named `caltech-experiment.ipynb`. 

There are 3 models that have been trained : 

1. **Simple CNN** :  
A simple 5 layer CNN Network with Convolutional, Max Pooling and FC layer. Relu is used as activation of intermediate layers while softmax for the last layer. Categorical Cross Entropy is used as the loss function.

2. **Simple CNN with Dropouts** :  
It was observed that the above model was overfitting badly. It is kind of expected as well considering the dataset is quite small. One way to avoid overfitting is Dropout Regularization. Dropouts have been used in this model and a couple of extra layers have been added to showcase the reduction in overfitting while training. However, the final results sort of remain similar.

3. **Transfer learning based on InceptionV3**:  
When the above two models didn't achieve a great validation accuracy, I thought of giving a tried-and-tested model a try. Here, we used InceptionV3 and added a few Fully Connected (FC) layers at the end. Dropout also exists in these FC layers. This model performed brilliantly with almost no overfitting!

Finally, it is suggested on the dataset page that since there's imbalance between the number of pictures that are available for each category, it is recommended for anyone experimenting with the dataset to find accuracy per category, and then average it out to find total average accuracy.  

This has been done in the code as well. The first two hand-crafted models give total average accuracy of about 80-83%, while the transfer learning model outputs a total average accuracy of 97%!


#### Why Deep Neural Network Models?

Considering the dataset is quite small, training deep neural networks doesn't seem to be a good idea to start with. However, due to the time crunch that I had, and since I had recently been working a lot with deep learning models, I decided to give it a try. 

One more reason for doing this was, people had already tried and tested SVM, KNN and decision tree classifiers on this dataset. I couldn't find many who had tried deep learning with it. So, for science! :)


# Files

**`caltech-experiment.ipynb`** : This jupyter notebook contains the write-up of whole experiment, and details the different approaches we take for solving object classification. I've tried to write it as a tutorial so anyone who wishes to go through it could understand what we're doing.

**`train.py`** : This script can be used to train any of the 3 models that have been trained in jupyter notebook as well. 

**`predict.py`** : This script can be used to do prediction using any of the 3 models provided in the repository.


# Installation

There are some ML libraries that need to be there for the scripts in this repository to work. You can easily install them by using pip.

```
pip install -r requirements.txt
```

# How to use?



The scripts **`train.py`** and **`predict.py`** have been written as CLI tools. Thanks to click library which makes this easy. So, it is pretty easy to understand how to use them as I've tried to document the commands. 

### Training


`python train.py --help`

wil give you the following output :

```
Usage: train.py [OPTIONS] COMMAND [ARGS]...

  This is a simple cli tool to train different neural network models for
  caltech 101 problem.

Options:
  --help  Show this message and exit.

Commands:
  train_cnn_model_with_dropouts  This trains an improvement over above
                                 simple...
  train_simple_cnn_model         This trains a simple CNN model with 5 layers.
  train_transfer_learning_model  This trains the transfer learning model.
```

You could further check each of the command for what options they take.

```
$ python train.py train_transfer_learning_model --help  

Using TensorFlow backend.
Usage: train.py train_transfer_learning_model [OPTIONS]

  This trains the transfer learning model.

Options:
  --print-average-accuracy / --no-print-average-accuracy
                                  Depending upon which you pass, model can
                                  print average accuracy after it has trained
                                  the model
  --help                          Show this message and exit.
```


So, if you had to train the transfer learning model and wanted to print the total average accuracy once it has trained, then the command would look like : 

```python train.py train_transfer_learning_model --print-average-accuracy```

### Testing / Predicting

You can use any of the three models that come in this repo (or can be trained via this repo). The corresponding model files are - `model_cnn.json`, `model_cnn_dropout.json` and `model_transfer_learning.json`.  

**Note**: It is necessary to either train the models first, or [download the pre-trained weights for each of the models from here](https://drive.google.com/file/d/1B89w0M6RYVTilanCnLwEEjFMPWnvrPu-/view?usp=sharing).  

To test for any image (say airplane.jpg) which contains an object, you can run the command like :   

```
python predict.py --model-file-path /path/to/repo/model_cnn.json --image-path /path/to/image/airplane.jpg
```

Again, if you need to know anything else about the command, it s documented like a CLI tool. So, just hit `--help` like so :  

```
python predict.py --help  

Using TensorFlow backend.
Usage: predict.py [OPTIONS]

  This prints out the label of the object it classifies in the image. Since
  the models have been trained on caltech 101, it can only print out one of
  the 101 object categories from the dataset.

Options:
  --model-file-path TEXT  Give the path of model json file. By default,
                          transfer learning model is used since it performs
                          the best out of the three models.
  --image-path TEXT       The image which needs to be classified
  --help                  Show this message and exit.
```




# Pending
- Pictures showcasing the architecture of each model
- Adding Visualization and graphs to the notebook as well as to this readme showcasing predictions and losses and comparisons between three models in terms of graphs
- Explanation of why did I choose Caltech 101 dataset to work on for linkedin assignment and not some other dataset
- Refactoring of `train.py` and `predict.py` - Even though they're readable, they're one long file for each which breaks SRP. 
- Add a feature of using webcame to predict the object
- Insights about which objects do all three approaches always get wrong - and why.

# Author

Bhavul Gauri
