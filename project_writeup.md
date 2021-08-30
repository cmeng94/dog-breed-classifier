# Project Writeup

## Table of Contents
1. [Project Definition](#def)
2. [Analysis](#analysis)
3. [Methodology](#method)
4. [Results](#result)
5. [Conclusion](#conclusion)


<a id='def'></a>
## 1. Project Definition

* **Project Overview**: In this project, we build a model to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. The [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) that the model will be trained on is provided by Udacity, which contains dog image data of 133 dog breeds split to train, validation and testing sets.

* **Problem Statement**: The algorithm we develop consists of three parts: a dog detector, a human face detector and a dog breed classifier. While we can use existing methods for detecting dog and human face, we will build a CNN for the purpose of dog breed classification. There are two ways of building a CNN: 1) construct from scratch, and 2) use [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning). Details of these two methods will be discussed later on.


* **Metrics**: We use accuracy score to measure the performance of our model, i.e., what percentage of data is correctly classified？ Since we are working on a multi-class classification problem with 133 classes, metrics that are frequently used for binary classification such as precision, recall and F1 score do not apply. Hence, we use the accuracy score only. When training the model, we minimize the [Cross-Entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss function.


<a id='analysis'></a>
## 2. Exploratory Data Analysis

The training data consists of `6680 dog images` belonging to `133 breeds`. Firstly, to learn whether the breeds are evenly distributed, we plot a histogram of breed frequencies. 

<img width="400" alt="breedfrequency" src="app_preparation/images/breed_frequency.png">

It can be seen that the 133 breeds are not evenly distributed. The bin that has the most frequency is 50-55, i.e., around 27 breeds have between 50 and 55 images. The breed that has the least data is *"Norwegian buhund"*, with 26 images, and the breed that has the most data is *"Alaskan malamute"*, with 77 images.

The task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that even a human would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

| Brittany | Welsh Springer Spaniel |
| - | - |
| <img src="app_preparation/images/Brittany_02625.jpg" width="100"> | <img src="app_preparation/images/Welsh_springer_spaniel_08203.jpg" width="200"> |

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

| Curly-Coated Retriever | American Water Spaniel |
| - | - |
| <img src="app_preparation/images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="app_preparation/images/American_water_spaniel_00648.jpg" width="200"> |


Likewise, recall that labradors come in yellow, chocolate, and black.  The vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

| Yellow Labrador | Chocolate Labrador | Black Labrador |
| - | - | - |
| <img src="app_preparation/images/Labrador_retriever_06457.jpg" width="150"> | <img src="app_preparation/images/Labrador_retriever_06455.jpg" width="220"> | <img src="app_preparation/images/Labrador_retriever_06449.jpg" width="200"> |

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

<a id='method'></a>
## 3. Methodology

* **Data Preprocessing**: When using [TensorFlow](https://www.tensorflow.org/) as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape `(nb_samples, rows, columns, channels)`, where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  
Note that we need to resize each input image to a square image that is `224 x 224` pixels, convert it to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the resulting tensor will always have shape `(1, 224, 224, 3)`. If we stack all input images together, the resulting tensor will take the size `(nb_samples, 224, 224, 3)`.  
When using `Transfer Learning`, however, we not only need to resize the input data as mentioned above, but also need to transfrom them into `bottleneck features`. In other words, we need to pass the resized data through the pre-trained neural network to obtain the output, i.e., the bottleneck features that are to be fed into newly added layers on top of the pre-trained model. 

* **Implementation**: The first CNN model we built was created from scrach. When trained using 10 epochs with the [Adam](https://arxiv.org/abs/1412.6980) optimizer to minimize the `Cross-Entropy` loss, this model gives a `4.3%` testing accuracy. The model architecture is shown below:  
    <img src="app_preparation/images/model1.png" width="500">  
    In this CNN, the first layer is a `convolutional layer`, which extracts the various features from the input image. In this layer, 16 convolution kernels is applied to the input, and the output is referred to as the Feature Map. The convolutional layer is usually followed by a `pooling layer`, which is primarily used to decrease the size of the convolved feature map hence reducing computational costs. Pooling methods include average pooling (smoothing out the image) and max pooling (extracting brighter pixels).  
    We repeat the above combination of the two layers 3 times to extract more features and increase model complexity. Given complex training data, we need correspondingly complex CNN structure to guarantee accuracy. But increasing the depth of a CNN may also cause overfitting, so we need to be careful with adding layers. Since the objective for this step is to create a model that has >1% testing accuracy, a simple CNN model like this would suffice.  
    The second to last layer is a `global pooling layer`, which reduces the dimensionality of the data from 3d to 1d. This transforms the data in preparation for the very last fully connected (dense) layer. The last layer is a `fully connected layer`. We specify the output size to be 133 since we need to map the data to exactly 133 categories. We apply the `softmax` activation function. This needs to be the last step of the CNN since it normalizes the output of a network to a probability distribution over predicted output classes. The class with the highest probability will be the predicted class.  


* **Refinement**: The model has only 4.3% testing accuracy when trained using 10 epochs. If we want to improve model accuracy, we need to increase model complexity and increase number of training epochs. Both of these requirement longer training time. An alternative we can consider is using `Transfer Learning`, which allows us to use pre-trained CNNs and train a few additional customized layers for our classficiation purpose. This greatly reduces training time. The idea is to first tranform training data to `bottleneck features` using the pre-trained CNN, and then feed these bottleneck features to train the additional customized layers, e.g., 
    <img src="app_preparation/images/model2.png" width="500">   
    The reasoning behind adding these layers is that in order to obtain class probabilities, we first need a global pooling layer to transform the 3D output to 1D, and then a fully connected layer with SoftMax activation to map to the 133 dog breed classes for probabilities.  
    There are a number of pre-trained model we can use here, for example, [AlexNext, VGGNet, ResNet and Inception](https://towardsdatascience.com/the-w3h-of-alexnet-vggnet-resnet-and-inception-7baaaecccc96). In this project, we used both VGG16 and InceptionV3 to build transfer learning CNN models.

<a id='result'></a>
## 4. Results

* **Model Evaluation and Validation**: We built 3 CNN models in total: 1) from sctach, 2) VGG16 with Transfer Learning (VGG16-TL), and 3) InceptionV3 with Tranfer Learning (InceptionV3-TL). For all models, the `Adam` optimizer is used to minimize the `Cross-Entropy` loss function. 20 training epoches were run on both Transfer Learning models that use the same additional layers, but only 10 epochs were run on the model built from scratch for training time consideration. During the training process, a model is generated at each epoch, and the model with the lowest validation loss is saved. The table below compares the testing accuracy of the 3 models. It can be seen that the `InceptionV3` CNN has the highest testing accuracy.

    | | From Scratch | VGG16-TL | InceptionV3-TL
    | - | - | - | - |
    | Optimizer | Adam | Adam | Adam |
    | Loss Function | Cross-Entropy | Cross-Entropy | Cross-Entropy |
    | Number of Epochs | 10 | 20 | 20 |
    | Testing Accuracy| 4.3062% | 69.3780% | 81.8182% |

* **Justification**:

* **Examples**:
    | Maltese | Maltese | Maltese|
    | - | - | - |
    | <img src="app_preparation/images/my_maltese1.png" width="300">  |   <img src="app_preparation/images/my_maltese2.png" width="300"> | <img src="app_preparation/images/my_maltese3.png" width="300">

    | Human 1 | Human 2 |
    | - | - |
    | <img src="app_preparation/images/human1.png" width="300">  |   <img src="app_preparation/images/human2.png" width="300"> |

<a id='conclusion'></a>
## 5. Conclusion

* **Reflection**:

* **Improvement**: