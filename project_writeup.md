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

* **Problem Statement**: The algorithm we develop consists of three parts: a dog detector, a human face detector and a dog breed classifier. While we can use existing methods for detecting dog and human face, we will build a CNN for the purpose of dog breed classification. There are two ways of building a CNN: 1) construct from scratch, and 2) use Transfer Learning. Details of these two methods will be discussed later on.


* **Metrics**: We use accuracy score to measure the performance of our model, i.e., what percentage of data is correctly classified？ Since we are working on a multi-class classification problem with 133 classes, precision, recall and F1 score do not apply. Hence, we use the accuracy score only.


<a id='analysis'></a>
## 2. Exploratory Data Analysis

<img width="400" alt="breedfrequency" src="app_preparation/images/breed_frequency.png">

The task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

| Brittany | Welsh Springer Spaniel |
| - | - |
| <img src="app_preparation/images/Brittany_02625.jpg" width="100"> | <img src="app_preparation/images/Welsh_springer_spaniel_08203.jpg" width="200"> |

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="app_preparation/images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="app_preparation/images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  The vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | - | -
<img src="app_preparation/images/Labrador_retriever_06457.jpg" width="150"> | <img src="app_preparation/images/Labrador_retriever_06455.jpg" width="220"> | <img src="app_preparation/images/Labrador_retriever_06449.jpg" width="200">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

<a id='method'></a>
## 3. Methodology
* Data Preprocessing
* Implementation
* Refinement


<a id='result'></a>
## 4. Results
* Model Evaluation and Validation
* Justification


<a id='conclusion'></a>
## 5. Conclusion
* Reflection
* Improvement