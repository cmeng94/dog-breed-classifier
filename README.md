# Dog Breed Classifier via CNNs and Transfer Learning

## Table of Contents
1. [Project Description](#intro)
2. [File Descriptions](#files)
3. [Getting Started](#start)
4. [Web App Usage](#use)
5. [Contact](#contact)
6. [Acknowledgement and Licensing](#acknowledge)


<a id='intro'></a>
## 1. Project Description
The goal of this project is to build a CNN for classifying the dog breed of a user-supplied image. The training data set contains 6680 dog images of 133 dog breeds. We use Transfer Learning on top of the pretrained [`InceptionV3`](https://cloud.google.com/tpu/docs/inception-v3-advanced) architecture to obtain a CNN for our dog breed classification purpose. The classifier has **81.81%** accuracy on the testing data set. The deliverable of this project is a Web App that is built upon the CNN model.

<img width="800" alt="screenshot1" src="https://user-images.githubusercontent.com/11303419/131235836-0e496cd9-0556-4ab8-8e8c-29d0ae4da98b.png">


<a id='files'></a>
## 2. File Descriptions
* `app`: This folder contains the Web App. Its contents include:
	- `run.py`: the main execution file for the Web App
	- `dog_classifier.py`: python file with classification methods that are used when the Web App is run
	- `dog_names.txt`: file containing the dog breed names
	- `templates`: folder with the html files
	- `static`: folder with input images for the Web App

* `app_prepataion`: This folder contains files used to prepare models and methods for the Web App in the `app` folder. Its contents include:
	- `dog_app.ipynb`: Ipython Notebook that loads training data, trains model and defines classification methods
	- `extract_bottleneck_features.py`: python file with helper functions for classification
	- `images`: folder containing images for display purposes in `dog_app.ipynb`
	- `saved_models`: folder containing saved models trained in `dog_app.ipynb`
	- `haarcascades`: folder containing pretrained face detector  


<a id='start'></a>
## 3. Getting Started
### Dependencies
The code is developed with Python 3.9 and is dependent on a number of python packages listed in `requirements.txt`. To install required packages, run the following line in terminal:
```sh
pip3 install -r requirements.txt
```
Note that the installation of some Python packages is different for Apple M1 CPU. The user may follow the instructions in this [YouTube Video](https://youtu.be/_CO-ND1FTOU).

### Installation
To run the code locally, create a copy of this GitHub repository by running the following code in terminal:
```sh
git clone https://github.com/cmeng94/dog-breed-classifier
```
### Execution
* To run the Web App, change to the `app` folder and execute the following line, then visit [http://0.0.0.0:3001/](http://0.0.0.0:3001/).
```sh
python3 run.py
```

<a id='use'></a>
## 4. Web App Usage
With the Web App running, the user can submit a path to an image for classification. **Note that the user-supplied image must be located in the `app/static` folder for the app to run properly.** For example,
<img width="800" alt="screenshot2" src="https://user-images.githubusercontent.com/11303419/131238943-d1a5fe73-7626-430b-afe2-872f82b337bb.png">

The classification result is shown after clicking the `Classify Dog Breed` button:
<img width="800" alt="screenshot3" src="https://user-images.githubusercontent.com/11303419/131238942-2af97b46-e43a-4eaa-8627-340dcb15ec88.png">

<a id='contact'></a>
## 5. Contact
**Chang Meng**
* Email: chang_meng@live.com
* Website: [https://sites.google.com/view/changmeng](https://sites.google.com/view/changmeng)

<a id='acknowledge'></a>
## 6. Acknowledgement and Licensing
This project is part of the [Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) program at [Udacity](https://www.udacity.com/). For licensing information, please refer to `LICENSE.txt`.

