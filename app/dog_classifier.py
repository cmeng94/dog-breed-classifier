import sys
import cv2
import numpy as np
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.callbacks import ModelCheckpoint  
from keras.preprocessing import image                  
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input

def face_detector(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image 
    
    Output:
    "True" if at least one face is detected in image stored at img_path
    "False" is no face is detected
    '''

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../app_preparation/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image 
    
    Output:
    a 4D tensor suitable for supplying to a Keras CNN with shape (1,224,224,3)
    '''

    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def ResNet50_predict_labels(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image 
    
    Output:
    prediction vector by ResNet50 for image located at img_path
    '''

    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    ResNet50_model_orig = ResNet50(weights='imagenet')
    return np.argmax(ResNet50_model_orig.predict(img))


def dog_detector(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image 
    
    Output:
    "True" if a dog is detected in image stored at img_path
    "False" is no dog is detected
    '''

    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def get_DL_model():

    '''
    Input:
    None
    
    Output:
    DL_model: CNN pretrained upon the InceptionV3 neural network using transfer learning
    '''

    print('Loading model...')
    DL_model = Sequential()
    DL_model.add(GlobalAveragePooling2D(input_shape=(5, 5, 2048)))
    DL_model.add(Dense(133, activation='softmax'))
    DL_model.load_weights('../app_preparation/saved_models/weights.best.InceptionV3.hdf5')
    return DL_model

def extract_InceptionV3(tensor):

    '''
    Input:
    tensor: image processed by path_to_tensor

    Output:
    bottleneck feature transformed by InceptionV3
    '''

    return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def DL_predict_breed(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image 
    
    Output:
    breed: breed of dog in input image predicted by CNN trained on top of the InceptionV3 neural network
    '''

    DL_model = get_DL_model()
    print('Predicting breed...')
    # extract bottleneck features
    bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = DL_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model

    with open('dog_names.txt', 'r') as f:
        dog_names = f.read().splitlines()
        f.close()

    breed = dog_names[np.argmax(predicted_vector)]
    return breed

def classify_dog_breed(img_path):

    '''
    Input:
    img_path: string-valued file path to a color image
    
    Output:
        - if a dog is detected in the image, the predicted breed is returned
        - else if a human is detected in the image, the resembling dog breed is returned
        - if neither is detected in the image, "neither" is returned
    '''
    
    if dog_detector(img_path):
        breed = DL_predict_breed(img_path)
        # print('I detect a {} dog!'.format(breed))
        return ['dog', breed]

    elif face_detector(img_path):
        breed = DL_predict_breed(img_path)
        # print('I detect a human face resembling a {} dog!'.format(breed))
        return ['human', breed]
        
    else:
        # print("Sorry, I can only classify a dog or a human!")   
        return 'neither'

# def main():
#     print(sys.argv)

#     if len(sys.argv) == 2:

#         img_path = sys.argv[1]
#         print('Loading image...')

#         classify_dog_breed(img_path)

#     else:
#         print('Please provide the filepath of the image as the first argument.')

# if __name__ == '__main__':
# 	main()