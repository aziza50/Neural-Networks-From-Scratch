#import the neural network and other libraries 
import cv2
import numpy as np
import networks as nn
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_iris


def neural_network(dataset):

    if dataset == "iris":
        '''
        Iris Dataset:

        Training Validation: accuracy: 0.975, loss = 0.059
        Testing Validation: accuracy: 0.967, loss: 0.065

        '''
        X, y = load_iris(return_X_y=True)
        #split the images
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        #Ensure shape matches the input for the neural network (#of samples, #of features)
        print(X.shape)
        #Shuffle to optimize training and learning
        X, y = shuffle(X, y, random_state=42)
        #Ensure the class distribution is as equal as possible to prevent the model from 'memorizing' which label is most often correct
        print("Class distribution:", np.bincount(y))

        model = nn.Model()

        model.add(nn.Layer_Dense(4, 128))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dropout(0.05))
        model.add(nn.Layer_Dense(128, 128))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dense(128, 3))  
        model.add(nn.Activation_Softmax())


        model.set(
            loss = nn.Loss_CategoricalCrossentropy(),
            optimizer=nn.Optimizer_Adam(decay=1e-6),
            accuracy = nn.Accuracy_Categorical()
        )

        model.finalize()

        model.train(X, y, validation_data=(X_test, y_test), epochs = 30, batch_size = 32, print_every = 100)

        model.evaluate(X, y)


    elif dataset == "mnist":
        '''
        MNIST Dataset

        Training validation: accuracy: 0.941, loss: 0.196
        Testing validation:  accuracy: 0.925, loss:0.276
        '''
        mnist = fetch_openml('mnist_784', version=1)
        X, y = mnist["data"], mnist["target"].astype(np.uint8)
        #split the images
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #56000 samples with shapes of 784
        print(X.shape)

        X, y = shuffle(X, y, random_state=42)
        #[5560 6277 5610 5708 5529 5040 5480 5790 5468 5538] class dist
        print("Class distribution:", np.bincount(y))

        model = nn.Model()

        model.add(nn.Layer_Dense(784, 256))
        model.add(nn.Activation_ReLU())
        #model.add(nn.Layer_Dropout(0.05))
        model.add(nn.Layer_Dense(256, 256))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dense(256, 10))  
        model.add(nn.Activation_Softmax())


        model.set(
            loss = nn.Loss_CategoricalCrossentropy(),
            optimizer=nn.Optimizer_Adam(decay=1e-6),
            accuracy = nn.Accuracy_Categorical()
        )

        model.finalize()

        model.train(X, y, validation_data=(X_test, y_test), epochs = 30, batch_size = 128, print_every = 100)

        model.evaluate(X, y)

    elif dataset == "malaria":
        '''
        Malaria Dataset:
        
        Training validation: accuracy: 0.653   loss: 0.626
        Testing validation:  accuracy: 0.674    loss: 0.602
        '''
        
        WIDTH = 64
        HEIGHT = 64
        def resize_image(image):
            #simply scale the image down to fit 100 by 100 and pad
            #if necessary
            image_h, image_w = image.shape[:2]
            scale = WIDTH / max(image_h, image_w)
            scaled_h, scaled_w = int(image_h * scale), int(image_w * scale)    
            resized = cv2.resize(image, (scaled_w,scaled_h))

            square = np.zeros((WIDTH, HEIGHT), dtype = np.uint8)
            x_center = (WIDTH - scaled_w)//2
            y_center = (WIDTH - scaled_h)//2

            square[y_center:y_center+scaled_h, x_center:x_center+scaled_w] = resized

            return square

        def load_images(folder, label):
            images = []
            labels = []
            for image_path in os.listdir(folder):
                image = cv2.imread(os.path.join(folder, image_path), cv2.IMREAD_GRAYSCALE)
                if image is not None: 
                    image = resize_image(image)
                    image = image.astype(np.float32) / 255.0
                    images.append(image)
                    labels.append(label)
            return images, labels

        parasitized_images, parasitized_labels = load_images('cell_images/Parasitized', 1)
        uninfected_images, uninfected_labels = load_images('cell_images/Uninfected',0) 

        images = parasitized_images + uninfected_images
        classes = parasitized_labels + uninfected_labels

        #turn into numpy arrays
        images = np.array(images)

        classes = np.array(classes).astype('uint8')
        #flatten since the nn takes in a 1D array
        images= images.reshape(images.shape[0], -1)

        #split the images
        X, X_test, y, y_test = train_test_split(images, classes, test_size=0.2, random_state=42)

        #print(X.shape)
        #print(X.min(), X.max())

        X, y = shuffle(X, y, random_state=42)
        print("Class distribution:", np.bincount(y))

        model = nn.Model()

        model.add(nn.Layer_Dense(4096, 512))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dropout(0.05))
        model.add(nn.Layer_Dense(512, 512))
        model.add(nn.Activation_ReLU())
        model.add(nn.Layer_Dense(512, 2))  
        model.add(nn.Activation_Softmax())


        model.set(
            loss = nn.Loss_CategoricalCrossentropy(),
            optimizer=nn.Optimizer_Adam(decay=1e-6),
            accuracy = nn.Accuracy_Categorical()
        )

        model.finalize()

        model.train(X, y, validation_data=(X_test, y_test), epochs = 15, batch_size = 128, print_every = 100)

        model.evaluate(X, y)


neural_network("malaria")