# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="kaggle_dogs_vs_cats",
	help="path to input dataset")
ap.add_argument("-m", "--model", default="output/simple_neural_network.hdf5",
	help="path to output model file")
args = vars(ap.parse_args())
 
# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
print("[Info] len {}".format(len(imagePaths)))
# initialize the data matrix and labels list
data = []
labels = []

for ( i, imagePath) in enumerate(imagePaths):
	#load the image
	#path formtai /path/{class}.{num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	#feature raw pixel intensities, then ipdate the matrix and labels list
	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(label)

	if i>0 and i %1000 == 0:
		print("[Info] processed {}/{}".format(i, len(imagePaths)))


labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(labels)

#sclae the input to the range [0,1], then transform into vectors in the range [0, mun:classes], this generates a vector for each label where the indes ox the label is set to 1 and all other entries to 0
data = np.array(data) /255.0
labels = np_utils.to_categorical(labels, 2)

print("[Info] constructing training/ test")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.25, random_state=42)

#architecture of the netword
model = Sequential()
model.add(Dense(768, input_dim=3072, init="uniform", activation="relu") )
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[Info] compiling model")
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
model.fit(trainData, trainLabels, epochs=50, batch_size=128, verbose=1)


print("[Info] evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size=128, verbose=1)
