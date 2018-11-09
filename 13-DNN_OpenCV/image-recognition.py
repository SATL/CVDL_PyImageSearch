import numpy as np
import argparse
import time
import cv2
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", default="bvlc_googlenet.prototxt",
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="bvlc_googlenet.caffemodel",
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", default="synset_words.txt",
	help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, height =500)

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]


# our CNN requires fixed spatial dimensions for our input image(s)
# so we need to ensure it is resized to 224x224 pixels while
# performing mean subtraction (104, 117, 123) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
print("[info] Loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[Info] classification took {:.5} seconds".format(end-start))

idxs = np.argsort(preds[0])[::-1][:5]

for (i,idx) in enumerate(idxs):
    if i ==0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx]*100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    print("[Info] {}, label:{}, prob: {:.5}".format(i+1, classes[idx], preds[0][idx]))

cv2.imshow("Image", image)
cv2.waitKey(0)