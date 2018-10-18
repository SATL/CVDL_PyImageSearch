# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='./images/omr_test_01.png',
	help="path to the input image")
args = vars(ap.parse_args())
 
def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

#load it blury it and find edges
image = cv2.imread(args["image"])
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(blurred, 75, 200)
show_image(edged)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
doc = None

if len(cnts) > 0:
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        cv2.drawContours(copy, [approx], -1, (0, 255, 0), 2)
        if len(approx) == 4:
            doc = approx
            cv2.drawContours(copy, [doc], -1, (255, 0, 0), 2)
            break
show_image(copy)

paper = four_point_transform(image, doc.reshape(4,2))
warped = four_point_transform(gray, doc.reshape(4,2))
show_image(paper)
show_image(warped)

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
show_image(thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionsCnts=[]
copy = paper.copy()
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    aspectRatio = w / float(h)
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)


    if w>=20 and h >=20 and aspectRatio >=0.9 and aspectRatio <=1.1:
        questionsCnts.append(c)
        cv2.drawContours(copy, [approx], -1, (0, 0, 255), 2)
show_image(copy)

questionsCnts = contours.sort_contours(questionsCnts, method='top-to-bottom')[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(questionsCnts), 5)):
    cnts = contours.sort_contours(questionsCnts[i: i+5])[0]
    bubbled = None

    for ( j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype='uint8')
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        
        if bubbled is None or total>bubbled[0]:
            bubbled = (total, j)
    
    print('bubbled', bubbled)
    color = (0,0,255) #red
    k = ANSWER_KEY[q]
    if k == bubbled[1]:
        color = (0, 255,0) #green
        correct+=1
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
