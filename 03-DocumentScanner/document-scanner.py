#imports
import numpy as np
import argparse
import cv2
import imutils
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

#parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="./images/receipt.jpg", help="Path to the document to scan")
args = vars(ap.parse_args())

HEIGHT=500

image = cv2.imread(args["image"])
ratio = image.shape[0]/HEIGHT
orig = image.copy()
image = imutils.resize(image, height =HEIGHT)

#detect edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 75, 200)

#show image 1.- Edge detection
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


#find contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

    cv2.drawContours(image, [approx], -1, (255, 0,0), 2)

    if len(approx) ==4:
        screenCnt = approx
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

cv2.imshow("Outline", image)
cv2.waitKey(0) 
cv2.destroyAllWindows()


#apply transform
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

cv2.imshow("Origin", imutils.resize(orig, height=650))
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
cv2.destroyAllWindows()

