from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
 
def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def imshow(image, title="Title"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='images/example_01.png',
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, default=24.26,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
imshow(image)
copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
imshow(gray)

edged = cv2.Canny(gray, 50, 100)
imshow(edged)
edged = cv2.dilate(edged, None, iterations=1)
imshow(edged)
edged = cv2.erode(edged, None, iterations=1)
imshow(edged)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1]

(cnts, _) = contours.sort_contours(cnts)


pixelsPerMetric=None
for c in cnts:
    if cv2.contourArea(c) <100:
        continue

    box = cv2.minAreaRect(c)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)

    cv2.drawContours(copy, [box.astype("int")], -1, (0,255,0), 2)

    # perimeter = cv2.arcLength(c, True)
    # approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    # cv2.drawContours(copy, [approx], -1, (0, 255, 0), 2)
    for (x, y) in box:
        cv2.circle(copy, (int(x), int(y)), 5, (0,0,255), -1)

    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    cv2.circle(copy, (int(tltrX), int(tltrY)), 5, (255,0,0), -1)
    cv2.circle(copy, (int(blbrX), int(blbrY)), 5, (255,0,0), -1)
    cv2.circle(copy, (int(tlblX), int(tlblY)), 5, (255,0,0), -1)
    cv2.circle(copy, (int(trbrX), int(trbrY)), 5, (255,0,0), -1)

    cv2.line(copy, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
    cv2.line(copy, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
 
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
 
	# draw the object sizes on the image
    cv2.putText(copy, "{:.1f}mm".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
    cv2.putText(copy, "{:.1f}mm".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

    imshow(copy)
