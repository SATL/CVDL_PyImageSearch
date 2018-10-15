#Imports
import imutils
import cv2

#Constants
PATH = '/Users/slem/Code/Learn/DataScience/CVDL_PyImageSearch/02-OpenCVBasics' 
IMG = '/image.jpg'

#Open image
image = cv2.imread(PATH+IMG)
(h,w,d) = image.shape
print("width={}, height={}, depth={}".format(w,h,d))
cv2.imshow("Image", image)
cv2.waitKey(0)

#Access pixel
(B, G, R) = image[100, 50]
print("R={} G={} B={}".format(R,G,B))

#Extracting ROI
roi = image[50:388,  148:450]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

#Resize image
scale = 0.34
resized = cv2.resize(image, (int(scale*w), int(scale*h)))
cv2.imshow("Resizing", resized)
cv2.waitKey(0)

#Resize using imutils
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils resized", resized)
cv2.waitKey(0)

#Rotating image
center = (w//2, h//2)
rotationMatrix = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, rotationMatrix, (w,h))
cv2.imshow("Rotate image", rotated)
cv2.waitKey(0)

#Imutils rotation
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

#Rotate no bound imutils
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils bound rotation", rotated )
cv2.waitKey(0)

def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)

#Gaussian blur
blurred = cv2.GaussianBlur(image, (11,11), 0)
show_image(blurred)

#Draw in the image
#Square
output = image.copy()
cv2.rectangle(output, (50, 140), (388,450), (0,0,255), 2)
show_image(output)

#Circle
output = image.copy()
cv2.circle(output, (200, 200), 200, (255,0,0), -1)
show_image(output)

#Line
output = image.copy()
cv2.line(output, (50, 140), (388,450), (0,0,255), 5)
show_image(output)

#Text image
output = image.copy()
cv2.putText(output, "OpenCV", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
show_image(output)