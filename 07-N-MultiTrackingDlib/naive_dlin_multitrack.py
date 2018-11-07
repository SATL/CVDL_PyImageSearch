from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd/MobileNetSSD_deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd/MobileNetSSD_deploy.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", default="race.mp4" , help="path to input video file")
ap.add_argument("-o", "--output", default="race_slow.mp4", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

print("[info] loading the model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[info[ starting video stream")
vs = cv2.VideoCapture(args["video"])
writer = None

trackers = []
labels= []

fps = FPS().start()

while True:
    (grabbed, frame) = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=600)
    #change to rbg for dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

    if len(trackers) ==0:
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w,h), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0,0,i,2]

            if confidence > args["confidence"]:

                idx = int(detections[0,0,i,1])
                label = CLASSES[idx]

                if label != "person":
                    continue

                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")

                # construct a dlib rectangle object from the bounding
				# box coordinates and start the correlation tracker
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                t.start_track(rgb, rect)
 
				# update our set of trackers and corresponding class
				# labels
                labels.append(label)
                trackers.append(t)
 
				# grab the corresponding class label for the detection
				# and draw the bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            # otherwise, we've already performed detection so let's track
	# multiple objects
    else:
		# loop over each of the trackers
        for (t, l) in zip(trackers, labels):
			# update the tracker and grab the position of the tracked
			# object
            t.update(rgb)
            pos = t.get_position()
 
			# unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
 
			# draw the bounding box from the correlation object tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, l, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    
    if writer is not None:
        writer.write(frame)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

	# update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()

