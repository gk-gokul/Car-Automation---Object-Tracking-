
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

from os.path import dirname, join



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "tree", "sheep",
	"sofa", "train", "tvmonitor"]

TARGET_CLASSES = [6, 7, 15, 16, 17]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

vs = cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# print(frame)
	#print("------------------")
	ret,frame = vs.read()
	#print(type(frame[1]))
	frame = imutils.resize(frame, width=400)

	# grab the frame dimensions and convert it to a blob
	
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)


	net.setInput(blob)
	detections = net.forward()
	# loop over the detections
	for i in np.arange(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]
		
		if confidence > 0.95:

			idx = int(detections[0, 0, i, 1])
			print("------------------"+CLASSES[idx])
			print("------------------", idx)

			if(idx in TARGET_CLASSES):
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[idx],
					confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()