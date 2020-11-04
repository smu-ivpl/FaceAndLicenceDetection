import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from detect import SSDDetector

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-c", "--config", type=str, default=None,
	help="path to model configuration file")
ap.add_argument("-w", "--weight", type=str, required=True, default=None,
	help="path to model trained weights file")
ap.add_argument("--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-u", "--use-gpu", type=bool, default=True,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())


def Blurring(frame, boxes, confidences, idxs, classIDs, color=(0, 0, 255)):
	name_list = ['licence', 'face']

	if len(idxs) > 0:
		for i in idxs.flatten():
			class_name = name_list[classIDs[i]]

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(class_name,
									   confidences[i])
			cv2.putText(frame, text, (x, y - 5),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			frame[y: y + h, x: x + w, :] = cv2.GaussianBlur(frame[y: y + h, x: x + w, :], (21, 21), 0)

	return frame


mode = None
if args["config"] == None:
	if os.path.splitext(os.path.basename(args["weight"]))[1] == '.pth':
		mode = 'pytorch'
else:
	if os.path.splitext(os.path.basename(args["config"]))[1] == '.cfg' and \
			os.path.splitext(os.path.basename(args["weight"]))[1] == '.weights' :
		mode = 'darknet'

	if os.path.splitext(os.path.basename(args["config"]))[1] == '.pbtxt' and \
			os.path.splitext(os.path.basename(args["weight"]))[1] == '.pb' :
		mode = 'tensorflow'

assert mode != None

if mode == 'pytorch':
	device = torch.device('cpu')
	if args["use_gpu"] and torch.cuda.is_available():
		device = torch.device('cuda')
	detector = SSDDetector(device, args['weight'])

	vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
	writer = None
	count = 1

	W = None
	H = None

	with tqdm(total=int(vs.get(cv2.CAP_PROP_FRAME_COUNT))) as t:
		t.set_description('{}'.format(os.path.basename(args['input'])))

		while True:

			(grabbed, frame) = vs.read()

			if not grabbed:
				print("영상을 찾을 수 없습니다.")
				break

			if W is None or H is None:
				(H, W) = frame.shape[:2]

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(frame)

			img = detector.detect(img, min_score=args["confidence"], max_overlap=args["threshold"], top_k=200)

			frame = np.asarray(img)
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

			if args["output"] != "" and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
										 (frame.shape[1], frame.shape[0]), True)

			if writer is not None:
				writer.write(frame)

			count += 1
			t.update(1)

elif mode == 'darknet':
	net_model = cv2.dnn.readNetFromDarknet(args["config"], args["weight"])

	if args["use_gpu"]:
		print("[INFO] setting preferable backend and target to CUDA...")
		net_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

	ln = net_model.getLayerNames()
	ln = [ln[i[0] - 1] for i in net_model.getUnconnectedOutLayers()]

	W = None
	H = None

	vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
	writer = None
	count = 1

	with tqdm(total=int(vs.get(cv2.CAP_PROP_FRAME_COUNT))) as t:
		t.set_description('{}'.format(os.path.basename(args['input'])))

		while True:
			(grabbed, frame) = vs.read()

			if not grabbed:
				break

			if W is None or H is None:
				(H, W) = frame.shape[:2]

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
			net_model.setInput(blob)

			net_output = net_model.forward(ln)

			boxes = []
			confidences = []
			classIDs = []

			for output in net_output:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					if confidence > args["confidence"]:

						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
				args["threshold"])

			frame = Blurring(frame, boxes, confidences, idxs, classIDs)

			if args["output"] != "" and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 30,
										 (frame.shape[1], frame.shape[0]), True)

			if writer is not None:
				writer.write(frame)

			count += 1
			t.update(1)

elif mode == 'tensorflow':
	pass

else:
	print('올바른 config, weight 파일이 아닙니다.')
	exit(-1)



# class Predictor(object):
# 	def __init__(self, args):
# 		self.vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
# 		self.writer = None
#
# 	def Blurring(self, boxes, confidences, idxs, color=(0,0,255), class_name=''):
# 		if len(idxs) > 0:
# 			for i in idxs.flatten():
# 				(x, y) = (boxes[i][0], boxes[i][1])
# 				(w, h) = (boxes[i][2], boxes[i][3])
#
# 				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
# 				text = "{}: {:.4f}".format(class_name,
# 										   confidences[i])
# 				cv2.putText(frame, text, (x, y - 5),
# 							cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#
# 				frame[y: y + h, x: x + w, :] = cv2.GaussianBlur(frame[y: y + h, x: x + w, :], (21, 21), 0)
#
# 	while True:
# 		(grabbed, frame) = vs.read()
#
# 		if not grabbed:
# 			print("영상을 찾을 수 없습니다.")
# 			break
#
# 		if W is None or H is None:
# 			(H, W) = frame.shape[:2]
#
# 		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
# 		net_model.setInput(blob)
#
# 		net_output = None
# 		if mode == 'pytorch':
# 			net_output = net_model.forward()
# 		elif mode == 'darknet':
# 			net_output = net_model.forward(ln)
#
# 		# initialize our lists of detected bounding boxes, confidences,
# 		# and class IDs, respectively
# 		boxes = []
# 		confidences = []
# 		classIDs = []
#
# 		# loop over each of the layer outputs
# 		for output in faces:
# 			# loop over each of the detections
# 			for detection in output:
# 				# extract the class ID and confidence (i.e., probability)
# 				# of the current object detection
# 				scores = detection[5:]
# 				classID = np.argmax(scores)
# 				confidence = scores[classID]
#
# 				# filter out weak predictions by ensuring the detected
# 				# probability is greater than the minimum probability
# 				if confidence > args["confidence"]:
# 					# scale the bounding box coordinates back relative to
# 					# the size of the image, keeping in mind that YOLO
# 					# actually returns the center (x, y)-coordinates of
# 					# the bounding box followed by the boxes' width and
# 					# height
# 					box = detection[0:4] * np.array([W, H, W, H])
# 					(centerX, centerY, width, height) = box.astype("int")
#
# 					# use the center (x, y)-coordinates to derive the top
# 					# and and left corner of the bounding box
# 					x = int(centerX - (width / 2))
# 					y = int(centerY - (height / 2))
#
# 					# update our list of bounding box coordinates,
# 					# confidences, and class IDs
# 					boxes.append([x, y, int(width), int(height)])
# 					confidences.append(float(confidence))
# 					classIDs.append(classID)
#
# 		# apply non-maxima suppression to suppress weak, overlapping
# 		# bounding boxes
# 		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
# 			args["threshold"])
#
# 		Blurring(boxes, confidences, idxs, class_name='Human Face')
#
# 		boxes = []
# 		confidences = []
# 		classIDs = []
#
# 		# loop over each of the layer outputs
# 		for output in plates:
# 			# loop over each of the detections
# 			for detection in output:
# 				# extract the class ID and confidence (i.e., probability)
# 				# of the current object detection
# 				scores = detection[5:]
# 				classID = np.argmax(scores)
# 				confidence = scores[classID]
#
# 				# filter out weak predictions by ensuring the detected
# 				# probability is greater than the minimum probability
# 				if confidence > args["confidence"]:
# 					# scale the bounding box coordinates back relative to
# 					# the size of the image, keeping in mind that YOLO
# 					# actually returns the center (x, y)-coordinates of
# 					# the bounding box followed by the boxes' width and
# 					# height
# 					box = detection[0:4] * np.array([W, H, W, H])
# 					(centerX, centerY, width, height) = box.astype("int")
#
# 					# use the center (x, y)-coordinates to derive the top
# 					# and and left corner of the bounding box
# 					x = int(centerX - (width / 2))
# 					y = int(centerY - (height / 2))
#
# 					# update our list of bounding box coordinates,
# 					# confidences, and class IDs
# 					boxes.append([x, y, int(width), int(height)])
# 					confidences.append(float(confidence))
# 					classIDs.append(classID)
#
# 		# apply non-maxima suppression to suppress weak, overlapping
# 		# bounding boxes
# 		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
# 								args["threshold"])
#
# 		Blurring(boxes, confidences, idxs, class_name='License Plate')
#
# 		# check to see if the output frame should be displayed to our
# 		# screen
# 		if args["display"] > 0:
# 			# show the output frame
# 			cv2.imshow("Frame", frame)
# 			key = cv2.waitKey(1) & 0xFF
#
# 			# if the `q` key was pressed, break from the loop
# 			if key == ord("q"):
# 				break
#
# 		# if an output video file path has been supplied and the video
# 		# writer has not been initialized, do so now
# 		if args["output"] != "" and writer is None:
# 			# initialize our video writer
# 			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# 			writer = cv2.VideoWriter(args["output"], fourcc, 30,
# 				(frame.shape[1], frame.shape[0]), True)
#
# 		# if the video writer is not None, write the frame to the output
# 		# video file
# 		if writer is not None:
# 			writer.write(frame)
#
# 		# update the FPS counter
# 		# fps.update()
# 		print(count)
# 		count += 1

	# stop the timer and display FPS information
	# fps.stop()
	#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))