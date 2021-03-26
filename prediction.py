import os
import numpy as np
import argparse
import cv2
from tqdm import tqdm
import torch
from PIL import Image
from detect import SSDDetector
from mrcnn import get_instance_segmentation_model
from torchvision.transforms import functional as F
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-m", "--model", type=str, choices=['yolo', 'ssd', 'mrcnn'], required=True,
	help="model")
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


if args['model'] == 'ssd':
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

			img = detector.detect(img, min_score=0.3, max_overlap=0.5, top_k=200)

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


elif args['model'] == 'yolo':
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


elif args['model'] == 'mrcnn':
	name_list = ['licence', 'face']

	device = torch.device('cpu')
	if args["use_gpu"] and torch.cuda.is_available():
		device = torch.device('cuda')

	net_model = get_instance_segmentation_model(num_classes=3, pretrainded=False).to(device)
	net_model.load_state_dict(torch.load(args['weight']))
	net_model.eval()

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

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			img = Image.fromarray(frame)
			img = torch.unsqueeze(F.to_tensor(img), 0).to(device)

			net_output = net_model(img)

			confidences = torch.where(net_output[0]['scores'][net_output[0]['scores'] > args['confidence']])
			boxes = net_output[0]['boxes'][confidences].cpu().detach().numpy()
			classIDs = net_output[0]['labels'][confidences].cpu().detach().numpy()

			boxes = np.uint64(boxes)
			confidences = confidences[0].cpu().detach().numpy()

			annotated_image = copy.deepcopy(frame)
			annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

			for i, box in enumerate(boxes):
				cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
				text = "{}".format(name_list[classIDs[i] - 1])
				cv2.putText(annotated_image, text, (box[0], int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				annotated_image[box[1]: box[3], box[0]: box[2], :] = cv2.GaussianBlur(annotated_image[box[1]: box[3], box[0]: box[2], :], (21, 21), 0)

			if args["output"] != "" and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

			if writer is not None:
				writer.write(frame)

			count += 1
			t.update(1)

else:
	print('올바른 config, weight 파일이 아닙니다.')
	exit(-1)
