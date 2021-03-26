from torchvision import transforms
from ssd_utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import copy
import cv2

class SSDDetector(object):
    def __init__(self, device, checkpoint):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model checkpoint
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        self.model = checkpoint['model']
        self.model = self.model.to(device)
        self.model.eval()

        # Transforms
        self.resize = transforms.Resize((300, 300))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])


    def detect(self, original_image, min_score, max_overlap, top_k, suppress=None):
        """
        Detect objects in an image with a trained SSD300, and visualize the results.

        :param original_image: image, a PIL Image
        :param min_score: minimum threshold for a detected box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
        :return: annotated image, a PIL Image
        """

        # Transform
        image = self.normalize(self.to_tensor(self.resize(original_image)))

        # Move to default device
        image = image.to(self.device)

        # Forward prop.
        predicted_locs, predicted_scores = self.model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs, predicted_scores,
                                                                      min_score=min_score,
                                                                      max_overlap=max_overlap, top_k=top_k)
        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor(
            [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in ssd_model.py
        if det_labels == ['background']:
            # Just return original image
            return original_image

        # Annotate
        annotated_image = original_image
        (W, H) = annotated_image.size
        # draw = ImageDraw.Draw(annotated_image)
        # fontFile = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
        # font = ImageFont.truetype(fontFile, 9)
        # font = ImageFont.load_default()

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            # draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            # draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            #     det_labels[i]])

            # annotated_image.setflags(write=1)

            x1 = int(box_location[0]) if int(box_location[0]) >= 0 else 0
            y1 = int(box_location[1]) if int(box_location[1]) >= 0 else 0
            x2 = int(box_location[2]) if int(box_location[2]) < W else W - 1
            y2 = int(box_location[3]) if int(box_location[3]) < H else H - 1

            temp = np.asarray(annotated_image)
            temp = copy.deepcopy(temp)

            cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "{}".format(det_labels[i])
            cv2.putText(temp, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            temp[y1: y2, x1: x2, :] = cv2.GaussianBlur(temp[y1: y2, x1: x2, :], (21, 21), 0)
            annotated_image = Image.fromarray(temp)

            # Text
            # text_size = font.getsize(det_labels[i].upper())
            # text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            # textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
            #                     box_location[1]]
            # draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            # draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
            #           font=font)

        # del draw

        return annotated_image


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = '/home/ywlee/PycharmProjects/checkpoint_ssd300_285_best(2.2070).pth'

    detector = SSDDetector(device, checkpoint)

    img_path = '/home/ywlee/PycharmProjects/AINetworks/data/TestCar.jpg'
    original_image = Image.open(img_path, mode='r')
    original_image = original_image.convert('RGB')
    detector.detect(original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
