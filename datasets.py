import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from ssd_utils import transform, resize
import numpy as np
import cv2
from torchvision.transforms import functional as F

class AINetDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, model_name='mrcnn', keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        assert model_name == 'ssd' or model_name == 'mrcnn'
        self.max_size = 20000
        self.split = split.upper()
        self.model_name = model_name

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, idx):
        # Read image
        try:
            image = Image.open(self.images[idx], mode='r')
        except Exception as e:
            print(e)

        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[idx]

        if len(objects['labels']) > 5 or len(objects['labels']) == 0:
            return self.__getitem__(idx + 1)

        temp = np.array(objects['boxes'])
        temp[temp < 0] = 0
        objects['boxes'] = temp.tolist()
        boxes = torch.as_tensor(objects['boxes'], dtype=torch.float32)  # (n_objects, 4)
        labels = torch.as_tensor(objects['labels'], dtype=torch.int64)  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        if self.model_name == 'ssd':
            # Apply transformations
            image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
            return image, boxes, labels, difficulties

        elif self.model_name == 'mrcnn':
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            new_size = image.size
            new_boxes = objects['boxes']
            old_dims = np.array([new_size[0], new_size[1], new_size[0], new_size[1]])

            if new_size[0] > self.max_size or new_size[1] > self.max_size:
                ratio = self.max_size / new_size[0]
                temp = (self.max_size, int(new_size[1] * ratio))

                if new_size[1] > new_size[0]:
                    ratio = self.max_size / new_size[1]
                    temp = (int(new_size[0] * ratio), self.max_size)

                new_size = temp
                new_dims = np.array([new_size[0], new_size[1], new_size[0], new_size[1]])
                box_ratio = new_boxes / old_dims
                new_boxes = np.int32(new_dims * box_ratio)

                image = image.resize(new_size)
                boxes = torch.as_tensor(new_boxes, dtype=torch.float32)

            masks = np.zeros(shape=(new_size[1], new_size[0]), dtype="uint8")

            num_objs = 0
            for i, bbox in enumerate(new_boxes):
                # if bbox[3] < bbox[1] or bbox[2] < bbox[0] or bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= new_size[0] or bbox[3] >= new_size[1]:
                #     return self.__getitem__(idx + 1)

                masks[bbox[1] : bbox[3], bbox[0] : bbox[2]] = i + 1
                num_objs = i + 1

            # num_objs, masks = cv2.connectedComponents(mask)
            # if num_objs <= 1:
            #     return self.__getitem__(idx + 1)

            obj_ids = np.unique(masks)
            obj_ids = obj_ids[1:]
            # cv2.imwrite('test_mask.png', masks * 255)
            masks = masks == obj_ids[:, None, None]
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = torch.tensor([idx])
            target["area"] = areas
            target["iscrowd"] = torch.zeros(num_objs, dtype=torch.int64)
            target["file"] = self.images[idx]

            image = F.to_tensor(image)

            return image, target

    def __len__(self):
        return len(self.images)


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
