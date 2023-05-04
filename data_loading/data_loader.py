import os
import numpy as np
import pandas as pd
from PIL import Image

import torch.onnx
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset


class CombineDataset(Dataset):
    def __init__(self, frame, id_col, label, img_dir, input_size, transform=None):
        """
        Args:
            frame (pd.DataFrame): Frame with the tabular data.
            id_col (string): Name of the column that connects image to tabular data
            label_name (string): Name of the column with the label to be predicted
            path_imgs (string): path to the folder where the images are.
            transform (callable, optional): Optional transform to be applied
                on a sample, you need to implement a transform to use this.
        """
        self.frame = frame
        self.id_col = id_col
        self.label = label
        self.img_dir = img_dir
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return (self.frame.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        img_name = self.frame[self.id_col].iloc[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)
        image = resize_img(image, self.input_size)

        if self.transform:
            image = self.transform(image)

        # get features
        feats = [feat for feat in self.frame.columns if feat not in [self.label, self.id_col]]
        feats = np.array(self.frame[feats].iloc[idx])
        feats = torch.from_numpy(feats.astype(np.float32))

        # get label
        label = np.array(self.frame[self.label].iloc[idx])
        label = torch.from_numpy(label.astype(np.int64))

        return feats, image, label


def resize_img(image, input_size):
    base_mask = np.zeros((input_size[1], input_size[0], 3), np.uint8)
    h, w = np.array(image).shape[:2]

    ash = input_size[1] / h
    asw = input_size[0] / w

    if asw < ash:
        sizeas = (int(w * asw), int(h * asw))
    else:
        sizeas = (int(w * ash), int(h * ash))

    img_resized = image.resize(sizeas)
    base_mask[int(input_size[1]/2 - sizeas[1]/2):int(input_size[1]/2 + sizeas[1]/2),
              int(input_size[0]/2 - sizeas[0]/2):int(input_size[0]/2 + sizeas[0]/2), :] = img_resized
    return base_mask
