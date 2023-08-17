import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils import *
import random
from matplotlib import pyplot as plt
from config import *
import copy

def visualize_augmentations(dataset, samples=10, cols=5):
    """Visualizes sample images after being augmented.

    Args:
        dataset (Dataset):
        samples (int, optional): Number of images to be visualized. Defaults to 10.
        cols (int, optional): Number of columns of output image. Defaults to 5.
    """
    dataset = copy.deepcopy(dataset)
    # Retain numpy format.
    vis_aug = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    idx = random.randint(0, len(dataset))
    original_image = cv2.imread(os.path.join(CLS_DATA_DIR, dataset.data_type, str(idx).zfill(5)+'.png'))
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(64, 32))
    for i in range(samples):
        image = vis_aug(image=original_image)['image']
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    fig.savefig(FIG_DIR + '/augmentation.png')

def get_transform(data_type, input_size, seq_len=1):
    """Returns appropriate transformations according to input data type.

    Args:
        data_type (str): One of train/val/test
        input_size (int): Input size of the model
        seq_len (int, optional): RNN sequence length in case of CNN+RNN model. Defaults to 1.

    Returns:
        Transformations:
    """
    max_size = int(input_size * 1.054)

    additional_targets = {}
    for i in range(1, seq_len):
        additional_targets[f'image{str(i)}'] = 'image'

    # data_type = 'val'
    if data_type == 'train':
        return A.Compose(
                [
                    A.Resize(max_size, max_size),
                    A.RandomResizedCrop(input_size, input_size, scale=(0.8, 1)),
                    A.HorizontalFlip(p=0.3),
                    A.GaussNoise(p=0.5),
                    A.OneOf([
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(p=0.1),
                        A.Blur(p=0.1)
                    ], p=0.5),
                    A.ShiftScaleRotate(rotate_limit=15),
                    A.OneOf([
                        A.GridDistortion(),
                        A.ElasticTransform(),
                        A.OpticalDistortion()
                    ], p=0.5),
                    A.RandomGamma(),
                    A.RandomBrightnessContrast(),
                    A.Normalize(mean=0.334, std=0.164),
                    ToTensorV2()
                ], additional_targets=additional_targets
            )
    else: # val / test
        return A.Compose(
                [
                    A.Resize(max_size, max_size),
                    A.CenterCrop(input_size, input_size),
                    A.Normalize(mean=0.334, std=0.164),
                    ToTensorV2()
                ], additional_targets=additional_targets
            )

# train/val (mean, std) = (0.3245, 0.1567) / (0.381, 0.202)

class ClsDataset(Dataset):
    def __init__(self, data_type, input_size, seq_len=1, contrast_learn=False):
        # All variables are changed to numpy not to get duplicated in multiprocessing
        self.data_type = data_type
        self.input_size = input_size
        self.seq_len = seq_len
        self.labels = np.load(os.path.join(CLS_DATA_DIR, f'{data_type}_label.npy')).astype(np.float32)[seq_len-1:]
        self.transform = get_transform(data_type, input_size, seq_len)
        self.contrast_learn = contrast_learn

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        if not self.contrast_learn:
            targets = {'image' : self._load_image(idx)}
            for i in range(1, self.seq_len):
                targets[f'image{str(i)}'] = self._load_image(idx+i)
            images = torch.stack(list(self.transform(**targets).values()), dim=0)
            return images, self.labels[idx]
        else: # Contrastive Learning
            img = self._load_image(idx)
            img1 = self.transform(image=img)['image']
            img2 = self.transform(image=img)['image']
            return torch.stack([img1, img2], dim=0), self.labels[idx]

    def _load_image(self, idx):
        return cv2.imread(os.path.join(CLS_DATA_DIR, self.data_type, str(idx).zfill(5)+'.png'), 0) # Grayscale image

    def pos_weight(self):
        """Returns ratio of classes for imbalanced classification

        Returns:
            float: ratio of sub-optimal to optimal.
        """
        return self.labels[self.labels==0].size / self.labels[self.labels==1].size


class SegDataset(Dataset):
    def __init__(self, data_type, input_size):
        self.data_type = data_type
        self.root = os.path.join(SEG_DATA_DIR, data_type)
        self.mask_path = os.path.join(self.root, 'mask')
        self.img_path = os.path.join(self.root, 'frame')
        self.objects = os.listdir(self.mask_path)
        self.transform = get_transform(data_type, input_size)
                
    def __len__(self):
        return len(os.listdir(self.img_path))

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, str(idx).zfill(3)+'.png'), 0)
        masks = []
        for obj in self.objects:
            f = os.path.join(self.mask_path, obj, str(idx).zfill(3)+'.png')
            if os.path.exists(f):
                mask = cv2.imread(f, 0)
                mask[mask!=0] = 1
            else:
                mask = np.zeros_like(img, dtype=np.uint8)
            masks.append(mask)

        transformed = self.transform(image=img, masks=masks)
        img = transformed['image']
        masks = transformed['masks']

        return img, masks

class VideoDataset(Dataset):
    """Dataset for realtime applications.

    Args:
        Dataset (Dataset):
    """
    def __init__(self, frames, mask, transform):
        # All variables changed to numpy not to get duplicated in multiprocessing
        self.frames = frames
        self.transform = transform
        self.mask = mask

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img, mean, std, bbox = crop(self.frames[idx], self.mask)
        return self.transform(image=img)['image'], np.array(bbox)