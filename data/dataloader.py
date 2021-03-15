from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

# Please note the sources of data used here are:
# MNIST: https://www.kaggle.com/c/digit-recognizer
# LFW: https://www.kaggle.com/jessicali9530/lfw-dataset

def get_datasets(train_csv_path, test_csv_path = None):
    train = pd.read_csv(train_csv_path)
    # test = pd.read_csv(test_csv_path)

    train, valid = train_test_split(train, test_size = 0.2, random_state = 42)

    Y_train = train["label"].values
    Y_valid = valid["label"].values
    # Y_test = test["label"]
    
    X_train = train.drop(labels = ["label"],axis = 1)
    X_valid = valid.drop(labels = ["label"],axis = 1)
    X_train = X_train.values.reshape(-1,28,28,1).astype(np.uint8)
    # X_train = np.concatenate([X_train, X_train, X_train], axis = -1)
    X_valid = X_valid.values.reshape(-1,28,28,1).astype(np.uint8)
    # X_valid = np.concatenate([X_valid, X_valid, X_valid], axis = -1)

    train_set = MNISTDataset(X_train, Y_train, type = 'train')
    valid_set = MNISTDataset(X_valid, Y_valid, type = 'valid')

    return train_set, valid_set

class MNISTDataset(Dataset):
    def __init__(self, imgs, labels, type = 'train'):
        """
        imgs: images (np.array of shape (N, 28, 28, 1))
        labels: image labels (np.array of shape (N,))
        """

        self.imgs = imgs
        self.labels = labels
        self.type = type

        if (type == 'train'):
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomAffine(20, translate=[0.05, 0.25], scale=[0.1, 0.2]),
                transforms.ToTensor() #,
                # transforms.Normalize((0.5,), (0.5,))
            ])
        else: # valid or test
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor() #,
                # transforms.Normalize((0.5,), (0.5,))
            ])
    
    def __getitem__(self, idx):
        
        img = self.imgs[idx, :, :, :]
        label = self.labels[idx]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return self.labels.shape[0]


def get_lfw_datasets(img_path):

    faces = []

    for path in Path(img_path).rglob('*.jpg'):
        faces.append(path)

    train, valid = train_test_split(faces, test_size = 0.2, random_state = 42)

    train_set = LFWDataset(train, type = 'train')
    valid_set = LFWDataset(valid, type = 'valid')

    return train_set, valid_set

class LFWDataset(Dataset):
    def __init__(self, faces, type = 'train'):
        """
        faces = list of path to photos
        """

        self.faces = faces
        self.type = type

        if (type == 'train'):
            self.transforms = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.25),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # transforms.RandomAffine(20, translate=[0.05, 0.25], scale=[0.1, 0.2]),
                transforms.ToTensor() #,
                # transforms.Normalize((0.5,), (0.5,))
            ])
        else: # valid or test
            self.transforms = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor() #,
                # transforms.Normalize((0.5,), (0.5,))
            ])
    
    def __getitem__(self, idx):

        im = Image.open(self.faces[idx])
        # im = im.resize((28,28))
        im = self.transforms(im)

        return im, -1

    def __len__(self):
        return len(self.faces)
