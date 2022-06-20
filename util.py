from pathlib import Path
from random import shuffle
import pandas as pd
import torch
import numpy as np
# import torch.utils.data.Subset
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset


class IamgesWithLabelsDataset(Dataset):
    """
    Given each of image the corresponding 
    labels into three class(Fractured, Intact, Shattered)
    """

    def __init__(self, root, transform=None) -> None:
        super().__init__()
        self.label2index = {'Fractured': 0, 'Intact': 1, 'Shattered': 2}  

        self.transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # Standardization
        ])

        root = '/content/drive/MyDrive/GoldSpot_Challenge_Waterloo_Dataset/'

        self.root = root

        images_path = Path(root + 'images/')

        images_list = list(
            images_path.glob('*.png'))  # Get the path of each of images
        images_list_str = [str(x) for x in images_list]  
        images_list_str.sort(
            key=lambda item: int(item[len(root + 'images/') - 0:-4]))
        self.images = images_list_str


    def __getitem__(self, item):
        """
        Arg: # of image
        Return:
            frames:shape->[patches_size,2*step,channel,width,height]
            label:shape->[patches_size,....]
        """
        image_path = self.images[item]

        labels_path = self.root + 'labels/' + str(item) + '.csv'   
        data = pd.read_csv(labels_path, header=0)
        id = data.loc[0, 'image_id']  
        label = data.loc[:, 'class']  
        for i in range(len(label)):
            label[i] = self.label2index[label[i]]   #convert strings in class to number
        label = torch.tensor(label)

        # Segmentation
        start_pixel = data.loc[:, 'start_pixel']
        end_pixel = data.loc[:, 'end_pixel']

        image = Image.open(image_path)  # Read RGB images
        width, height = image.size

        # Segmentation and resize the images
        image_patches = [
            image.crop(box=(start, 0, end, height)).resize((128, 171)) 
            for start, end in zip(start_pixel, end_pixel)   
        ]

        image_patches = [
            self.transform(_image_patch) for _image_patch in image_patches   
        ]  

        frames = []

        step = 1
        image_patches = [image_patches[0]] * step + image_patches + [image_patches[-1]] * step 

        for i in range(step, len(image_patches) - step):
            frame = image_patches[i - step:i + step + 1]   #There are 2*step+1 element in list
            frame = np.array([item.numpy() for item in frame])
            frames.append(frame)  
        frames = np.array(frames)

        return frames, label

    def __len__(self):
        return len(self.images)

def onehot2label(onehot):
    table = {
        0: 'Fractured',
        1: 'Intact',
        2: 'Shattered',
    }

    # onehot = onehot[0]
    index = onehot.item()
    return table[index]

class Data(Dataset):
    """
    Converting to Iterative ZIP file
    """

    def __init__(self, data) -> None:
        super().__init__()
        self.data = list(data)
        self.data = [list(item) for item in self.data]

    def add(self, frame, label):
        self.data.append((frame, label))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def split_train_test(data):
    dataset_frame = []
    dataset_label = []
    for frames, labels in data:
        for frame, label in zip(frames, labels):
            dataset_frame.append(frame)
            dataset_label.append(label)

    # dataset_frame = np.array(dataset_frame)
    # dataset_label = np.array(dataset_label)
    # dataset = np.concatenate([dataset_frame, dataset_frame], axis=1)
    train_size = int(0.7 * len(dataset_frame))

    # Create train/test dataset
    train = list(zip(dataset_frame[:train_size], dataset_label[:train_size]))
    test = Data(zip(dataset_frame[train_size:], dataset_label[train_size:]))

    fractured_num = 0
    shattered_num = 0
    for i in dataset_label[:train_size]:
        if (i == 0): fractured_num += 1
        elif (i == 2): shattered_num += 1

    fractured_weight = train_size / fractured_num
    shattered_weight = train_size / shattered_num
    weights = [fractured_weight, shattered_weight]
    looptrain = [item for item in train]  
    for (frame, label) in looptrain:
        for _ in range(
                1,
                int(weights[0 if 'Fractured' == onehot2label(label) else 1])):
            train.append((frame, label))
            
    # random shuffle
    shuffle(train)
    train = Data(train)

    return train, test
