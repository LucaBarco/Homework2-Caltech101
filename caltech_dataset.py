from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
from typing import List, Tuple

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

'''
class Caltech(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        ''''''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        ''''''
        ''''''
        we can create a list of (id, image, idLabel)
        we can create a map of (idLabel, label)
        ''''''
        file = open('Caltech101/' + split + '.txt', "r")
        self.images = {}
        self.labels = {}
        self.labels_of_images = {}
        self.indexes_for_class = {}
        lines = file.readlines()
        self.count_images = 0
        self.count_labels = 0
        for line in lines:
            if "BACKGROUND_Google" not in line:
                label_name = line.split('/')[0]

                if label_name not in self.labels.keys():
                    self.labels[label_name] = self.count_labels
                    self.count_labels = self.count_labels + 1

                if label_name not in self.indexes_for_class.keys():
                    self.indexes_for_class[label_name] = [self.count_images]
                else:
                    self.indexes_for_class[label_name].append(self.count_images)
                line=line.strip('\n')
                self.images[self.count_images] = pil_loader(root + '/' + line)
                self.labels_of_images[self.count_images] = self.labels[label_name]
                self.count_images = self.count_images + 1
        print(split+" Ho contato %d immagini"%self.count_images)

    def __getitem__(self, index):
        '''
   #     __getitem__ should access an element through its index
    #    Args:
     #       index (int): Index

      #  Returns:
       #     tuple: (sample, target) where target is class_index of the target class.
'''# Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int
        if index <self.count_images:
            image = self.images[index]
            label= self.labels_of_images[index]
            # Applies preprocessing when accessing the image
            if self.transform is not None:
                image = self.transform(image)
        else:
            image , label=None,None

        return image, label

    def __len__(self):
        ''''''
      #  The __len__ method returns the length of the dataset
       # It is mandatory, as this is used by several other components
        ''''''
        length=0
        if self.images :
            length = self.count_images  # Provide a way to get the length (number of elements) of the dataset
        return length

    def split_training_set(self):
        training_indexes=[]
        validation_indexes=[]

        #Let's start putting an half of each class
        for c in self.indexes_for_class.keys():
            for i in range(0,len(self.indexes_for_class[c])):
                if i % 2:
                    training_indexes.append(self.indexes_for_class[c][i])
                else:
                    validation_indexes.append(self.indexes_for_class[c][i])


        return (training_indexes, validation_indexes)
'''

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

        _dataset: List[Tuple[Image, int]] = []
        if split in ("train", "test"):
            with open(f"./{split}.txt", 'r') as f:
                classes = []
                for path in f.readlines():
                    path = path.strip("\n")
                    clazz = path.split("/")[0]
                    if clazz != "BACKGROUND_Google":
                        if clazz not in classes:
                            classes.append(clazz)
                        _dataset.append((pil_loader(os.path.join(root, path)), classes.index(clazz)))
        self.dataset = _dataset

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        if index < len(self.dataset):
            image, label = self.dataset[index]
        else:
            image, label = None, None
                           # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        # Provide a way to get the length (number of elements) of the dataset
        return len(self.dataset) if self.dataset else 0
