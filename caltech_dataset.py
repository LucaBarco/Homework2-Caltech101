from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    images = {}
    labels = {}
    indexes_for_class={}
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        '''
        we can create a list of (id, image, idLabel)
        we can create a map of (idLabel, label)
        '''
        file = open('Caltech101/' + split + '.txt', "r")

        lines = file.readlines()
        count_images = 0
        count_labels = 0
        for line in lines:
            if "BACKGROUND_Google" not in line:
                label_name = line.split('/')[0]

                if label_name not in self.labels.keys():
                    self.labels[label_name] = count_labels
                    count_labels = count_labels + 1

                if label_name not in self.indexes_for_class.keys():
                    self.indexes_for_class[label_name] = [count_images]
                else:
                    self.indexes_for_class[label_name].append(count_images)

                self.images[count_images] = (pil_loader(root + '/' + line[:-1]), self.labels[label_name])
                count_images = count_images + 1


    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        # Provide a way to access image and label via index
        # Image should be a PIL Image
        # label can be int
        image, label = self.images[index]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.images)  # Provide a way to get the length (number of elements) of the dataset
        return length

    def split_training_set(self):
        training_indexes=[]
        validation_indexes=[]

        #Let's start putting an half of each class
        for c in self.indexes_for_class.keys():
            for i in range(0,len(self.indexes_for_class[c])):
                if i%2:
                    training_indexes.append(self.indexes_for_class[c][i])
                else:
                    validation_indexes.append(self.indexes_for_class[c][i])


        return (training_indexes, validation_indexes)