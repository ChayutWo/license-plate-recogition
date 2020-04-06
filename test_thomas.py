from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.patches as patches
# Ignore warnings
import warnings
import cv2
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

import os
import os.path
import numpy as np
import pandas as pd
import re
from utils_thomas import dataloader, transform


def show_landmarks(image, box):
    ### Function for checking whether landmarks are correct.
    """Show image with landmarks"""
    print (box)
    plt.imshow(image)
    plt.scatter(box[0], box[1], s=10, marker='.', c='r')
    # plt.scatter(landmarks[1,0],landmarks[1,1], s=10, marker='.', c='r')
    # plt.scatter(landmarks[2,0], landmarks[2,1], s=10, marker='.', c='r')
    # plt.scatter(landmarks[3,0],landmarks[3,1], s=10, marker='.', c='r')
    #patches.Rectangle((landmarks[0], landmarks[1]),landmarks[2],landmarks[3],linewidth=1,color='red',fill=False)
    plt.pause(100)  # pause a bit so that plots are updated

transform.rotate(10)
face_dataset = dataloader.LicenseLandmarksDataset(csv_file='directory_training.csv',\
root_dir='dataset/UFPR-ALPR dataset/training',transform=transform.rotate(10))

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, len(sample['box']))

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break

# rotat = transform.rotate(10)
# flip = transform.HorizontalFlip()
# fig = plt.figure()
# #sample = face_dataset[45]
# for k in range(len(face_dataset)):
#     sample = face_dataset[k]
#     for i, tsfrm in enumerate([rotat,flip]):
#         transformed_sample = tsfrm(sample)
#
#         ax = plt.subplot(1, 3, i + 1)
#         plt.tight_layout()
#         ax.set_title(type(tsfrm).__name__)
#         show_landmarks(**transformed_sample)
#     plt.show()
#     if k ==2:
#         break
