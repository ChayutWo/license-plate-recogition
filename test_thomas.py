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
import os
import os.path
import numpy as np
import pandas as pd
import re
from utils import dataloader, transform


composed = transforms.Compose([transform.HorizontalFlip(p=0.5),
                               transform.rotate(Maxangle=10),
                               transform.PILconvert(),
                               transform.ColorJitter(brightness=0.1, contrast=0.1,
                                                     saturation=0.1, hue=0.1),
                               transform.resize()])
composed = transforms.Compose([transform.PILconvert(), transform.resize()])
face_dataset = dataloader.LicenseLandmarksDataset(csv_file='directory_training.csv',
                                                  root_dir='dataset/resized/UFPR-ALPR dataset/training',transform=composed)

fig = plt.figure()
i = 200
sample = face_dataset[i]
plt.tight_layout()
plt.imshow(sample['image'])
plt.scatter(sample['box'][0], sample['box'][1], s=10, marker='.', c='r')
plt.show()
