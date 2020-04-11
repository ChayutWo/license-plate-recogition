# Import required package
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.utils.data as data
import os
import os.path
from torchsummary import summary
import csv
from models.generate import *

from utils.dataloader import *
from utils.optimizer import *
from utils.train_test_step import *
from utils.transform import *
from utils.plot_result import *
from utils.confusion_matrix import *
from utils.iou import *
from utils.variables import *
####################################################################
model_path = model_name + '.pth'
model_path ='training_model.pth'
model = create_model(model_name, in_channels, out_channels)
model.load_state_dict(torch.load(model_path))
model = model.to(device)



composed_test = transforms.Compose([PILconvert(), resize(), tensorize()])

traindata = LicenseLandmarksDataset(train_csv, train_path, composed_test)
testdata = LicenseLandmarksDataset(test_csv, test_path, composed_test)

test_loader = torch.utils.data.DataLoader(testdata,
                                          batch_size=test_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)

output = iou_from_model(test_loader, model, device)
print(np.mean(output))
plt.hist(output, bins = 30)
plt.show()
with open('iou.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(output)

plot_iou(test_loader, model, device)