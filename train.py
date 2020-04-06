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

from models.generate import *

from utils.dataloader import *
from utils.optimizer import *
from utils.train_test_step import *
from utils.transform import *
from utils.plot_result import *
from utils.confusion_matrix import *
####################################################################
"""
PART I: Construct train data generator and test data generator
"""

# Compose transformation for dataloader
# Reshape -> Separate Real and Imaginary Layer -> Convert to Tensor
composed = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomRotation(degrees=10),
                               torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                                  saturation=0.1, hue=0.1)])

# Train and test dataset using RadarDataset class
traindata = RadarDataset(train_csv, train_path, composed)
testdata = RadarDataset(test_csv, test_path, composed)

# Train and test dataloader setup
train_loader = torch.utils.data.DataLoader(traindata,
                                           batch_size=train_batch_size, shuffle=True,
                                           num_workers=8, pin_memory=True)

test_loader = torch.utils.data.DataLoader(testdata,
                                          batch_size=test_batch_size, shuffle=False,
                                          num_workers=8, pin_memory=True)

####################################################################
"""
PART II: Preparation for training
"""
torch.manual_seed(5)
torch.cuda.manual_seed(5)

# Create model and setup criterior, optimizer, and scheduler
model_CNN = create_model(model_name, in_channels, out_channels).to(device)

# weight
weight = torch.tensor([0.7, 1.5, 3.25, 3.25, 2.25], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weight)
#criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = make_optimizer(optimizer_name, model_CNN, lr=lr, momentum=0.9, weight_decay=0)
scheduler = make_scheduler(scheduler_name, optimizer, milestones=milestones, factor=0.5)

# training and testing loss
acc_0 = test(model_CNN, device, test_loader, criterion, 0)
acc = []
train_loss = []
test_loss = []

####################################################################
"""
PART III: Training and test performance
"""
best_loss = 1e5
# path for saving model
model_path = model_name + '.pth'

for epoch in range(1, num_epochs+1):
    train_loss_i = train(model_CNN, device, train_loader, criterion, optimizer, epoch)
    test_loss_i, acc_i = test(model_CNN, device, test_loader, criterion, epoch)
    scheduler.step()
    print('Optimizer Learning rate: {0:.5f}'.format(optimizer.param_groups[0]['lr']))
    acc.append(acc_i)
    train_loss.append(train_loss_i)
    test_loss.append(test_loss_i)
    if best_loss > test_loss_i:
        # if the current loss is lower than the best possible loss
        # save the model
        print('>>Saving the model: Test loss at {:.4f}'.format(test_loss_i))
        torch.save(model_CNN.state_dict(), model_path)
        best_loss = test_loss_i

plot_result(train_loss, test_loss, acc, num_epochs)
plot_confusion_matrix(model_path, test_loader, model_name, in_channels, out_channels)
