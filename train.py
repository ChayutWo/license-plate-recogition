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
composed = transforms.Compose([HorizontalFlip(p=0.5),
                               rotate(Maxangle=5),
                               PILconvert(),
                               ColorJitter(brightness=0.2, contrast=0.2,
                                                     saturation=0.2, hue=0.2),
                               resize(),
                               tensorize()])

composed_test = transforms.Compose([PILconvert(), resize(), tensorize()])

# Train and test dataset using RadarDataset class
traindata = LicenseLandmarksDataset(train_csv, train_path, composed)
validatedata = LicenseLandmarksDataset(validate_csv, validate_path, composed_test)
testdata = LicenseLandmarksDataset(test_csv, test_path, composed_test)

# Train and test dataloader setup
train_loader = torch.utils.data.DataLoader(traindata,
                                           batch_size=train_batch_size, shuffle=True,
                                           num_workers=0, pin_memory=True)

validation_loader = torch.utils.data.DataLoader(validatedata,
                                          batch_size=test_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)

test_loader = torch.utils.data.DataLoader(testdata,
                                          batch_size=test_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)

print('>> Finish creating data loader')
####################################################################
"""
PART II: Preparation for training
"""
torch.manual_seed(5)
torch.cuda.manual_seed(5)

# Create model and setup criterior, optimizer, and scheduler
model_CNN = create_model(model_name, in_channels, out_channels).to(device)

# make criterion
criterion = nn.MSELoss()

criterion = criterion.to(device)
optimizer = make_optimizer(optimizer_name, model_CNN, lr=lr, momentum=0.9, weight_decay=0)
scheduler = make_scheduler(scheduler_name, optimizer, milestones=milestones, factor=0.1)

print('>> Finish creating model')
summary(model_CNN, (3, 640, 360))
print('Using {}'.format(device))

# training and testing loss
acc_0 = test(model_CNN, device, validation_loader, criterion, 0)
train_loss = []
validation_loss = []

####################################################################
"""
PART III: Training and test performance
"""
best_loss = 1e5
# path for saving model
model_path = model_name + '.pth'

for epoch in range(1, num_epochs+1):
    train_loss_i = train(model_CNN, device, train_loader, criterion, optimizer, epoch)
    validation_loss_i = test(model_CNN, device, validation_loader, criterion, epoch)
    scheduler.step()
    print('Optimizer Learning rate: {0:.5f}'.format(optimizer.param_groups[0]['lr']))
    train_loss.append(train_loss_i)
    validation_loss.append(validation_loss_i)
    if best_loss > validation_loss_i:
        # if the current loss is lower than the best possible loss
        # save the model
        print('>>Saving the model: Test loss at {:.4f}'.format(validation_loss_i))
        torch.save(model_CNN.state_dict(), model_path)
        best_loss = validation_loss_i
    if epoch == num_epochs:
        # Save training model to evaluate performance on training set
        torch.save(model_CNN.state_dict(), 'training_model.pth')
        # Last loop, evaluate on test set
        print('Complete the training process, Evaluating on the test set...')
        test_loss = test(model_CNN, device, test_loader, criterion, epoch)
plot_result(train_loss, validation_loss, num_epochs)
