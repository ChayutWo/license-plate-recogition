# Import required package
import torch
import torch.optim

##################################################################
"""
Hyper-parameters that can be changed
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_batch_size = 64
test_batch_size = 256
# optimizer name: SGD or Adam
optimizer_name = 'Adam'
lr = 1e-2
scheduler_name = 'MultiStepLR'
milestones = [15, 70]
num_epochs = 80

in_channels = 3
out_channels = 4
model_name = 'resnet'

train_csv = 'dataset/cropped/training.csv'
train_path = 'dataset/cropped/training/'
validate_csv = 'dataset/cropped/validation.csv'
validate_path = 'dataset/cropped/validation/'
test_csv = 'dataset/cropped/testing.csv'
test_path = 'dataset/cropped/testing/'
