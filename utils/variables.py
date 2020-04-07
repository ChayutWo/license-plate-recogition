# Import required package
import torch
import torch.optim

##################################################################
"""
Hyper-parameters that can be changed
"""
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
train_batch_size = 64
test_batch_size = 128
# optimizer name: SGD or Adam
optimizer_name = 'SGD'
lr = 1e-3
scheduler_name = 'MultiStepLR'
milestones = [5]
num_epochs = 10

in_channels = 3
out_channels = 4
model_name = 'resnet'

train_csv = 'directory_training.csv'
train_path = 'dataset/UFPR-ALPR dataset/training'
validate_csv = 'directory_validation.csv'
validate_path = 'dataset/UFPR-ALPR dataset/validation'
test_csv = 'directory_testing.csv'
test_path = 'dataset/UFPR-ALPR dataset/testing'
