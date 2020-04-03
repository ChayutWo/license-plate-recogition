# Import required package
import torch
import torch.optim

##################################################################
"""
Hyper-parameters that can be changed
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_batch_size = 128
test_batch_size = 512
# optimizer name: SGD or Adam
optimizer_name = 'SGD'
lr = 1e-3
scheduler_name = 'MultiStepLR'
milestones = [50, 100]
num_epochs = 100

in_channels = 2
out_channels = 5
model_name = 'resnetL'
data_size = 'L'

##################################################################
# csv containing train set metadata and path to the training data

if data_size == 'L':
    # large data set
    train_csv = 'datasetsL.csv'
    train_path = './dataL/'
    test_csv = 'testsetsL.csv'
    test_path = './testL/'
elif data_size == 'M':
    # medium size data set
    train_csv = 'datasets.csv'
    train_path = './data/'
    test_csv = 'testsets.csv'
    test_path = './test/'
elif data_size == 'binary':
    # large data set
    out_channels = 1
    train_csv = 'binary.csv'
    train_path = './dataL/'
    test_csv = 'testbinary.csv'
    test_path = './testL/'

# constants specific to the dataset
narray = 11 # number of radar receiver arrays: height
npulse = 32 # number of radar pulse: width
signal_channel = 2 # Real, Complex data input channel
