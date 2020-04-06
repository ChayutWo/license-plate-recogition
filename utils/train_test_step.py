# Import module and packages
import torch
from utils.variables import *

# This code is obtained from Pytorch tutorial of ECE deep learning class
def train(model, device, train_loader, criterion, optimizer, epoch):
    train_loss = 0
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'].to(device), sample['box'].to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Feed forward propagation
        output = model(data)

        # Compute loss and do back propagation
        lossx = criterion(output[0], target[0])
        lossy = criterion(output[1], target[1])
        lossw = criterion(output[2], target[2])
        lossh = criterion(output[3], target[3])
        loss = lossx + lossy + lossw + lossh
        loss.backward()

        # Update parameters
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss * train_batch_size / len(train_loader.dataset)
    print('Train({}): Loss: {:.4f}'.format(epoch, train_loss))
    return train_loss


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sample in test_loader:
            data, target = sample['image'].to(device), sample['box'].to(device)
            output = model(data)

            # Compute loss and do back propagation
            lossx = criterion(output[0], target[0])
            lossy = criterion(output[1], target[1])
            lossw = criterion(output[2], target[2])
            lossh = criterion(output[3], target[3])
            loss = lossx + lossy + lossw + lossh
            test_loss += loss

    test_loss = (test_loss * test_batch_size) / len(test_loader.dataset)
    print('Test({}): Loss: {:.4f}'.format(epoch, test_loss))
    return test_loss
