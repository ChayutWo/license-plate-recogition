# Import module and packages
import torch
from utils.variables import *

# This code is obtained from Pytorch tutorial of ECE deep learning class
def train(model, device, train_loader, criterion, optimizer, epoch):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Feed forward propagation
        output = model(data)

        # Compute loss and do back propagation
        loss = criterion(output, target)
        loss.backward()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)  # get index of predicted class
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Update parameters
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss * train_batch_size / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    print('Train({}): Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch, train_loss, acc))
    return train_loss


def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get index of predicted class
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = (test_loss * test_batch_size) / len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test({}): Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch, test_loss, acc))
    return test_loss, acc

# This code is obtained from Pytorch tutorial of ECE deep learning class
def train_binary(model, device, train_loader, criterion, optimizer, epoch):
    train_loss = 0
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Feed forward propagation
        output = model(data)

        # Compute loss and do back propagation
        loss = criterion(output, torch.unsqueeze(target, dim = 1))
        loss.backward()

        # Calculate accuracy
        pred = output > 0
        correct += pred.eq(target.view_as(pred)).sum().item()

        # Update parameters
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss * train_batch_size / len(train_loader.dataset)
    acc = 100. * correct / len(train_loader.dataset)
    print('Train({}): Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch, train_loss, acc))
    return train_loss

def test_binary(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.float().to(device)
            output = model(data)
            test_loss += criterion(output, torch.unsqueeze(target, dim = 1)).item()
            pred = output > 0
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss = (test_loss * test_batch_size) / len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test({}): Loss: {:.4f}, Accuracy: {:.4f}%'.format(epoch, test_loss, acc))
    return test_loss, acc