import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.variables import *
from models.generate import *

def plot_confusion_matrix(model_path, dataloader, model_name, in_channels, out_channels):
    # Load best model
    model = create_model(model_name, in_channels, out_channels)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    nb_classes = out_channels
    # Calculate confusion matrix
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    # Normalize the matrix
    result = (confusion_matrix / confusion_matrix.sum(1)).numpy()

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(result)

    # Loop over data dimensions and create text annotations.
    for i in range(nb_classes):
        for j in range(nb_classes):
            text = ax.text(j, i, result[i, j],
                           ha="center", va="center", color="w")
    plt.title(model_name)
    plt.imshow(result, vmin=0, vmax=1.0)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.colorbar()
    fig.savefig('confusion_matrix.png')

def plot_confusion_matrix_binary(model_path, dataloader, model_name, in_channels, out_channels):
    # Load best model
    model = create_model(model_name, in_channels, out_channels)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    nb_classes = 2
    # Calculate confusion matrix
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            preds = outputs > 0
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    # Normalize the matrix
    result = (confusion_matrix / confusion_matrix.sum(1)).numpy()

    # Plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(result)

    # Loop over data dimensions and create text annotations.
    for i in range(nb_classes):
        for j in range(nb_classes):
            text = ax.text(j, i, result[i, j],
                           ha="center", va="center", color="w")
    plt.title(model_name)
    plt.imshow(result, vmin=0, vmax=1.0)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.colorbar()
    fig.savefig('confusion_matrix.png')
