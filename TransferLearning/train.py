import torch
import torchvision
import numpy as np
from TransferLearning import dataloader, transfer_utils, transfer_iou
from TransferLearning.engine import train_one_epoch
from TransferLearning.transforms import get_transform

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)


#
RANDOM_SAMPLE = None  # Change it to None to ran whole sample
dataset = dataloader.TransferLearningDataset('../dataset/cropped/training.csv', '../dataset/cropped/training/',
                                             get_transform(train=False), random_sample=RANDOM_SAMPLE)
dataset_test = dataloader.TransferLearningDataset('../dataset/cropped/testing.csv', '../dataset/cropped/testing/',
                                                  get_transform(train=False), random_sample=RANDOM_SAMPLE)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=0,
    collate_fn=transfer_utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=False, num_workers=0,
    collate_fn=transfer_utils.collate_fn)

# test code
# images,targets = next(iter(data_loader))

# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]

# output = model(images,targets)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-2,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

RUN_MODEL = False

if RUN_MODEL:
    num_epochs = 5

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
    torch.save(model, 'transferLearning5Epoch.pth')
else:
    model = torch.load('transferLearning5Epoch.pth')

iou_list = transfer_iou.iou_from_model(data_loader_test, model, device)
print(np.mean(iou_list))
transfer_iou.plot_iou(data_loader_test, model, device, random_sample=10, save_fig=True)
