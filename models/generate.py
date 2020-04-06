from models.resnet import resnet
from models.simple import simple


def create_model(model_name, in_channels, out_channels):
    if model_name=='resnet':
        model = resnet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'simple':
        model = simple(in_channels=in_channels, out_channels=out_channels)
    return model
