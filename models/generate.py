from models.resnet import resnet
from models.resnetS import resnetS
from models.resnetSS import resnetSS
from models.simple import simple

def create_model(model_name, in_channels, out_channels):
    if model_name == 'resnet':
        model = resnet(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'resnetS':
        model = resnetS(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'resnetSS':
        model = resnetSS(in_channels=in_channels, out_channels=out_channels)
    elif model_name == 'simple':
        model = simple(in_channels=in_channels, out_channels = out_channels)
    return model
