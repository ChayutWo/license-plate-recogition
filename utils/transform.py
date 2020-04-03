# Import module and packages
import numpy as np
from utils.variables import *

# Transformation class to be performed while loading
class reshapeSignal(object):
    """
    Resize signal into an array of size narrayxnpulse
    input: array of dimension lenx1
    output: 1 channel image of dimension narrayxnpulsex1
    """

    def __init__(self, narray, npulse):
        self.height = narray
        self.width = npulse

    def __call__(self, signal):
        signal = np.reshape(signal, (self.height, self.width), order='F')
        # change dim to [width, height, 1]
        signal = np.expand_dims(signal, axis=2)
        return signal


class separateComplexLayers(object):
    """
    Separate the complex number array into two different channel
    The first channel will be real number
    The second channel will be imaginary part
    input: array of complex number with dimension [width, height, 1]
    output: array of dimension [height, width, 2]
    """

    def __init__(self):
        pass

    def __call__(self, signal):
        signal = np.concatenate((signal.real, signal.imag), axis=2)
        return signal