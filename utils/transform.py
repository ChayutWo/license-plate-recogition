# Import module and packages
import numpy as np
from utils.variables import *
import cv2
from torchvision import transforms, utils
import random
from skimage import transform

# Transformation class to be performed while loading
class rotate(object):
    """
    Resize signal into an array of size narrayxnpulse
    input: array of dimension lenx1
    output: 1 channel image of dimension narrayxnpulsex1
    """

    def __init__(self, Maxangle):
        self.Maxangle = Maxangle

    def rotate_img(self,image):
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), self.angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW,nH))
        return image

    def rotate_box(self,corners,angle,  cx, cy, h, w):

        """Rotate the bounding box.
​
​
        Parameters
        ----------
​
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
​
        angle : float
            angle by which the image is to be rotated
​
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
​
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
​
        h : int
            height of the image
​
        w : int
            width of the image
​
        Returns
        -------
​
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

        M = cv2.getRotationMatrix2D((cx, cy), self.angle, 1.0)


        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        calculated = calculated.reshape(-1,2)
        return calculated

    def __call__(self, sample):
        self.angle = np.random.uniform(low = -self.Maxangle, high = self.Maxangle)
        image, landmarks = sample['image'], sample['landmarks']
        # change dim to [width, height, 1]
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        image = self.rotate_img(image)
        landmarks = self.rotate_box(landmarks, self.angle,cX,cY,h,w)
        return {'image': image, 'landmarks': landmarks}

class HorizontalFlip(object):
    def __init__(self,p=0.5):
        self.p = p
        pass
    """
     Randomly horizontally flips the Image
​
​
    Returns
    -------
​
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
​
    numpy.ndarray
        corner of the box that identifies license plate
​
    """


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img_center = np.array(image.shape[:2])[::-1]/2
        if random.random() < self.p:
            image =  image[:,::-1,:]
            landmarks[:,0] += 2*(img_center[0] - landmarks[:,0])
        return {'image': image, 'landmarks': landmarks}


class ColorJitter(object):
    def __init__(self,brightness=0.1, contrast=0.1,saturation=0.1, hue=0.1):
        self.brightness=brightness
        self.contrast=contrast
        self.saturation=saturation
        self.hue=hue
        pass

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        jitter = transforms.ColorJitter(brightness=self.brightness,
                                        contrast=self.contrast,
                                        saturation=self.saturation,
                                        hue=self.hue)

        image = jitter(image)

        return {'image': image, 'landmarks': landmarks}

class resize(object):
    def __init__(self):
        pass
    """
     Randomly horizontally flips the Image
​
​
    Returns
    -------
​
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
​
    numpy.ndarray
        corner of the box that identifies license plate
​
    """


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_width = 640
        image_height = 360
        original_width = image.width
        original_height = image.height
        res = transforms.Resize((image_height,image_width))
        image = res(image)
        landmarks[:,1] = landmarks[:,1]*image_height/original_height
        landmarks[:, 0] = landmarks[:, 0] * image_width / original_width

        return {'image': image, 'landmarks': landmarks}

class PILconvert(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        transformToPIL = transforms.ToPILImage()
        image = transformToPIL(image)
        return {'image': image, 'landmarks': landmarks}

class tensorize(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        transformToTensor = transforms.ToTensor()
        image = transformToTensor(image)

        return {'image': image, 'landmarks': landmarks}