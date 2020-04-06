# Import module and packages
import numpy as np
from utils.variables import *
import cv2

# Transformation class to be performed while loading
class rotate(object):
    """
    Resize signal into an array of size narrayxnpulse
    input: array of dimension lenx1
    output: 1 channel image of dimension narrayxnpulsex1
    """

    def __init__(self, angle):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"
        
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
        image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
        return image
    def rotate_box(self,corners,angle,  cx, cy, h, w):

        """Rotate the bounding box.


        Parameters
        ----------

        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

        angle : float
            angle by which the image is to be rotated

        cx : int
            x coordinate of the center of image (about which the box will be rotated)

        cy : int
            y coordinate of the center of image (about which the box will be rotated)

        h : int
            height of the image

        w : int
            width of the image

        Returns
        -------

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
        image, landmarks = sample['image'], sample['landmarks']
        # change dim to [width, height, 1]
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        image = self.rotate_img(image)
        # (h, w) = image.shape[:2]
        # (cX, cY) = (w // 2, h // 2)
        landmarks = self.rotate_box(landmarks, self.angle,cX,cY,h,w)
        #landmarks = self.get_enclosing_box(landmarks)
        return {'image': image, 'landmarks': landmarks}

class HorizontalFlip(object):
    def __init__(self):
        pass
    """
     Randomly horizontally flips the Image


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        corner of the box that identifies license plate

    """


    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img_center = np.array(image.shape[:2])[::-1]/2
        image =  image[:,::-1,:]
        landmarks[:,0] += 2*(img_center[0] - landmarks[:,0])


        return {'image': image, 'landmarks': landmarks}
