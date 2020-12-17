import imgaug.augmenters as iaa
import random
import numpy as np


class ImageAugment(object):
    """
    class for augment the training data using imgaug
    """
    def __init__(self, ratio):
        self.choice = ratio
        self.rotate = np.random.randint(-10, 10)
        self.scale_x = random.uniform(0.8, 1.0)
        self.scale_y = random.uniform(0.8, 1.0)
        self.translate_x = random.uniform(0, 0.1)
        self.translate_y = random.uniform(-0.1, 0.1)
        self.brightness = np.random.randint(-10, 10)
        self.linear_contrast = random.uniform(0.5, 2.0)
        self.alpha = random.uniform(0, 1.0)
        self.lightness = random.uniform(0.75, 1.5)
        self.Gaussian = random.uniform(0.0, 0.05*255)
        self.Gaussian_blur = random.uniform(0, 3.0)

    def aug(self, image, sequence):
        """
        :param image: need size (H, W, C) one image once
        :param sequence: collection of augment function
        :return:
        """
        image_aug = sequence(image=image)
        return image_aug

    def aug_sequence(self):
        sequence = self.aug_function()
        seq = iaa.Sequential(sequence, random_order=True)
        return seq

    def aug_function(self):
        sequence = []
        sequence.extend(iaa.SomeOf((1, self.choice),
                                       [
                                           iaa.OneOf([
                                               iaa.GaussianBlur(self.Gaussian_blur),  # blur images with a sigma between 0 and 3.0
                                           ]),
                                           iaa.Fliplr(0.5),  # 50% horizontally flip all batch images
                                           iaa.Flipud(0.5),  # 50% vertically flip all batch images
                                           iaa.Affine(
                                               scale={"x": self.scale_x, "y": self.scale_y},  # scale images to 80-100% of their size
                                               translate_percent={"x": self.translate_x, "y": self.translate_y},  # translate by -10 to +10 percent (per axis)
                                               rotate=(self.rotate),  # rotate by -15 to +15 degrees
                                           )
                                       ]))
        return sequence

