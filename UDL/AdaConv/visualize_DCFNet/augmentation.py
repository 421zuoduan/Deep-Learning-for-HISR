# -*- coding: utf-8 -*-
"""
Created on 2020-09-10 19:14
@author: jyw
"""
import torch
import random
import numpy as np
import PIL
from PIL import Image, ImageFilter
import torchvision.transforms.functional as tf
import torch.nn.functional as F


def _check_type(image, label):
    """ Check the type of the given image and label to be transformed"""
    if not isinstance(image, torch.Tensor):
        raise TypeError('image type should be Torch.Tensor. Got {}'.format(type(image)))
    if not isinstance(label, torch.Tensor):
        raise TypeError('label type should be Torch.Tensor. Got {}'.format(type(label)))


def to_tensor(image: PIL.Image.Image, label, transform):
    """Convert PIL Image to torch.tensor
    Args:
        transform: pre-defined torch transform operations.
    """
    if not isinstance(image, Image.Image):
        raise TypeError('image type should be PIL.Image.Image. Got {}'.format(type(image)))
    if not isinstance(label, Image.Image):
        raise TypeError('label type should be PIL.Image.Image. Got {}'.format(type(label)))
    if not isinstance(transform, list):
        raise TypeError('transform type should be list. Got {}'.format(type(transform)))

    image_transform, label_transform = transform
    image = image_transform(image)
    label = label_transform(label)
    return image.unsqueeze(dim=0), label.unsqueeze(dim=0)


def random_rotation(image, label, angle=[-90, 90], p=1, image_resample_mode=Image.BILINEAR,
                    label_resample_mode=Image.NEAREST):
    """ randomly rotate the input image and corresponding label with a given angle."""
    if random.random() <= p:
        r = 0
        if isinstance(angle, list):
            r = random.randrange(angle[0], angle[1])
        else:
            assert "angle should be list. Got {}".format(type(angle))

        image = image.rotate(r, resample=image_resample_mode)
        label = label.rotate(r, resample=label_resample_mode)

    return image, label


def random_flip(image, label, p=1):
    if random.random() <= p:
        if random.random() <= 0.5:
            image = tf.hflip(image)
            label = tf.hflip(label)
        else:
            image = tf.vflip(image)
            label = tf.vflip(label)

    image, label = to_tensor(image, label, transform)

    return image, label


def random_resize(image, label, transforms, scale=(0.5, 2), p=1):  # scale表示随机crop出来的图片会在的0.5倍至2倍之间，ratio表示长宽比
    """randomly resize the input image and label
    :param label:
    :param image:
    :param transforms: (list) pre-defined transforms
    :param scale: (list) cropped scale
    :param p: (float) perform probability
    """

    # Note: image size is 1*3*H*W, label size is 1*1*H*W
    image, label = to_tensor(image, label, transforms)

    if random.random() <= p:
        rows, cols = image.size()[2], image.size()[3]
        r = random.randint(scale[0] * 10, scale[1] * 10) / 10

        new_rows, new_cols = int(r * rows), int(r * cols)

        image = F.interpolate(image, size=(new_rows, new_cols), mode='bilinear', align_corners=True)
        label = F.interpolate(label, size=(new_rows, new_cols), mode='nearest')

        if new_rows > rows:  # resize后的图像尺寸大于原图则crop至原图大小
            x1 = int(new_rows / 2 - rows / 2)
            x2 = x1 + rows
            y1 = int(new_cols / 2 - cols / 2)
            y2 = int(y1 + cols)

            # print('r: {}'.format(r))
            # print('new_rows: {}, new_cols: {}'.format(new_rows, new_cols))
            # print('x1: {}, x2: {}, y1: {}, y2: {}'.format(x1, x2, y1, y2))

            image = image[:, :, x1:x2, y1:y2]
            label = label[:, :, x1:x2, y1:y2]

        if new_rows < rows:  # resize后的图像尺寸小于原图则pad至原图大小
            padding = int((rows - new_rows) / 2)
            image = F.pad(image, pad=[padding, padding, padding, padding], mode='constant', value=0.0)
            label = F.pad(label, pad=[padding, padding, padding, padding], mode='constant', value=0.0)
            if image.size()[2] != rows:
                image = F.interpolate(image, size=(rows, cols), mode='bilinear', align_corners=True)
                label = F.interpolate(label, size=(rows, cols), mode='nearest')

    image = image.squeeze(dim=0)
    label = label.squeeze(dim=0)
    return image, label


def adjust_contrast(image, label, scale=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_contrast(image, scale)
    return image, label


def adjust_brightness(image, label, factor=0.125, p=1):
    if random.random() <= p:
        image = tf.adjust_brightness(image, factor)
    return image, label


def adjust_saturation(image, label, factor=0.5, p=1):
    if random.random() <= p:
        image = tf.adjust_saturation(image, factor)
    return image, label


def adjust_hue(image, label, factor=0.2, p=1):
    if random.random() <= p:
        image = tf.adjust_hue(image, hue_factor=factor)
    return image, label


def gaussian_blur(image, label, radius=3, p=1):
    if random.random() <= p:
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return image, label


def gaussian_noise(image, label, noise_sigma=10, p=1):
    if random.random() <= p:
        temp_image = np.float64(np.copy(image))
        h, w, _ = temp_image.shape
        # 标准正态分布*noise_sigma
        noise = np.random.randn(h, w) * noise_sigma
        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

            image = Image.fromarray(np.uint8(noisy_image))
    return image, label


total_strategy = ['gaussian_noise', 'adjust_brightness', 'adjust_contrast', 'adjust_hue', 'adjust_saturation',
                  'gaussian_blur', 'random_flip', 'random_resize', 'random_rotation']

if __name__ == '__main__':
    from torchvision.transforms import transforms as tff

    x_transform = tff.Compose([
        tff.ToTensor(),
        # tf.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD)
    ])
    y_transform = tff.ToTensor()

    imsrc = "/home/jyw/数据/建筑提取/WHU/train/image/4.tif"
    gtsrc = "/home/jyw/数据/建筑提取/WHU/train/label/4.tif"

    im = Image.open(imsrc).convert('RGB')
    gt = Image.open(gtsrc).convert('L')

    from matplotlib import pyplot as plt

    #
    plt.subplot(121)
    plt.imshow(np.asarray(im))
    plt.subplot(122)
    plt.imshow(np.asarray(gt))
    plt.show()

    im, gt = random_rotation(im, gt)

    if not isinstance(im, torch.Tensor):
        im = np.asarray(im).astype(np.uint8)
        gt = np.asarray(gt).astype(np.uint8)
    else:
        im = np.uint8(im.cpu().numpy().transpose(1, 2, 0) * 255)
        gt = gt.squeeze().cpu().numpy()

    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(gt)
    plt.show()

    # im = Image.open(imsrc).convert('RGB')
    # gt = Image.open(gtsrc).convert('L')
    #
    # im, gt = random_rotation(im, gt)
    #
    # from matplotlib import pyplot as plt
    #
    # plt.subplot(121)
    # plt.imshow(np.asarray(im))
    # plt.subplot(122)
    # plt.imshow(np.asarray(gt))
    # plt.show()
