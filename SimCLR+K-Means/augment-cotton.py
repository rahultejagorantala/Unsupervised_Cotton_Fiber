# List of augmentations based on randaugment
import random
import cv2
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
# from sobel import sobel_edge_detection
from data.sobel import sobel_edge_detection
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
# from gaussian_smoothing import gaussian_blur
from data.gaussian_smoothing import gaussian_blur
import math

random_mirror = True

def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Identity(img, v):
    return img

def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def augment_list():
    l = [
        (Identity, 0, 1),  
        (AutoContrast, 0, 1),
        (Equalize, 0, 1), 
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Sobel(object):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img):
        # img = gaussian_blur(np.array(img), self.kernel_size, verbose=False)
        # sigma = math.sqrt(kernel_size)
        # img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
        # plt.imshow(img)
        # plt.title("Gaussian Blur")
        # plt.show()
        sobel = sobel_edge_detection(np.array(img))
        return sobel


def count_color_gray_pixels(image):

    # Check for color points (different RGB values in each channel)
    color_points_mask = np.logical_or.reduce((
        image[:, :, 0] != image[:, :, 1],
        image[:, :, 1] != image[:, :, 2],
        image[:, :, 0] != image[:, :, 2]
    ))
    # Count the number of color points
    num_color_pixels = np.sum(color_points_mask)

    # Count the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # plt.imshow(color_points_mask)
    # plt.title(num_color_pixels)
    # plt.show()
    # Calculate the number of gray points (same RGB values in all channels)
    # num_gray_pixels = total_pixels - num_color_pixels

    return num_color_pixels, total_pixels

class RandomCrop(object):
    def __init__(self, crop_size,  num_of_crops):
        self.crop_size = crop_size
        self.num_of_crops =  num_of_crops

    def __call__(self, image):
        # Define the transform to apply random crop
        transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
        ])
        # highest_pixels = 0
        # selected_image = None
        # for _ in range(0,self.num_of_crops):
        #     cropped_image = transform(image)
        #     # import matplotlib.pyplot as plt
        #     # plt.imshow(cropped_image)
        #     # plt.show()
        #     color_pixels, total_pixels = count_color_gray_pixels(np.array(cropped_image))
        #     if color_pixels > highest_pixels:
        #         highest_pixels = color_pixels
        #         selected_image = cropped_image
            
        # if selected_image == None:
        #     selected_image = transform(image)
        # calculation of black and white intensity
        biggest_threshold = 9999999999

        for i in range(0,self.num_of_crops):
            cropped_image = transform(image)
            intensity = np.sum(cropped_image) /( self.crop_size * self.crop_size)
            # plt.imshow(np.transpose(cropped_image, (1, 2, 0)))
            # plt.title(str(intensity))
            # plt.show()
            if intensity < biggest_threshold:
                biggest_threshold = intensity
                selected_image = np.array(cropped_image)

        # print(type(selected_image))
      
        return selected_image
    
if __name__ == '__main__':
    import os
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    folder_path = r'C:\Users\AVLguest\work\Unsupervised\Unsupervised-Classification-master\Datasets\cotton\Train'

    for i in range(100):
        image_name = random.choice(os.listdir(folder_path))
        random_image_path = os.path.join(folder_path, image_name)
        rand = RandomCrop(crop_size = 900, num_of_crops = 3)
        
        image = image_array = cv2.imread(random_image_path)
        
        image = Image.fromarray(image.astype(np.uint8))
        cropped_image1 = rand.__call__(image)
        cropped_image2 = rand.__call__(image)
        

        sobel = Sobel(5)
        cropped_image1 = sobel.__call__(cropped_image1)
    # cropped_image2 = sobel.__call__(cropped_image2)
    # plt.figure()
    # fig, axes = plt.subplots(nrows=1, ncols=3)
    # axes[0].imshow(cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB), cmap = 'gray')
    # axes[0].axis('off')
    # axes[0].set_title('Original Image - ' + str(image_name), fontsize=6)
    # axes[1].imshow(cropped_image1)
    # axes[1].axis('off')
    # axes[1].set_title('Augmentation 1')
    # axes[2].imshow(cropped_image2)
    # axes[2].axis('off')
    # axes[2].set_title('Augmentation 2')
    # # # rand.__call__(img)
    # # # find_fiber_centers(image, min_fiber_area=100, crop_size=140)
    # plt.show()
    # plt.figure()
        plt.imshow(cropped_image1)
        plt.axis('off')
        plt.savefig(os.path.join(r'C:\Users\AVLguest\work\Unsupervised\Sobel-Filter-Testing\data\sobel', image_name ))

    # image1 = cv2.imread("image1.jpg")
    # image2 = cv2.imread("image2.jpg")

    # Calculate the similarity score
    # similarity_score = image_similarity(image, np.transpose(cropped_image, (1, 2, 0)))

    # print(f"The similarity score between the images is: {similarity_score}")
