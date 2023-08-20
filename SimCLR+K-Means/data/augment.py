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
import os
from PIL import Image

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
        # img = gaussian_blur(np.array(img), 5, verbose=False)
        img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), 0)
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

class Selection_crops(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img

class RandomCrop(object): # cropping the synthetic fiber image
    def __init__(self, crop_size,  num_of_crops):
        self.crop_size = crop_size
        self.num_of_crops =  num_of_crops

    def __call__(self, image):
        # Define the transform to apply random crop
        transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
        ])
        highest_pixels = 0
        selected_image = None
        for _ in range(0,self.num_of_crops):
            cropped_image = transform(image)
            # import matplotlib.pyplot as plt
            # plt.imshow(cropped_image)
            # plt.show()
            color_pixels, total_pixels = count_color_gray_pixels(np.array(cropped_image))
            if color_pixels > highest_pixels:
                highest_pixels = color_pixels
                selected_image = cropped_image
            
        if selected_image == None:
            selected_image = transform(image)
        #calculation of black and white intensity
        # biggest_threshold = 9999999999

        # for i in range(0,num_of_crops):
        #     cropped_image = transform(image)
        #     intensity = np.sum(cropped_image.detach().numpy()) /( 240 * 240)
        #     # plt.imshow(np.transpose(cropped_image, (1, 2, 0)))
        #     # plt.title(str(intensity))
        #     # plt.show()
        #     if intensity < biggest_threshold:
        #         biggest_threshold = intensity
        #         selected_image = cropped_image
        # selected_image = transform(image)
        selected_image = cv2.cvtColor(np.array(selected_image), cv2.COLOR_BGR2GRAY)
        image_3channel = np.zeros((selected_image.shape[0], selected_image.shape[1], 3), dtype = np.uint8)

        # Copy the grayscale channel to all three channels
        image_3channel[:, :, 0] = selected_image
        image_3channel[:, :, 1] = selected_image
        image_3channel[:, :, 2] = selected_image
        
        return image_3channel
    
# class RandomCrop_2(object): # cropping the cotton fiber based on the segment.
#     def __init__(self, crop_size,  num_of_crops, basefolder, set_ind):
#         self.crop_size = crop_size
#         self.num_of_crops =  num_of_crops
#         path = os.path.join(os.getcwd(), 'Datasets', 'cotton')
#         print(path)
#         if set_ind == 'Train':
#             filenames = 'train_names_seg.txt'
#             images = 'train_seg.npy'
#         else:
#             filenames = 'test_names_seg.txt'
#             images = 'test_seg.npy'
#         image_names = np.loadtxt(os.path.join(path,filenames), delimiter=',', dtype=str)
#         data = np.load(os.path.join(path, images))
#         self.segdata = data
#         self.segnames = image_names
#         print(self.segdata, self.segnames)
#         print(self.segdata.shape, len(self.segnames))
#         import matplotlib.pyplot as plt
#         plt.imshow(self.segdata[0])
#         plt.title(self.segnames[0])
#         plt.show()
#         dd


    # def __call__(self, image):
    #     # Define the transform to apply random crop
    #     transform = transforms.Compose([
    #         transforms.RandomCrop(self.crop_size),
    #     ])
    #     highest_pixels = 0
    #     selected_image = None
    #     for _ in range(0,self.num_of_crops):
    #         cropped_image = transform(image)
    #         # import matplotlib.pyplot as plt
    #         # plt.imshow(cropped_image)
    #         # plt.show()
    #         color_pixels, total_pixels = count_color_gray_pixels(np.array(cropped_image))
    #         if color_pixels > highest_pixels:
    #             highest_pixels = color_pixels
    #             selected_image = cropped_image
            
    #     if selected_image == None:
    #         selected_image = transform(image)
    #     #calculation of black and white intensity
    #     # biggest_threshold = 9999999999

    #     # for i in range(0,num_of_crops):
    #     #     cropped_image = transform(image)
    #     #     intensity = np.sum(cropped_image.detach().numpy()) /( 240 * 240)
    #     #     # plt.imshow(np.transpose(cropped_image, (1, 2, 0)))
    #     #     # plt.title(str(intensity))
    #     #     # plt.show()
    #     #     if intensity < biggest_threshold:
    #     #         biggest_threshold = intensity
    #     #         selected_image = cropped_image
    #     # selected_image = transform(image)
    #     selected_image = cv2.cvtColor(np.array(selected_image), cv2.COLOR_BGR2GRAY)
    #     image_3channel = np.zeros((selected_image.shape[0], selected_image.shape[1], 3), dtype = np.uint8)

    #     # Copy the grayscale channel to all three channels
    #     image_3channel[:, :, 0] = selected_image
    #     image_3channel[:, :, 1] = selected_image
    #     image_3channel[:, :, 2] = selected_image
        
    #     return image_3channel


def select_crop(index, dataset, size, image, no_of_crops):
    image_name = dataset.names[index]
    frame_index = image_name.find("frame")  # Find the index of "frame" in the input string
    if frame_index != -1:
        substring = image_name[:frame_index-1]
        # print("Substring:", substring)
    else:
        print("image formatting is wrong")
        raise NotImplementedError
    # print(self.dataset.seg_image_names)
    ind = dataset.seg_image_names.index(substring)
    # print(index, substring)
    seg_image = dataset.seg_images[ind]
                
    transform = transforms.Compose([
                    transforms.RandomCrop(size)])
    
    highest_pixels = 0
    selected_image = None
    crop_image = None
    seg_image = Image.fromarray(seg_image.astype(np.uint8))
    for _ in range(0,no_of_crops):
        seed = np.random.randint(10000)
        random.seed(seed)  # Set a seed for reproducibility
        torch.manual_seed(seed)

        cropped_image = transform(seg_image)
        import matplotlib.pyplot as plt
        # plt.imshow(cropped_image)
        # plt.title(seed)
        # plt.show()
        cropped_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_BGR2GRAY)
        white_pixel_count = cv2.countNonZero(np.array(cropped_image))
        # print(white_pixel_count)
        if white_pixel_count > highest_pixels:
            highest_pixels = white_pixel_count
            selected_image = cropped_image
            random.seed(seed)
            torch.manual_seed(seed)
            crop_image = transform(image)

    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # axes[0].imshow(selected_image)
    # axes[0].set_title('segmented image')
    # axes[0].axis('off')
    # axes[1].imshow(crop_image)
    # axes[1].set_title('crop image ')
    # axes[1].axis('off')
    # plt.show()

    if crop_image ==  None:
        crop_image = transform(image)
    return crop_image

if __name__ == '__main__':
    import glob
    import os
    folder_path = r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_images_new'
    file_path = r'C:\Users\AVLguest\work\Random_experiments\Synthetic_texture\new_data\texture_2_new.txt'
    names = []
    with open(file_path, "r") as file:
        for line in file:
            image_name = line.strip().split(' - ')[0]
            names.append(image_name + ".jpg")

    random_image_path = random.choice(names)
    random_image_path = os.path.join(folder_path, random_image_path)
    rand = RandomCrop(crop_size = 100, num_of_crops = 10)
    import cv2
    import matplotlib.pyplot as plt
    from PIL import Image
    image = image_array = cv2.imread(random_image_path)
    
    image = Image.fromarray(image.astype(np.uint8))
    cropped_image1 = rand.__call__(image)
    cropped_image2 = rand.__call__(image)
    fig, axes = plt.subplots(nrows=1, ncols=3)

    # sobel = Sobel()
    # cropped_image1 = sobel.__call__(cropped_image1)
    # cropped_image2 = sobel.__call__(cropped_image2)
    axes[0].imshow(image, cmap = 'gray')
    axes[0].axis('off')
    axes[0].set_title('Original Image')
    axes[1].imshow(cropped_image1)
    axes[1].axis('off')
    axes[1].set_title('Augmentation 1')
    axes[2].imshow(cropped_image2)
    axes[2].axis('off')
    axes[2].set_title('Augmentation 2')
    # # rand.__call__(img)

    # # find_fiber_centers(image, min_fiber_area=100, crop_size=140)
    plt.show()

    # image1 = cv2.imread("image1.jpg")
    # image2 = cv2.imread("image2.jpg")

    # Calculate the similarity score
    # similarity_score = image_similarity(image, np.transpose(cropped_image, (1, 2, 0)))

    # print(f"The similarity score between the images is: {similarity_score}")
