import numpy as np
from scipy.signal import convolve2d
# from scipy.misc import imread
import imageio
import matplotlib.pyplot as plt
import cv2

def sobel_edge_detection(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img)
    # plt.show()
    # print(img.shape, np.max(img))
    # dd
    # Prepare the kernels
    a1 = np.matrix([1, 2, 1])
    a2 = np.matrix([-1, 0, 1])
    Kx = a1.T * a2
    Ky = a2.T * a1

    # Apply the Sobel operator
    Gx = convolve2d(img, Kx, "same", "symm")
    Gy = convolve2d(img, Ky, "same", "symm")
    G = np.sqrt(Gx**2 + Gy**2)
    # or using the absolute values
    # G = np.abs(Gx) + np.abs(Gy)
    G *= 255.0 / G.max()
    # print(np.max(G))
    # G = G/255
    # print(np.max(G))
    # DD
    image_3channel = np.zeros((G.shape[0], G.shape[1], 3), dtype=np.uint8)

        # Copy the grayscale channel to all three channels
    image_3channel[:, :, 0] = G
    image_3channel[:, :, 1] = G
    image_3channel[:, :, 2] = G

    return image_3channel

