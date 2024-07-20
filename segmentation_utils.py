import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import skimage
import cv2 as cv
import os
from tqdm import tqdm
import torch
from sklearn.covariance import LedoitWolf
from skimage import morphology
from skimage.morphology import disk, binary_closing, binary_opening
import math



def read_image(image_path: str) -> np.ndarray:
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, image_path: str):
    """save image in .png"""
    if image is None or image.size == 0:
        print(f"Skipping saving empty image to {image_path}")
        return
    cv.imwrite(image_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))

def read_all_images(folder_path: str):
    image_paths = list(sorted(os.listdir(folder_path)))
    imgs = [read_image(f"{folder_path}/{img}") for img in tqdm(image_paths, desc=f'Reading images in {folder_path}...')]
    return imgs


USE_SIZE_REF = (1200, 800)

def resize_image(image: np.ndarray, size : tuple) -> np.ndarray:
    #check the scale in conserved
    assert size[0]/image.shape[1] == size[1]/image.shape[0]

    return cv.resize(image, size)

def average_all_images(imgs : list[np.ndarray]) -> np.ndarray:
    """
    Average all images in the list imgs

    Args:
    imgs : list[np.ndarray] : list of nxm images to average

    Returns:
    np.ndarray : the average images of size nxm
    
    """
    #print(np.array(imgs).shape)
    average_image = np.mean(imgs, axis=0)
    average_image = np.uint8(average_image)
    #print(average_image.shape)


    return average_image

def median_all_images(imgs : list[np.ndarray]) -> np.ndarray:
    """
    Median all images in the list imgs

    Args:
    imgs : list[np.ndarray] : list of nxm images to average

    Returns:
    np.ndarray : the median images of size nxm
    
    """
    median_image = np.median(imgs, axis=0)
    median_image = np.uint8(median_image)

    return median_image

class MahalanobisClassifier:
    """Mahalanobis based classifier"""

    def __init__(self):
        """
        Attributes:
            means: (torch.tensor): (n_classes, d) Mean of each class
            inv_covs: (torch.tensor): (n_classes, d, d) Inverse of covariance matrix across d features for each class   
        """
        super().__init__()
        self.means = None
        self.inv_covs = None
        
    def fit(self, train_x: torch.Tensor):
        """Computes parameters for Mahalanobis Classifier (self.mean and self.cov), fitting the training data.

        Args:
            train_x (torch.Tensor): (N, d) The tensor of training features
        """
        # Define number of classes
        n_classes = 3
        n, d = train_x.shape
        
        # Set default values
        means = torch.zeros((n_classes, d), dtype=train_x.dtype)
        inv_covs = torch.zeros((n_classes, d, d), dtype=train_x.dtype)

        for i in range(n_classes):
            train_x_class = train_x[train_x[:, -1] == i]
            means[i] = torch.mean(train_x_class[:, :-1], dim=0)
            cov = np.cov(train_x_class[:, :-1].T)
            inv_covs[i] = np.linalg.inv(cov)
            
        self.means = means
        self.inv_covs = inv_covs

    def predict(self, test_x: torch.Tensor) -> torch.Tensor:
        """Predicts the background of every image, based on the Mahalanobis distance to the class means.

        Args:
            test_x (torch.Tensor): (N, d) The tensor of test features

        Returns:
            preds (torch.Tensor): (N,) The predictions tensor (id of the predicted class {0, 1, ..., n_classes-1})
            dists (torch.Tensor): (N, n_classes) Mahalanobis distance from sample to class means
        """
        # Define default output value
        N, d = test_x.shape
        dists = torch.zeros((N, self.means.shape[0]), dtype=test_x.dtype)
        preds = torch.zeros(N, dtype=torch.long)

        for i in range(N):
            for j in range(self.means.shape[0]):
                dists[i, j] = torch.sqrt(torch.matmul(test_x[i, :-1] - self.means[j], torch.matmul(self.inv_covs[j], (test_x[i, :-1] - self.means[j]).unsqueeze(-1))).squeeze())
            preds[i] = torch.argmin(dists[i])

        return preds, dists 
    
def remove_background(image, background):
    """Remove background from the image, if the difference is lower than 0 set it to 0"""
    diff = cv.absdiff(image, background)
    
    return diff

def plot_image(image: np.ndarray, title: str = None):
    """Plot an image"""
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    
    return data_red, data_green, data_blue

def plot_rgb_channels(img, title):
    red_channel, green_channel, blue_channel = extract_rgb_channels(img)

    plt.subplot(1, 3, 1)
    plt.imshow(red_channel, cmap='Reds')
    plt.title(f'{title}: Red Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(green_channel, cmap='Greens')
    plt.title(f'{title}: Green Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blue_channel, cmap='Blues')
    plt.title(f'{title}: Blue Channel')
    plt.axis('off')

def plot_colors_histo(
    img: np.ndarray,
    func: Callable,
    labels: list[str],
):
    """
    Plot the original image (top) as well as the channel's color distributions (bottom).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    func: Callable
        A callable function that extracts D channels from the input image
    labels: list of str
        List of D labels indicating the name of the channel
    """

    # Extract colors
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    # Use random seed to downsample image colors (increase run speed - 10%)
    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1
    
    # Plot base image
    ax = fig.add_subplot(gs[:2, :])
    ax.imshow(img)
    # Remove axis
    ax.axis('off')
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])

    # Plot channel distributions
    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("{} vs {}".format(labels[0], labels[1]))
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))
        
    plt.tight_layout()
    plt.show()

def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    img_hsv = skimage.color.rgb2hsv(rgb=img)
    
    data_h = img_hsv[:,:,0]
    data_s = img_hsv[:,:,1]
    data_v = img_hsv[:,:,2]
    
    return data_h, data_s, data_v

def extract_lab_channels(img):
    """
    Extract LAB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_l: np.ndarray (M, N)
        L channel of input image
    data_a: np.ndarray (M, N)
        A channel of input image
    data_b: np.ndarray (M, N)
        B channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for LAB channels
    data_l = np.zeros((M, N))
    data_a = np.zeros((M, N))
    data_b = np.zeros((M, N))

    img_lab = skimage.color.rgb2lab(rgb=img)
    
    data_l = img_lab[:,:,0]
    data_a = img_lab[:,:,1]
    data_b = img_lab[:,:,2]
    
    return data_l, data_a, data_b

def plot_hsv_channels(img, title):
    hue_channel, saturation_channel, value_channel = extract_hsv_channels(img)

    plt.subplot(1, 3, 1)
    plt.imshow(hue_channel, cmap='hsv')
    plt.title(f'{title}: Hue Channel')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(saturation_channel, cmap='hsv')
    plt.title(f'{title}: Saturation Channel')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(value_channel, cmap='hsv')
    plt.title(f'{title}: Value Channel')
    plt.axis('off')

    plt.show()

class ThresholdRGB:
    def __init__(self, min_blue, max_blue, min_green, max_green, min_red, max_red, type: str = '+'):
        self.min_blue = min_blue
        self.max_blue = max_blue
        self.min_green = min_green
        self.max_green = max_green
        self.min_red = min_red
        self.max_red = max_red
        self.type = type
    
    def __call__(self, img):
        data_red, data_green, data_blue = extract_rgb_channels(img=img)

        mask_red = (data_red >= self.min_red) & (data_red <= self.max_red)
        mask_green = (data_green >= self.min_green) & (data_green <= self.max_green)
        mask_blue = (data_blue >= self.min_blue) & (data_blue <= self.max_blue)

        if self.type == '+':
            return mask_red & mask_green & mask_blue
        elif self.type == '-':
            return ~(mask_red & mask_green & mask_blue)

class ThresholdHSV:
    def __init__(self, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value, type: str = '+'):
        self.min_hue = min_hue
        self.max_hue = max_hue
        self.min_saturation = min_saturation
        self.max_saturation = max_saturation
        self.min_value = min_value
        self.max_value = max_value
        self.type = type

    def __call__(self, img):
        data_h, data_s, data_v = extract_hsv_channels(img=img)

        mask_hue = (data_h >= self.min_hue) & (data_h <= self.max_hue)
        mask_saturation = (data_s >= self.min_saturation) & (data_s <= self.max_saturation)
        mask_value = (data_v >= self.min_value) & (data_v <= self.max_value)

        if self.type == '+':
            return mask_hue & mask_saturation & mask_value
        elif self.type == '-':
            return ~(mask_hue & mask_saturation & mask_value)

class ThresholdLAB:
    def __init__(self, min_l, max_l, min_a, max_a, min_b, max_b, type: str = '+'):
        self.min_l = min_l
        self.max_l = max_l
        self.min_a = min_a
        self.max_a = max_a
        self.min_b = min_b
        self.max_b = max_b
        self.type = type
    def __call__(self, img):
        data_l, data_a, data_b = extract_lab_channels(img=img)

        mask_l = (data_l >= self.min_l) & (data_l <= self.max_l)
        mask_a = (data_a >= self.min_a) & (data_a <= self.max_a)
        mask_b = (data_b >= self.min_b) & (data_b <= self.max_b)

        if self.type == '+':
            return mask_l & mask_a & mask_b
        elif self.type == '-':
            return ~(mask_l & mask_a & mask_b)

def apply_thresholds(img: np.ndarray, thresholds: list, type: str = 'or'):
    """
    Apply threshold to RGB input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    thresholds: list
        List of threshold functions.
    type: str
        Type of combination. Default is 'or'.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N), dtype=np.uint8)

    if type == 'or':
        for threshold in thresholds:
            img_th = img_th | threshold(img)
    elif type == 'and':
        for threshold in thresholds:
            img_th = img_th & threshold(img)
    else:
        raise ValueError('Invalid type. Choose between "or" and "and".')

    return img_th.astype(np.uint8)

def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for closing

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)

    # ------------------
    # Your code here ... 
    # ------------------
    selem = disk(disk_size)
    img_closing = binary_closing(img_th, selem)

    return img_closing

def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    # ------------------
    # Your code here ... 
    # ------------------

    selem = disk(disk_size)
    img_opening = binary_opening(img_th, selem)

    return img_opening

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    img_th_bool = img_th.astype(bool)

    # Remove holes smaller than the specified size
    img_holes_bool = morphology.remove_small_holes(img_th_bool, area_threshold=size)

    # Convert back to original dtype
    img_holes = img_holes_bool.astype(img_th.dtype)



    # ------------------
    # Your code here ... 
    # ------------------

    return img_holes

def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    img_th_obj = img_th.astype(bool)

    # Remove holes smaller than the specified size
    img_holes_obj= morphology.remove_small_objects(img_th_obj, min_size=size)

    # Convert back to original dtype
    img_obj = img_holes_obj.astype(img_th.dtype)

    # ------------------
    # Your code here ... 
    # ------------------

    return img_obj

def find_circles(image, dp=2.1, param1 = 150, param2 = 0.15, blur = 9):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, blur)

    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, dp=dp, minDist=20,
                              param1=param1, param2=param2, minRadius=30, maxRadius=250)
    
    if circles is not None:
        return np.round(circles[0, :]).astype("int")
    else:
        return None

def draw_circles(image, circles):
    img = image.copy()
    if circles is not None:
        for i in circles:
            center = (i[0], i[1])
            radius = i[2]
            
            cv.circle(img, center, radius, (255, 0, 0), 3)
    return img

def coin_extraction(img, background_img, transformation_function, thresholds, pad_bg, processing_size = USE_SIZE_REF, min_area = 600, dp=2.1, param1 = 150, param2 = 0.15, blur = 9):
    '''
    
    '''

    img_no_bg = remove_background(img, background_img)
    img_no_bg_reduced = resize_image(img_no_bg, processing_size)

    img_no_bg_reduced_thresholded = apply_thresholds(img_no_bg_reduced, thresholds)

    processed_img = transformation_function(img_no_bg_reduced_thresholded)

    img_filtered = np.where(processed_img[..., np.newaxis] == 0, 0, img_no_bg_reduced)
    img_no_bg_reduced = (img_filtered/np.max(img_no_bg_reduced) * 500).astype(np.uint8)
    img_no_bg_reduced[img_no_bg_reduced > 255] = 255

    #getting the circles
    circles = find_circles(img_no_bg_reduced, dp=dp, param1 = param1, param2 = param2, blur = blur)

    #when a circle is continaed in another, remove it the contained one
    for i_, (xi, yi, ri) in enumerate(circles):
        for j, (xe, ye, re) in enumerate(circles):
            if i_ != j:
                if ri <= re:
                    distance = math.sqrt((xe - xi) ** 2 + (ye - yi) ** 2)
                    if distance + ri <= re:
                        circles[i_] = (0, 0, 0)
                        break
 
    
    circles = [circle for circle in circles if any(circle != (0, 0, 0))]

    #getting bounding boxes of the circles with a 20% r margin
    bounding_boxes = [(x - r - 0.2*r, y - r - 0.2*r, 2.4*r, 2.4*r) for x, y, r in circles]
    
    #filtering boxes that are too small
    bounding_boxes = [bbox for bbox in bounding_boxes if bbox[2]*bbox[3]>min_area]
    
    
    #reshaping
    rf = img.shape[0]/processing_size[1]
    bounding_boxes = [(x*rf, y*rf, w*rf, h*rf) for x,y,w,h in bounding_boxes]

    coins = [img[int(y) -5 :int(y+h) +5, int(x)-5:int(x+w) +5] for x, y, w, h in bounding_boxes]

    padded_coins = [pad_bg.copy() for  _ in coins]

    pad_h = pad_bg.shape[0]
    pad_w = pad_bg.shape[1]
    middle_x = pad_h//2
    middle_y = pad_w//2

    for i, coin in enumerate(coins):

        coin_w = coin.shape[0]
        coin_h = coin.shape[1]

        if coin_w <= pad_bg.shape[0] and coin_h <= pad_bg.shape[1]: #case coin bounding box is smaller than the background
            low_x = int(middle_x - np.floor(coin_w/2))
            high_x = int(middle_x + (coin_w - np.floor(coin_w/2)))

            low_y = int(middle_y - np.floor(coin_h/2))
            high_y = int(middle_y + (coin_h - np.floor(coin_h/2)))
            
            padded_coins[i][low_x:high_x,low_y:high_y,:] = coin
        else: #case coin bounding box is bigger than the background
            middle_x_coin = coin_w//2
            middle_y_coin = coin_h//2

            low_x = int(middle_x_coin - np.floor(pad_w/2))
            high_x = int(middle_x_coin + (pad_w - np.floor(pad_w/2)))

            low_y = int(middle_y_coin - np.floor(pad_h/2))
            high_y = int(middle_y_coin + (pad_h - np.floor(pad_h/2)))

            padded_coins[i] = coin[low_x:high_x,low_y:high_y,:]

    return padded_coins


def segment_and_save_all_img(imgs_folder_path,
                             background_img,
                             transformation_function,
                             thresholds, 
                             pad_bg,
                             output_folder_path,
                             processing_size = USE_SIZE_REF,
                             min_area = 600,
                             dp=2.1,
                             param1 = 150,
                             param2 = 0.15,
                             blur = 9,
                             plot = False,
                             return_image = False):
    
    for img_path in tqdm(os.listdir(imgs_folder_path)):
        img = read_image(f"{imgs_folder_path}/{img_path}")
        padded_coins = coin_extraction(img,
                                    background_img = background_img,
                                    transformation_function = transformation_function,
                                    thresholds = thresholds,
                                    pad_bg = pad_bg,
                                    processing_size = processing_size,
                                    min_area = min_area,
                                    dp=dp,
                                    param1 = param1,
                                    param2 = param2,
                                    blur = blur)
                                    
        img_id = img_path[:-4]
        dir_path = f"{output_folder_path}/{img_id}"
        os.makedirs(dir_path, exist_ok=True)
        if plot:
            plot_image(img)
        for i,coin in enumerate(padded_coins):
            if plot:
                plot_image(coin)
            if return_image:
                return coin
            save_image(coin,f"{dir_path}/{img_id}_{i}.JPG")

def segment_and_save_all_img_2(imgs_folder_path,
                             pad_bg,
                             output_folder_path,
                             processing_size = USE_SIZE_REF,
                             min_area = 3000,
                             plot = False,
                             return_image = False):
    
    for img_path in tqdm(os.listdir(imgs_folder_path)):
        img = read_image(f"{imgs_folder_path}/{img_path}")
        padded_coins = extract(img, pad_bg, processing_size = processing_size, min_area = min_area)                                
        img_id = img_path[:-4]
        dir_path = f"{output_folder_path}/{img_id}"
        os.makedirs(dir_path, exist_ok=True)
        if plot:
            plot_image(img)
        for i,coin in enumerate(padded_coins):
            if plot:
                plot_image(coin)
            if return_image:
                return coin
            save_image(coin,f"{dir_path}/{img_id}_{i}.JPG")


def extract(img, pad_bg, processing_size = USE_SIZE_REF, min_area = 3000):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (15, 15), 0)
    thresh = cv.adaptiveThreshold(blurred, 255, 
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv.THRESH_BINARY_INV, 11, 3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    morphed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=5)
    edges = cv.Canny(morphed, 50, 150)
    
    # Detect circles using Hough Transform
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=0.8, minDist=700,
                                param1=100, param2=30, minRadius=100, maxRadius=350)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    else: 
        None

    scale_factor = 5

    # Adjust the circle containment check
    for i_, (xi, yi, ri) in enumerate(circles):
        for j, (xe, ye, re) in enumerate(circles):
            if i_ != j:
                if ri <= re:
                    distance = math.sqrt((xe - xi) ** 2 + (ye - yi) ** 2)
                    if distance + ri <= re:
                        circles[i_] = (0, 0, 0)
                        break

    circles = [circle for circle in circles if any(circle != (0, 0, 0))]

    # Getting bounding boxes of the circles with a 20% r margin
    bounding_boxes = [(x - r - 0.25*r, y - r - 0.25*r, 2.6*r, 2.6*r) for x, y, r in circles]

    # Filtering boxes that are too small
    bounding_boxes = [bbox for bbox in bounding_boxes if bbox[2]*bbox[3]>min_area]

    # Reshaping
    rf = (img.shape[0] / processing_size[1]) / scale_factor
    bounding_boxes = [(x*rf, y*rf, w*rf, h*rf) for x,y,w,h in bounding_boxes]

    coins = [img[int(y) - 5 : int(y + h) + 5, int(x) - 5 : int(x + w) + 5] for x, y, w, h in bounding_boxes]

    padded_coins = [pad_bg.copy() for _ in coins]

    pad_h = pad_bg.shape[0]
    pad_w = pad_bg.shape[1]
    middle_x = pad_h // 2
    middle_y = pad_w // 2

    for i, coin in enumerate(coins):
        coin_w = coin.shape[0]
        coin_h = coin.shape[1]

        if coin_w <= pad_bg.shape[0] and coin_h <= pad_bg.shape[1]:  # case coin bounding box is smaller than the background
            low_x = int(middle_x - np.floor(coin_w / 2))
            high_x = int(middle_x + (coin_w - np.floor(coin_w / 2)))

            low_y = int(middle_y - np.floor(coin_h / 2))
            high_y = int(middle_y + (coin_h - np.floor(coin_h / 2)))

            padded_coins[i][low_x:high_x, low_y:high_y, :] = coin
        else:  # case coin bounding box is bigger than the background
            middle_x_coin = coin_w // 2
            middle_y_coin = coin_h // 2

            low_x = int(middle_x_coin - np.floor(pad_w / 2))
            high_x = int(middle_x_coin + (pad_w - np.floor(pad_w / 2)))

            low_y = int(middle_y_coin - np.floor(pad_h / 2))
            high_y = int(middle_y_coin + (pad_h - np.floor(pad_h / 2)))

            padded_coins[i] = coin[low_x:high_x, low_y:high_y, :]

    return padded_coins

def neutral_img_transform(img):
            processed_img = img.copy()
            processed_img = apply_closing(processed_img, 6)
            return processed_img.astype(np.uint8)

def noisy_img_transform(img):
    processed_img = img.copy()
    
    processed_img = remove_objects(processed_img, size = 500)
    processed_img = apply_closing(processed_img, 7)

    return processed_img.astype(np.uint8)