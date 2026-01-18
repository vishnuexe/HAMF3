import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import cv2
from scipy.fftpack import fft2, fftshift
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
from tqdm import tqdm
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from skimage import io
from torchvision import transforms
import dlib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as cbar
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

import scipy.special
import PIL.Image as Image
from scipy.ndimage import convolve, minimum_filter, maximum_filter
from scipy.fftpack import fft2, fftshift


# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
print("Facial landmark detector and predictor loaded successfully.")

test_save_path = "Test/features"
test_load_path = test_save_path
test_data_dir = 'test_input'


def compute_frequency_spectrum(image_tensor,fl=0):
    """Compute 2D Fourier Transform magnitude spectrum from PyTorch tensor."""
    # Check if the input is a PyTorch tensor
    if fl==0:
        img = image_tensor.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
    else:
        img = image_tensor
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:,:,0]
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)
    return magnitude_spectrum


def convert_to_ycbcr(image_tensor):
    """Convert PyTorch tensor to YCbCr color space."""
    img = image_tensor.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
    # Undo normalization (approximately) for OpenCV
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    return ycbcr

def convert_to_hsv(image_tensor):
    """Convert PyTorch tensor to HSV color space."""
    img = image_tensor.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
    # Undo normalization (approximately) for OpenCV
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def rich_models_filters(image_tensor):
    """Apply filters from Rich Models for Steganalysis paper to PyTorch tensor."""
    img = image_tensor.numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img[:,:,0]
    
    gray = gray.astype(np.float32)
    
    # Linear filters (gradient-like)
    kernel_x = np.array([[1, 0, -1]])
    kernel_y = np.array([[1], [0], [-1]])
    
    grad_x = convolve(gray, kernel_x)
    grad_y = convolve(gray, kernel_y)
    
    # Non-linear filters (min/max)
    min_filtered = minimum_filter(gray, size=3)
    max_filtered = maximum_filter(gray, size=3)
    
    # Pixel comparison filters
    diff_horizontal = gray[:, 1:] - gray[:, :-1]
    diff_vertical = gray[1:, :] - gray[:-1, :]
    
    # Pad to maintain original size
    diff_horizontal = np.pad(diff_horizontal, ((0, 0), (0, 1)), mode='constant')
    diff_vertical = np.pad(diff_vertical, ((0, 1), (0, 0)), mode='constant')
    
    return {
        'gradient_x': grad_x,
        'gradient_y': grad_y,
        'min_filter': min_filtered,
        'max_filter': max_filtered,
        'diff_horizontal': diff_horizontal,
        'diff_vertical': diff_vertical
    }
    # return grad_x,grad_y,min_filtered,max_filtered,diff_horizontal, diff_vertical



import numpy as np
import cv2
import pywt
import torch

def inpainting_feature_maps(region_img):
    """
    Produce three per‑pixel feature maps useful for inpainting detection.

    Returns a dict:
        {
           "log_edge"   : (H,W) float32  [0-1],
           "noise_resid": (H,W) float32  [0–1],
           "haar_HH"    : (H,W) float32  [0–1]
        }
    """
    region_img = region_img.permute(1,2,0)
    
    # ---- to CPU numpy & grayscale ----
    if isinstance(region_img, torch.Tensor):
        region_img = region_img.detach().cpu().numpy()

    
    
    region_img = region_img.astype(np.float32)
    if region_img.ndim == 3:                         # RGB → gray
        r, g, b = np.moveaxis(region_img, 2, 0)[:3]
        gray = 0.299*r + 0.587*g + 0.114*b
    else:
        gray = region_img

    

    H, W = gray.shape

    # 1️⃣ LoG edge magnitude map
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    log_map = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_map = cv2.normalize(np.abs(log_map), None, 0, 1.0, cv2.NORM_MINMAX)


    

    # 2️⃣ Median‑residual (sensor‑noise) map
    med = cv2.medianBlur(gray, 3)
    resid = cv2.normalize(np.abs(gray - med), None, 0, 1.0, cv2.NORM_MINMAX)

    # 3️⃣ Haar HH wavelet band (high‑freq texture)
    _, (LH, HL, HH) = pywt.dwt2(gray, 'haar')
    HH_up = cv2.resize(np.abs(HH), (W, H), interpolation=cv2.INTER_CUBIC)
    HH_up = cv2.normalize(HH_up, None, 0, 1.0, cv2.NORM_MINMAX)

    return {
        "log_edge"   : log_map.astype(np.float32),
        "noise_resid": resid.astype(np.float32),
        "haar_HH"    : HH_up.astype(np.float32)
    }



def extract_features_from_batch(batch):
    """Extract features from a batch of images."""
    # global images
    images, labels = batch
    batch_features = []
    
    for i in range(images.shape[0]):
        image_tensor = images[i]
        features = {}
        
        # Frequency spectrum
        features['frequency_spectrum'] = compute_frequency_spectrum(image_tensor)
        
        # Color spaces (only for RGB images)
        if image_tensor.shape[0] == 3:  # 3 channels
            features['ycbcr'] = convert_to_ycbcr(image_tensor)
            features['hsv'] = convert_to_hsv(image_tensor)
            # features['ycbcr_fft'] = compute_frequency_spectrum(convert_to_ycbcr(image_tensor),fl=1)
            # features['hsv_fft'] = compute_frequency_spectrum(convert_to_hsv(image_tensor),fl=1)
        # Rich Models filters
        features['rich_filters'] = rich_models_filters(image_tensor)
        # features['gradient_x'], features['gradient_y'], features['min_filter'], features['max_filter'], features['diff_horizontal'], features['diff_vertical'] = rich_models_filters(image_tensor)
        # features['lbp_hist_entropy'] = lbp_hist_entropy_torch(image_tensor)
        # Add label if available
        if labels is not None:
            features['label'] = labels[i].item()
        
        batch_features.append(features)
    
    return batch_features





def extract_features_from_region_batch(regions):
    """Extract features from a batch of images."""
    # global images
    # images, labels = batch
    batch_features = []
    
    for i in range(len(regions)):
        region_tensor = regions[i]
    
        region_features = {}
        
        for key in region_tensor.keys():
            features = {}
            if key == 'label':
                region_features[key] = region_tensor[key]
                continue
            image_tensor = region_tensor[key]

            features['frequency_spectrum'] = compute_frequency_spectrum(image_tensor)
            # features['frequency_spectrum'] = inpainting_feature_maps(image_tensor)['haar_HH']
            
            # Color spaces (only for RGB images)
            if image_tensor.shape[0] == 3:  # 3 channels
                features['ycbcr'] = convert_to_ycbcr(image_tensor)
                features['hsv'] = convert_to_hsv(image_tensor)
                # features['ycbcr_fft'] = compute_frequency_spectrum(convert_to_ycbcr(image_tensor),fl=1)
                # features['hsv_fft'] = compute_frequency_spectrum(convert_to_hsv(image_tensor),fl=1)
            # Rich Models filters
            features['rich_filters'] = rich_models_filters(image_tensor)
            # features['gradient_x'], features['gradient_y'], features['min_filter'], features['max_filter'], features['diff_horizontal'], features['diff_vertical'] = rich_models_filters(image_tensor)
            features['log_edge'] = inpainting_feature_maps(image_tensor)['log_edge']
            features['noise_resid'] = inpainting_feature_maps(image_tensor)['noise_resid']
            features['haar_HH'] = inpainting_feature_maps(image_tensor)['haar_HH']
            
            region_features[key] = features
        batch_features.append(region_features)
            
    return batch_features

def crop_and_extract_features_region(image_tensor, device='cpu'):
    cropped_features_batch = []


    ignore = 0 #to ignore images with no facial regions
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)  # Undo normalization and scale to [0, 255]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for dlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for dlib

    # Detect faces in the image
    faces = detector(image) #{"error": "No faces detected"}
    if len(faces) == 0:
        return print("No faces detected in the image.")

    # Use the first detected face
    face = faces[0]

    plt.imshow(image)
    plt.title("Input image")
    plt.axis("off")
    plt.show()

    # Get facial landmarks
    landmarks = predictor(image, face)
    limage = image.copy()
    # Draw landmarks on the image
    for i in range(68):  # 68 landmarks
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(limage, (x, y), 2, (0, 255, 0), -1)  # Draw a small circle at each landmark

    # Show the image with landmarks
    plt.imshow(limage)
    plt.title("Image with Landmarks")
    plt.axis("off")
    plt.show()
    # Define regions for eyes, nose, lips, etc.
    regions = {
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
        "forehead": list(range(17, 27)),  # Eyebrows plus forehead
        "chin": list(range(4, 13)),  # Combine jawline and lips for chin
        # "nose_bridge": list(range(27, 31)),  # Bridge of the nose
        "upper_lip": list(range(48, 55)),  # Upper lip
        "lower_lip": list(range(55, 60)),  # Lower lip
        "left_cheek": list(range(1, 7)),  # Left cheek
        "right_cheek": list(range(10, 16)),  # Right cheek
        "left_eyebrow": list(range(17, 22)),  # Left eyebrow
        "right_eyebrow": list(range(22, 27)),  # Right eyebrow,
        "jawline_neck": list(range(4, 14))  # Jawline plus neck region
    }

    cropped_tensors = {}

    # Add the entire face as a region
    face_region = image[face.top():face.bottom(), face.left():face.right()]
    face_tensor = transforms.ToTensor()(face_region)
    if face_tensor.shape[1] == 0 or face_tensor.shape[2] == 0: # cropped_features_batch.append({"error": "Face region is empty"})
        return print("No faces detected in the image.")
    # Resize and normalize the tensor
    transform = transforms.Compose([
        transforms.Resize((50, 50)),  # Resize to desired dimensions
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    face_tensor = transform(face_tensor)
    

    cropped_tensors["face"] = face_tensor

    for region_name, indices in regions.items():
        # if ignore == 1:
            # continue
        # Get the coordinates of the region
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]

        # Compute the bounding box for the region
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        if region_name == "jawline_neck":
            y_max = min(image.shape[0], y_max + 50)  # Extend downward by 50 pixels or to the image boundary
    
        # Extend the bounding box for the forehead region
        if region_name == "forehead":
            y_min = max(0, y_min - 50)  # Extend upward by 50 pixels or to the image boundary
        # Crop the region
        cropped_region = image[y_min:y_max, x_min:x_max]

        # Convert the cropped region to a tensor
        cropped_tensor = transforms.ToTensor()(cropped_region)

        
        if cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0: # cropped_features_batch.append({"error": "Face region is empty"})
            ignore = 1
            break
        cropped_tensor = transform(cropped_tensor)
        # Add to the dictionary
        cropped_tensors[region_name] = cropped_tensor
    if ignore == 0:
        cropped_features_batch.append(cropped_tensors)
    ignore = 0
    

    return cropped_features_batch

def crop_and_extract_features(image_tensor, device='cpu',show=1):
    """
    Crop facial features (eyes, nose, lips, etc.) using dlib and extract features.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        list: A list of tuples containing region names and their corresponding cropped tensors.
    """


    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)  # Undo normalization and scale to [0, 255]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for dlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for dlib

    
    # Show the original image

    plt.imshow(image)
    plt.show()
    
    # Detect faces in the image
    faces = detector(image)
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
    
    # Use the first detected face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(image, face)
    limage = image.copy()
    # Draw landmarks on the image
    for i in range(68):  # 68 landmarks
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(limage, (x, y), 2, (0, 255, 0), -1)  # Draw a small circle at each landmark

    # Show the image with landmarks
    plt.imshow(limage)
    plt.title("Image with Landmarks")
    plt.axis("off")
    plt.show()
    
    
    # Define regions for eyes, nose, lips, etc.
    regions = {
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
        "forehead": list(range(17, 27)),  # Eyebrows plus forehead
        "chin": list(range(4, 13)),  # Combine jawline and lips for chin
        # "nose_bridge": list(range(27, 31)),  # Bridge of the nose
        "upper_lip": list(range(48, 55)),  # Upper lip
        "lower_lip": list(range(55, 60)),  # Lower lip
        "left_cheek": list(range(1, 7)),  # Left cheek
        "right_cheek": list(range(10, 16)),  # Right cheek
        "left_eyebrow": list(range(17, 22)),  # Left eyebrow
        "right_eyebrow": list(range(22, 27)),  # Right eyebrow,
        "jawline_neck": list(range(4, 14))  # Jawline plus neck region
    }
    
    cropped_tensors = []
    
    # Add the entire face as a region
    face_region = image[face.top():face.bottom(), face.left():face.right()]
    plt.imshow(face_region)
    plt.title("Cropped Region: face")
    plt.axis("off")
    plt.show()
    

    # Convert the face region to a tensor before resizing
    face_tensor = transforms.ToTensor()(face_region)
    
    # Resize and normalize the tensor
    transform = transforms.Compose([
        transforms.Resize((50, 50)),  # Resize to desired dimensions
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    face_tensor = transform(face_tensor)
    
    # Append the face region to the list
    cropped_tensors.append(("face", face_tensor))
    
    for region_name, indices in regions.items():
        # Get the coordinates of the region
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        
        # Compute the bounding box for the region
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)
        
        # Extend the bounding box for the jawline_neck region
        if region_name == "jawline_neck":
            y_max = min(image.shape[0], y_max + 50)  # Extend downward by 50 pixels or to the image boundary
        
        # Extend the bounding box for the forehead region
        if region_name == "forehead":
            y_min = max(0, y_min - 50)  # Extend upward by 50 pixels or to the image boundary
        
        # Crop the region
        cropped_region = image[y_min:y_max, x_min:x_max]
        if show == 1:
            # Show the cropped region
            plt.imshow(cropped_region)
            plt.title(f"Cropped Region: {region_name}")
            plt.axis("off")
            plt.show()
        
        # Convert the cropped region to a tensor before resizing
        cropped_tensor = transforms.ToTensor()(cropped_region)
        
        # Resize and normalize the tensor
        cropped_tensor = transform(cropped_tensor)
        
        # Append the region name and tensor to the list
        cropped_tensors.append((region_name, cropped_tensor))
    
    return cropped_tensors

def crop_and_extract_features_imagepath(image_path, device='cpu'):
    """
    Crop facial features (eyes, nose, lips, etc.) using dlib and extract features.

    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        list: A list of tuples containing region names and their corresponding cropped tensors.
    """

    image = io.imread(image_path)
    plt.imshow(image)
    plt.show()
    
    # Detect faces in the image
    faces = detector(image)
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
    
    # Use the first detected face
    face = faces[0]
    
    # Get facial landmarks
    landmarks = predictor(image, face)
    
    # Define regions for eyes, nose, lips, etc.
    regions = {
        "left_eye": list(range(36, 42)),
        "right_eye": list(range(42, 48)),
        "nose": list(range(27, 36)),
        "mouth": list(range(48, 68)),
        "forehead": list(range(17, 27)),  # Eyebrows plus forehead
        "chin": list(range(4, 13)),  # Combine jawline and lips for chin
        # "nose_bridge": list(range(27, 31)),  # Bridge of the nose
        "upper_lip": list(range(48, 55)),  # Upper lip
        "lower_lip": list(range(55, 60)),  # Lower lip
        "left_cheek": list(range(1, 7)),  # Left cheek
        "right_cheek": list(range(10, 16)),  # Right cheek
        "left_eyebrow": list(range(17, 22)),  # Left eyebrow
        "right_eyebrow": list(range(22, 27)),  # Right eyebrow,
        "jawline_neck": list(range(4, 14))  # Jawline plus neck region
    }
    
    cropped_tensors = []
    
    # Add the entire face as a region
    face_region = image[face.top():face.bottom(), face.left():face.right()]
    plt.imshow(face_region)
    plt.title("Cropped Region: face")
    plt.axis("off")
    plt.show()
    
    # Convert the face region to a tensor before resizing
    face_tensor = transforms.ToTensor()(face_region)
    
    # Resize and normalize the tensor
    transform = transforms.Compose([
        transforms.Resize((50, 50)),  # Resize to desired dimensions
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    face_tensor = transform(face_tensor)
    
    # Append the face region to the list
    cropped_tensors.append(("face", face_tensor))
    
    for region_name, indices in regions.items():
        # Get the coordinates of the region
        points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]
        
        # Compute the bounding box for the region
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)
        
        # Extend the bounding box for the jawline_neck region
        if region_name == "jawline_neck":
            y_max = min(image.shape[0], y_max + 50)  # Extend downward by 50 pixels or to the image boundary
        
        # Extend the bounding box for the forehead region
        if region_name == "forehead":
            y_min = max(0, y_min - 50)  # Extend upward by 50 pixels or to the image boundary
        
        # Crop the region
        cropped_region = image[y_min:y_max, x_min:x_max]
        
        # Show the cropped region
        plt.imshow(cropped_region)
        plt.title(f"Cropped Region: {region_name}")
        plt.axis("off")
        plt.show()
        
        # Convert the cropped region to a tensor before resizing
        cropped_tensor = transforms.ToTensor()(cropped_region)
        
        # Resize and normalize the tensor
        cropped_tensor = transform(cropped_tensor)
        
        # Append the region name and tensor to the list
        cropped_tensors.append((region_name, cropped_tensor))
    
    return cropped_tensors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def crop_and_extract_features_batch(batch, test=False, device=device):
    """
    Crop facial features (eyes, nose, lips, etc.) for a batch of image tensors using dlib.

    Args:
        batch (tuple): A tuple containing a batch of image tensors and their labels.
        device (torch.device): Device to use (CPU or GPU).

    Returns:
        list: A list of dictionaries, each containing cropped regions for an image.
    """
    if test:
        images, labels, paths = batch
    else:
        images, labels = batch
    cropped_features_batch = []

    for i in range(len(images)):
        ignore = 0 #to ignore images with no facial regions
        image = images[i].permute(1, 2, 0).cpu().numpy()
        image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)  # Undo normalization and scale to [0, 255]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for dlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for dlib

        # Detect faces in the image
        faces = detector(image) #{"error": "No faces detected"}
        if len(faces) == 0:
            continue

        # Use the first detected face
        face = faces[0]

        # Get facial landmarks
        landmarks = predictor(image, face)

        # Define regions for eyes, nose, lips, etc.
        regions = {
            "left_eye": list(range(36, 42)),
            "right_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "mouth": list(range(48, 68)),
            "forehead": list(range(17, 27)),  # Eyebrows plus forehead
            "chin": list(range(4, 13)),  # Combine jawline and lips for chin
            # "nose_bridge": list(range(27, 31)),  # Bridge of the nose
            "upper_lip": list(range(48, 55)),  # Upper lip
            "lower_lip": list(range(55, 60)),  # Lower lip
            "left_cheek": list(range(1, 7)),  # Left cheek
            "right_cheek": list(range(10, 16)),  # Right cheek
            # "left_eyebrow": list(range(17, 22)),  # Left eyebrow
            "left_eyebrow": list(range(22, 27)),  # Left eyebrow
            "right_eyebrow": list(range(22, 27)),  # Right eyebrow,
            "jawline_neck": list(range(4, 14))  # Jawline plus neck region
        }

        cropped_tensors = {}

        # Add the entire face as a region
        face_region = image[face.top():face.bottom(), face.left():face.right()]
        face_tensor = transforms.ToTensor()(face_region)
        if face_tensor.shape[1] == 0 or face_tensor.shape[2] == 0: # cropped_features_batch.append({"error": "Face region is empty"})
            continue
        # Resize and normalize the tensor
        transform = transforms.Compose([
            transforms.Resize((50, 50)),  # Resize to desired dimensions
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        face_tensor = transform(face_tensor)
        

        cropped_tensors["face"] = face_tensor

        for region_name, indices in regions.items():
            # if ignore == 1:
                # continue
            # Get the coordinates of the region
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in indices]

            # Compute the bounding box for the region
            x_min = min(p[0] for p in points)
            x_max = max(p[0] for p in points)
            y_min = min(p[1] for p in points)
            y_max = max(p[1] for p in points)

            if region_name == "jawline_neck":
                y_max = min(image.shape[0], y_max + 50)  # Extend downward by 50 pixels or to the image boundary
        
            # Extend the bounding box for the forehead region
            if region_name == "forehead":
                y_min = max(0, y_min - 50)  # Extend upward by 50 pixels or to the image boundary
            # Crop the region
            cropped_region = image[y_min:y_max, x_min:x_max]

            # Convert the cropped region to a tensor
            cropped_tensor = transforms.ToTensor()(cropped_region)

            
            if cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0: 
                ignore = 1
                break
            cropped_tensor = transform(cropped_tensor)
            # Add to the dictionary
            cropped_tensors[region_name] = cropped_tensor
            cropped_tensors['label'] = labels[i].item()
        if ignore == 0:
            cropped_features_batch.append(cropped_tensors)
        ignore = 0
    

    return cropped_features_batch

def get_feature_representations_region(feature_dict):
    """Convert features to standardized flattened vectors."""
    features = {}
    # Flatten all features in consistent order
    for key in feature_dict.keys():
        if key == 'label':
            continue
        features[key + '_frequency'] = torch.FloatTensor(feature_dict[key]['frequency_spectrum'].flatten()).to(device)
        features[key + '_ycbcr_Y'] = torch.FloatTensor(feature_dict[key]['ycbcr'][:,:,0].flatten()).to(device)
        features[key + '_ycbcr_Cb'] = torch.FloatTensor(feature_dict[key]['ycbcr'][:,:,1].flatten()).to(device)
        features[key + '_ycbcr_Cr'] = torch.FloatTensor(feature_dict[key]['ycbcr'][:,:,2].flatten()).to(device)
        features[key + '_hsv_H'] = torch.FloatTensor(feature_dict[key]['hsv'][:,:,0].flatten()).to(device)
        features[key + '_hsv_S'] = torch.FloatTensor(feature_dict[key]['hsv'][:,:,1].flatten()).to(device)
        features[key + '_hsv_V'] = torch.FloatTensor(feature_dict[key]['hsv'][:,:,2].flatten()).to(device)
        features[key + '_gradient_x'] = torch.FloatTensor(feature_dict[key]['rich_filters']['gradient_x'].flatten()).to(device)
        features[key + '_gradient_y'] = torch.FloatTensor(feature_dict[key]['rich_filters']['gradient_y'].flatten()).to(device)
        features[key + '_min_filter'] = torch.FloatTensor(feature_dict[key]['rich_filters']['min_filter'].flatten()).to(device)
        features[key + '_max_filter'] = torch.FloatTensor(feature_dict[key]['rich_filters']['max_filter'].flatten()).to(device)
        features[key + '_diff_h'] = torch.FloatTensor(feature_dict[key]['rich_filters']['diff_horizontal'].flatten()).to(device)
        features[key + '_diff_v'] = torch.FloatTensor(feature_dict[key]['rich_filters']['diff_vertical'].flatten()).to(device)
        features[key + '_log_edge'] = torch.FloatTensor(feature_dict[key]['log_edge'].flatten()).to(device)
        features[key + '_noise_resid'] = torch.FloatTensor(feature_dict[key]['noise_resid'].flatten()).to(device)
        features[key + '_haar_HH'] = torch.FloatTensor(feature_dict[key]['haar_HH'].flatten()).to(device)

        # features[key + '_label'] = torch.FloatTensor([feature_dict[key]['label']]).to(device)

    # Standardize each feature
    for k in features:
        features[k] = (features[k] - torch.mean(features[k])) / (torch.std(features[k]) + 1e-8)
    
    return features


def reshape_to_feature_dict(flat_tensor, feature_names=None):
    """Reshape flattened tensor (32500) back to dictionary of 13 features (2500 each)."""
    if feature_names is None:
        feature_names = ['face_frequency', 'face_ycbcr_Y', 'face_ycbcr_Cb', 'face_ycbcr_Cr', 'face_hsv_H', 'face_hsv_S', 'face_hsv_V', 'face_gradient_x', 'face_gradient_y', 'face_min_filter', 'face_max_filter', 'face_diff_h', 'face_diff_v', 'face_log_edge', 'face_noise_resid', 'face_haar_HH',
                        'left_eye_frequency', 'left_eye_ycbcr_Y', 'left_eye_ycbcr_Cb', 'left_eye_ycbcr_Cr', 'left_eye_hsv_H', 'left_eye_hsv_S', 'left_eye_hsv_V', 'left_eye_gradient_x', 'left_eye_gradient_y', 'left_eye_min_filter', 'left_eye_max_filter', 'left_eye_diff_h', 'left_eye_diff_v', 'left_eye_log_edge', 'left_eye_noise_resid', 'left_eye_haar_HH',
                        'right_eye_frequency', 'right_eye_ycbcr_Y', 'right_eye_ycbcr_Cb', 'right_eye_ycbcr_Cr', 'right_eye_hsv_H', 'right_eye_hsv_S', 'right_eye_hsv_V', 'right_eye_gradient_x', 'right_eye_gradient_y', 'right_eye_min_filter', 'right_eye_max_filter', 'right_eye_diff_h', 'right_eye_diff_v', 'right_eye_log_edge', 'right_eye_noise_resid', 'right_eye_haar_HH',
                        'nose_frequency', 'nose_ycbcr_Y', 'nose_ycbcr_Cb', 'nose_ycbcr_Cr', 'nose_hsv_H', 'nose_hsv_S', 'nose_hsv_V', 'nose_gradient_x', 'nose_gradient_y', 'nose_min_filter', 'nose_max_filter', 'nose_diff_h', 'nose_diff_v', 'nose_log_edge', 'nose_noise_resid', 'nose_haar_HH',
                        'mouth_frequency', 'mouth_ycbcr_Y', 'mouth_ycbcr_Cb', 'mouth_ycbcr_Cr', 'mouth_hsv_H', 'mouth_hsv_S', 'mouth_hsv_V', 'mouth_gradient_x', 'mouth_gradient_y', 'mouth_min_filter', 'mouth_max_filter', 'mouth_diff_h', 'mouth_diff_v', 'mouth_log_edge', 'mouth_noise_resid', 'mouth_haar_HH',
                        'forehead_frequency', 'forehead_ycbcr_Y', 'forehead_ycbcr_Cb', 'forehead_ycbcr_Cr', 'forehead_hsv_H', 'forehead_hsv_S', 'forehead_hsv_V', 'forehead_gradient_x', 'forehead_gradient_y', 'forehead_min_filter', 'forehead_max_filter', 'forehead_diff_h', 'forehead_diff_v', 'forehead_log_edge', 'forehead_noise_resid', 'forehead_haar_HH',
                        'chin_frequency', 'chin_ycbcr_Y', 'chin_ycbcr_Cb', 'chin_ycbcr_Cr', 'chin_hsv_H', 'chin_hsv_S', 'chin_hsv_V', 'chin_gradient_x', 'chin_gradient_y', 'chin_min_filter', 'chin_max_filter', 'chin_diff_h', 'chin_diff_v', 'chin_log_edge', 'chin_noise_resid', 'chin_haar_HH',
                        'upper_lip_frequency', 'upper_lip_ycbcr_Y', 'upper_lip_ycbcr_Cb', 'upper_lip_ycbcr_Cr', 'upper_lip_hsv_H', 'upper_lip_hsv_S', 'upper_lip_hsv_V', 'upper_lip_gradient_x', 'upper_lip_gradient_y', 'upper_lip_min_filter', 'upper_lip_max_filter', 'upper_lip_diff_h', 'upper_lip_diff_v', 'upper_lip_log_edge', 'upper_lip_noise_resid', 'upper_lip_haar_HH',
                        'lower_lip_frequency', 'lower_lip_ycbcr_Y', 'lower_lip_ycbcr_Cb', 'lower_lip_ycbcr_Cr', 'lower_lip_hsv_H', 'lower_lip_hsv_S', 'lower_lip_hsv_V', 'lower_lip_gradient_x', 'lower_lip_gradient_y', 'lower_lip_min_filter', 'lower_lip_max_filter', 'lower_lip_diff_h', 'lower_lip_diff_v', 'lower_lip_log_edge', 'lower_lip_noise_resid', 'lower_lip_haar_HH',
                        'left_cheek_frequency', 'left_cheek_ycbcr_Y', 'left_cheek_ycbcr_Cb', 'left_cheek_ycbcr_Cr', 'left_cheek_hsv_H', 'left_cheek_hsv_S', 'left_cheek_hsv_V', 'left_cheek_gradient_x', 'left_cheek_gradient_y', 'left_cheek_min_filter', 'left_cheek_max_filter', 'left_cheek_diff_h', 'left_cheek_diff_v', 'left_cheek_log_edge', 'left_cheek_noise_resid', 'left_cheek_haar_HH',
                        'right_cheek_frequency', 'right_cheek_ycbcr_Y', 'right_cheek_ycbcr_Cb', 'right_cheek_ycbcr_Cr', 'right_cheek_hsv_H', 'right_cheek_hsv_S', 'right_cheek_hsv_V', 'right_cheek_gradient_x', 'right_cheek_gradient_y', 'right_cheek_min_filter', 'right_cheek_max_filter', 'right_cheek_diff_h', 'right_cheek_diff_v', 'right_cheek_log_edge', 'right_cheek_noise_resid', 'right_cheek_haar_HH',
                        'left_eyebrow_frequency', 'left_eyebrow_ycbcr_Y', 'left_eyebrow_ycbcr_Cb', 'left_eyebrow_ycbcr_Cr', 'left_eyebrow_hsv_H', 'left_eyebrow_hsv_S', 'left_eyebrow_hsv_V', 'left_eyebrow_gradient_x', 'left_eyebrow_gradient_y', 'left_eyebrow_min_filter', 'left_eyebrow_max_filter', 'left_eyebrow_diff_h', 'left_eyebrow_diff_v', 'left_eyebrow_log_edge', 'left_eyebrow_noise_resid', 'left_eyebrow_haar_HH',
                        'right_eyebrow_frequency', 'right_eyebrow_ycbcr_Y', 'right_eyebrow_ycbcr_Cb', 'right_eyebrow_ycbcr_Cr', 'right_eyebrow_hsv_H', 'right_eyebrow_hsv_S', 'right_eyebrow_hsv_V', 'right_eyebrow_gradient_x', 'right_eyebrow_gradient_y', 'right_eyebrow_min_filter', 'right_eyebrow_max_filter', 'right_eyebrow_diff_h', 'right_eyebrow_diff_v', 'right_eyebrow_log_edge', 'right_eyebrow_noise_resid', 'right_eyebrow_haar_HH',
                        'jawline_neck_frequency', 'jawline_neck_ycbcr_Y', 'jawline_neck_ycbcr_Cb', 'jawline_neck_ycbcr_Cr', 'jawline_neck_hsv_H', 'jawline_neck_hsv_S', 'jawline_neck_hsv_V', 'jawline_neck_gradient_x', 'jawline_neck_gradient_y', 'jawline_neck_min_filter', 'jawline_neck_max_filter', 'jawline_neck_diff_h', 'jawline_neck_diff_v', 'jawline_neck_log_edge', 'jawline_neck_noise_resid', 'jawline_neck_haar_HH']
        
    
    # Reshape to (13, 2500)
    reshaped = flat_tensor.reshape(224, 2500) #16x14,50x50
    
    # Convert to dictionary
    feature_dict = {
        name: reshaped[i] for i, name in enumerate(feature_names)
    }
    
    return feature_dict

def reshape_to_feature_dict_batch(flat_tensor, feature_names=None):
    """
    Reshape flattened tensor (B, 224*2500) into list of dictionaries of 224 features (2500 each).
    Each sample → {feature_name: feature_vector (2500,)}
    """
    if feature_names is None:
        feature_names = ['face_frequency', 'face_ycbcr_Y', 'face_ycbcr_Cb', 'face_ycbcr_Cr', 'face_hsv_H', 'face_hsv_S', 'face_hsv_V', 'face_gradient_x', 'face_gradient_y', 'face_min_filter', 'face_max_filter', 'face_diff_h', 'face_diff_v', 'face_log_edge', 'face_noise_resid', 'face_haar_HH',
                        'left_eye_frequency', 'left_eye_ycbcr_Y', 'left_eye_ycbcr_Cb', 'left_eye_ycbcr_Cr', 'left_eye_hsv_H', 'left_eye_hsv_S', 'left_eye_hsv_V', 'left_eye_gradient_x', 'left_eye_gradient_y', 'left_eye_min_filter', 'left_eye_max_filter', 'left_eye_diff_h', 'left_eye_diff_v', 'left_eye_log_edge', 'left_eye_noise_resid', 'left_eye_haar_HH',
                        'right_eye_frequency', 'right_eye_ycbcr_Y', 'right_eye_ycbcr_Cb', 'right_eye_ycbcr_Cr', 'right_eye_hsv_H', 'right_eye_hsv_S', 'right_eye_hsv_V', 'right_eye_gradient_x', 'right_eye_gradient_y', 'right_eye_min_filter', 'right_eye_max_filter', 'right_eye_diff_h', 'right_eye_diff_v', 'right_eye_log_edge', 'right_eye_noise_resid', 'right_eye_haar_HH',
                        'nose_frequency', 'nose_ycbcr_Y', 'nose_ycbcr_Cb', 'nose_ycbcr_Cr', 'nose_hsv_H', 'nose_hsv_S', 'nose_hsv_V', 'nose_gradient_x', 'nose_gradient_y', 'nose_min_filter', 'nose_max_filter', 'nose_diff_h', 'nose_diff_v', 'nose_log_edge', 'nose_noise_resid', 'nose_haar_HH',
                        'mouth_frequency', 'mouth_ycbcr_Y', 'mouth_ycbcr_Cb', 'mouth_ycbcr_Cr', 'mouth_hsv_H', 'mouth_hsv_S', 'mouth_hsv_V', 'mouth_gradient_x', 'mouth_gradient_y', 'mouth_min_filter', 'mouth_max_filter', 'mouth_diff_h', 'mouth_diff_v', 'mouth_log_edge', 'mouth_noise_resid', 'mouth_haar_HH',
                        'forehead_frequency', 'forehead_ycbcr_Y', 'forehead_ycbcr_Cb', 'forehead_ycbcr_Cr', 'forehead_hsv_H', 'forehead_hsv_S', 'forehead_hsv_V', 'forehead_gradient_x', 'forehead_gradient_y', 'forehead_min_filter', 'forehead_max_filter', 'forehead_diff_h', 'forehead_diff_v', 'forehead_log_edge', 'forehead_noise_resid', 'forehead_haar_HH',
                        'chin_frequency', 'chin_ycbcr_Y', 'chin_ycbcr_Cb', 'chin_ycbcr_Cr', 'chin_hsv_H', 'chin_hsv_S', 'chin_hsv_V', 'chin_gradient_x', 'chin_gradient_y', 'chin_min_filter', 'chin_max_filter', 'chin_diff_h', 'chin_diff_v', 'chin_log_edge', 'chin_noise_resid', 'chin_haar_HH',
                        'upper_lip_frequency', 'upper_lip_ycbcr_Y', 'upper_lip_ycbcr_Cb', 'upper_lip_ycbcr_Cr', 'upper_lip_hsv_H', 'upper_lip_hsv_S', 'upper_lip_hsv_V', 'upper_lip_gradient_x', 'upper_lip_gradient_y', 'upper_lip_min_filter', 'upper_lip_max_filter', 'upper_lip_diff_h', 'upper_lip_diff_v', 'upper_lip_log_edge', 'upper_lip_noise_resid', 'upper_lip_haar_HH',
                        'lower_lip_frequency', 'lower_lip_ycbcr_Y', 'lower_lip_ycbcr_Cb', 'lower_lip_ycbcr_Cr', 'lower_lip_hsv_H', 'lower_lip_hsv_S', 'lower_lip_hsv_V', 'lower_lip_gradient_x', 'lower_lip_gradient_y', 'lower_lip_min_filter', 'lower_lip_max_filter', 'lower_lip_diff_h', 'lower_lip_diff_v', 'lower_lip_log_edge', 'lower_lip_noise_resid', 'lower_lip_haar_HH',
                        'left_cheek_frequency', 'left_cheek_ycbcr_Y', 'left_cheek_ycbcr_Cb', 'left_cheek_ycbcr_Cr', 'left_cheek_hsv_H', 'left_cheek_hsv_S', 'left_cheek_hsv_V', 'left_cheek_gradient_x', 'left_cheek_gradient_y', 'left_cheek_min_filter', 'left_cheek_max_filter', 'left_cheek_diff_h', 'left_cheek_diff_v', 'left_cheek_log_edge', 'left_cheek_noise_resid', 'left_cheek_haar_HH',
                        'right_cheek_frequency', 'right_cheek_ycbcr_Y', 'right_cheek_ycbcr_Cb', 'right_cheek_ycbcr_Cr', 'right_cheek_hsv_H', 'right_cheek_hsv_S', 'right_cheek_hsv_V', 'right_cheek_gradient_x', 'right_cheek_gradient_y', 'right_cheek_min_filter', 'right_cheek_max_filter', 'right_cheek_diff_h', 'right_cheek_diff_v', 'right_cheek_log_edge', 'right_cheek_noise_resid', 'right_cheek_haar_HH',
                        'left_eyebrow_frequency', 'left_eyebrow_ycbcr_Y', 'left_eyebrow_ycbcr_Cb', 'left_eyebrow_ycbcr_Cr', 'left_eyebrow_hsv_H', 'left_eyebrow_hsv_S', 'left_eyebrow_hsv_V', 'left_eyebrow_gradient_x', 'left_eyebrow_gradient_y', 'left_eyebrow_min_filter', 'left_eyebrow_max_filter', 'left_eyebrow_diff_h', 'left_eyebrow_diff_v', 'left_eyebrow_log_edge', 'left_eyebrow_noise_resid', 'left_eyebrow_haar_HH',
                        'right_eyebrow_frequency', 'right_eyebrow_ycbcr_Y', 'right_eyebrow_ycbcr_Cb', 'right_eyebrow_ycbcr_Cr', 'right_eyebrow_hsv_H', 'right_eyebrow_hsv_S', 'right_eyebrow_hsv_V', 'right_eyebrow_gradient_x', 'right_eyebrow_gradient_y', 'right_eyebrow_min_filter', 'right_eyebrow_max_filter', 'right_eyebrow_diff_h', 'right_eyebrow_diff_v', 'right_eyebrow_log_edge', 'right_eyebrow_noise_resid', 'right_eyebrow_haar_HH',
                        'jawline_neck_frequency', 'jawline_neck_ycbcr_Y', 'jawline_neck_ycbcr_Cb', 'jawline_neck_ycbcr_Cr', 'jawline_neck_hsv_H', 'jawline_neck_hsv_S', 'jawline_neck_hsv_V', 'jawline_neck_gradient_x', 'jawline_neck_gradient_y', 'jawline_neck_min_filter', 'jawline_neck_max_filter', 'jawline_neck_diff_h', 'jawline_neck_diff_v', 'jawline_neck_log_edge', 'jawline_neck_noise_resid', 'jawline_neck_haar_HH']
          # your long list (unchanged)

    B = flat_tensor.shape[0]
    reshaped = flat_tensor.view(B, 224, 2500)  # (B, 224, 2500)

    feature_dicts = []
    for b in range(B):
        feature_dict = {
            name: reshaped[b, i] for i, name in enumerate(feature_names)
        }
        feature_dicts.append(feature_dict)

    return feature_dicts  # list of dicts, len = B


def save_model(attention_model, ffn_model, optimizer, epoch, val_acc, filepath):
    """
    Save the complete model state with all necessary components
    
    Args:
        attention_model: Trained attention model
        ffn_model: Trained FFN classifier
        optimizer: Training optimizer
        epoch: Last training epoch
        val_acc: Best validation accuracy
        filepath: Path to save the model (.pth or .pt)
    """
    torch.save({
        'epoch': epoch,
        'attention_state_dict': attention_model.state_dict(),
        'ffn_state_dict': ffn_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'feature_order': [  # Preserve feature processing order
            'frequency', 'ycbcr_Y', 'ycbcr_Cb', 'ycbcr_Cr',
            'hsv_H', 'hsv_S', 'hsv_V', 'gradient_x', 'gradient_y',
            'min_filter', 'max_filter', 'diff_h', 'diff_v', 'log_edge', 'noise_resid', 'haar_HH',
        ]
    }, filepath)
    print(f"Model saved to {filepath} at epoch {epoch} with val_acc {val_acc:.4f}")

# Utility Functions
def print_feature_importance(importance_dict):
    """Print sorted feature importance scores."""
    print("\nFeature Importance Scores:")
    for name, score in sorted(importance_dict.items(), key=lambda x: -x[1]):
        print(f"{name}: {score:.4f}")


def plot_attention_matrix(importance_dict):
    """Visualize feature importance."""
    import matplotlib.pyplot as plt
    names = list(importance_dict.keys())
    scores = list(importance_dict.values())
    
    plt.figure(figsize=(10, 30))
    plt.barh(names, scores)
    plt.xlabel("Attention Importance")
    plt.title("Cross-Feature Attention Weights")
    plt.tight_layout()
    plt.show()

from PIL import Image
from torchvision import transforms

def read_image_to_tensor(image_path):
    """Read an image and convert it to a tensor."""
    transform = transforms.Compose([
        # transforms.Resize((50, 50)),  # Resize to desired dimensions
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    image = Image.open(image_path).convert('RGB')  # Open image and ensure it's RGB
    image_tensor = transform(image)               # Apply transformations
    return image_tensor

def convert_image_to_features(image_tensor, device):
    """
    Convert a single image tensor to precomputed features for testing.

    Args:
        image_tensor (torch.Tensor): The input image tensor (C, H, W).
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        torch.Tensor: Flattened precomputed features.
    """
    # Ensure the image tensor is on the correct device
    # image_tensor = image_tensor.to(device)

    # Extract features
    features = {}
    
    # Frequency spectrum
    features['frequency_spectrum'] = compute_frequency_spectrum(image_tensor)
    
    # Color spaces (only for RGB images)
    if image_tensor.shape[0] == 3:  # 3 channels
        features['ycbcr'] = convert_to_ycbcr(image_tensor)
        features['hsv'] = convert_to_hsv(image_tensor)
        # features['ycbcr_fft'] = compute_frequency_spectrum(convert_to_ycbcr(image_tensor),fl=1)
        # features['hsv_fft'] = compute_frequency_spectrum(convert_to_hsv(image_tensor),fl=1)
    
    # Rich Models filters
    features['rich_filters'] = rich_models_filters(image_tensor)
    features['log_edge'] = inpainting_feature_maps(image_tensor)['log_edge']
    features['noise_resid'] = inpainting_feature_maps(image_tensor)['residual']
    features['haar_HH'] = inpainting_feature_maps(image_tensor)['haar_HH']
    # Convert features to standardized flattened vectors
    feature_tensors = get_feature_representations(features)
    
    # Flatten all features in consistent order
    flat_features = []
    feature_names = [
            'frequency', 'ycbcr_Y', 'ycbcr_Cb', 'ycbcr_Cr',
            'hsv_H', 'hsv_S', 'hsv_V', 'gradient_x', 'gradient_y',
            'min_filter', 'max_filter', 'diff_h', 'diff_v', 'log_edge', 'noise_resid', 'haar_HH'
        ]
    for name in feature_names:
        flat_features.append(feature_tensors[name].cpu().numpy())
    
    return torch.FloatTensor(np.concatenate(flat_features)).to(device)

feature_names = ['face_frequency', 'face_ycbcr_Y', 'face_ycbcr_Cb', 'face_ycbcr_Cr', 'face_hsv_H', 'face_hsv_S', 'face_hsv_V', 'face_gradient_x', 'face_gradient_y', 'face_min_filter', 'face_max_filter', 'face_diff_h', 'face_diff_v', 'face_log_edge', 'face_noise_resid', 'face_haar_HH',
                        'left_eye_frequency', 'left_eye_ycbcr_Y', 'left_eye_ycbcr_Cb', 'left_eye_ycbcr_Cr', 'left_eye_hsv_H', 'left_eye_hsv_S', 'left_eye_hsv_V', 'left_eye_gradient_x', 'left_eye_gradient_y', 'left_eye_min_filter', 'left_eye_max_filter', 'left_eye_diff_h', 'left_eye_diff_v', 'left_eye_log_edge', 'left_eye_noise_resid', 'left_eye_haar_HH',
                        'right_eye_frequency', 'right_eye_ycbcr_Y', 'right_eye_ycbcr_Cb', 'right_eye_ycbcr_Cr', 'right_eye_hsv_H', 'right_eye_hsv_S', 'right_eye_hsv_V', 'right_eye_gradient_x', 'right_eye_gradient_y', 'right_eye_min_filter', 'right_eye_max_filter', 'right_eye_diff_h', 'right_eye_diff_v', 'right_eye_log_edge', 'right_eye_noise_resid', 'right_eye_haar_HH',
                        'nose_frequency', 'nose_ycbcr_Y', 'nose_ycbcr_Cb', 'nose_ycbcr_Cr', 'nose_hsv_H', 'nose_hsv_S', 'nose_hsv_V', 'nose_gradient_x', 'nose_gradient_y', 'nose_min_filter', 'nose_max_filter', 'nose_diff_h', 'nose_diff_v', 'nose_log_edge', 'nose_noise_resid', 'nose_haar_HH',
                        'mouth_frequency', 'mouth_ycbcr_Y', 'mouth_ycbcr_Cb', 'mouth_ycbcr_Cr', 'mouth_hsv_H', 'mouth_hsv_S', 'mouth_hsv_V', 'mouth_gradient_x', 'mouth_gradient_y', 'mouth_min_filter', 'mouth_max_filter', 'mouth_diff_h', 'mouth_diff_v', 'mouth_log_edge', 'mouth_noise_resid', 'mouth_haar_HH',
                        'forehead_frequency', 'forehead_ycbcr_Y', 'forehead_ycbcr_Cb', 'forehead_ycbcr_Cr', 'forehead_hsv_H', 'forehead_hsv_S', 'forehead_hsv_V', 'forehead_gradient_x', 'forehead_gradient_y', 'forehead_min_filter', 'forehead_max_filter', 'forehead_diff_h', 'forehead_diff_v', 'forehead_log_edge', 'forehead_noise_resid', 'forehead_haar_HH',
                        'chin_frequency', 'chin_ycbcr_Y', 'chin_ycbcr_Cb', 'chin_ycbcr_Cr', 'chin_hsv_H', 'chin_hsv_S', 'chin_hsv_V', 'chin_gradient_x', 'chin_gradient_y', 'chin_min_filter', 'chin_max_filter', 'chin_diff_h', 'chin_diff_v', 'chin_log_edge', 'chin_noise_resid', 'chin_haar_HH',
                        'upper_lip_frequency', 'upper_lip_ycbcr_Y', 'upper_lip_ycbcr_Cb', 'upper_lip_ycbcr_Cr', 'upper_lip_hsv_H', 'upper_lip_hsv_S', 'upper_lip_hsv_V', 'upper_lip_gradient_x', 'upper_lip_gradient_y', 'upper_lip_min_filter', 'upper_lip_max_filter', 'upper_lip_diff_h', 'upper_lip_diff_v', 'upper_lip_log_edge', 'upper_lip_noise_resid', 'upper_lip_haar_HH',
                        'lower_lip_frequency', 'lower_lip_ycbcr_Y', 'lower_lip_ycbcr_Cb', 'lower_lip_ycbcr_Cr', 'lower_lip_hsv_H', 'lower_lip_hsv_S', 'lower_lip_hsv_V', 'lower_lip_gradient_x', 'lower_lip_gradient_y', 'lower_lip_min_filter', 'lower_lip_max_filter', 'lower_lip_diff_h', 'lower_lip_diff_v', 'lower_lip_log_edge', 'lower_lip_noise_resid', 'lower_lip_haar_HH',
                        'left_cheek_frequency', 'left_cheek_ycbcr_Y', 'left_cheek_ycbcr_Cb', 'left_cheek_ycbcr_Cr', 'left_cheek_hsv_H', 'left_cheek_hsv_S', 'left_cheek_hsv_V', 'left_cheek_gradient_x', 'left_cheek_gradient_y', 'left_cheek_min_filter', 'left_cheek_max_filter', 'left_cheek_diff_h', 'left_cheek_diff_v', 'left_cheek_log_edge', 'left_cheek_noise_resid', 'left_cheek_haar_HH',
                        'right_cheek_frequency', 'right_cheek_ycbcr_Y', 'right_cheek_ycbcr_Cb', 'right_cheek_ycbcr_Cr', 'right_cheek_hsv_H', 'right_cheek_hsv_S', 'right_cheek_hsv_V', 'right_cheek_gradient_x', 'right_cheek_gradient_y', 'right_cheek_min_filter', 'right_cheek_max_filter', 'right_cheek_diff_h', 'right_cheek_diff_v', 'right_cheek_log_edge', 'right_cheek_noise_resid', 'right_cheek_haar_HH',
                        'left_eyebrow_frequency', 'left_eyebrow_ycbcr_Y', 'left_eyebrow_ycbcr_Cb', 'left_eyebrow_ycbcr_Cr', 'left_eyebrow_hsv_H', 'left_eyebrow_hsv_S', 'left_eyebrow_hsv_V', 'left_eyebrow_gradient_x', 'left_eyebrow_gradient_y', 'left_eyebrow_min_filter', 'left_eyebrow_max_filter', 'left_eyebrow_diff_h', 'left_eyebrow_diff_v', 'left_eyebrow_log_edge', 'left_eyebrow_noise_resid', 'left_eyebrow_haar_HH',
                        'right_eyebrow_frequency', 'right_eyebrow_ycbcr_Y', 'right_eyebrow_ycbcr_Cb', 'right_eyebrow_ycbcr_Cr', 'right_eyebrow_hsv_H', 'right_eyebrow_hsv_S', 'right_eyebrow_hsv_V', 'right_eyebrow_gradient_x', 'right_eyebrow_gradient_y', 'right_eyebrow_min_filter', 'right_eyebrow_max_filter', 'right_eyebrow_diff_h', 'right_eyebrow_diff_v', 'right_eyebrow_log_edge', 'right_eyebrow_noise_resid', 'right_eyebrow_haar_HH',
                        'jawline_neck_frequency', 'jawline_neck_ycbcr_Y', 'jawline_neck_ycbcr_Cb', 'jawline_neck_ycbcr_Cr', 'jawline_neck_hsv_H', 'jawline_neck_hsv_S', 'jawline_neck_hsv_V', 'jawline_neck_gradient_x', 'jawline_neck_gradient_y', 'jawline_neck_min_filter', 'jawline_neck_max_filter', 'jawline_neck_diff_h', 'jawline_neck_diff_v', 'jawline_neck_log_edge', 'jawline_neck_noise_resid', 'jawline_neck_haar_HH']
        

def convert_image_to_features_region(image_tensor, device):
    """
    Convert a single image tensor to precomputed features for testing.

    Args:
        image_tensor (torch.Tensor): The input image tensor (C, H, W).
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        torch.Tensor: Flattened precomputed features.
    """

    features = {}
    regions = crop_and_extract_features_region(image_tensor)
    for region_name, region_tensor in regions[0].items():
        features[region_name] = {}
        features[region_name]['frequency_spectrum'] = compute_frequency_spectrum(region_tensor)
        features[region_name]['ycbcr'] = convert_to_ycbcr(region_tensor)
        features[region_name]['hsv'] = convert_to_hsv(region_tensor)
        features[region_name]['rich_filters'] = rich_models_filters(region_tensor)
        features[region_name]['log_edge'] = inpainting_feature_maps(region_tensor)['log_edge']
        features[region_name]['noise_resid'] = inpainting_feature_maps(region_tensor)['noise_resid']
        features[region_name]['haar_HH'] = inpainting_feature_maps(region_tensor)['haar_HH']

    feature_tensors = get_feature_representations_region(features)
    
    flat_features = []

    for name in feature_names:
        flat_features.append(feature_tensors[name].cpu().numpy())
    
    return torch.FloatTensor(np.concatenate(flat_features)).to(device)



def image_to_tensor(image_path):
    """
    Convert an image to a tensor given its path.

    Args:
        image_path (str): Path to the image.

    Returns:
        torch.Tensor: The image as a tensor.
    """
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # Resize to desired dimensions
        transforms.ToTensor(),         # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    image = Image.open(image_path).convert('RGB')  # Open image and ensure it's RGB
    image_tensor = transform(image)               # Apply transformations
    return image_tensor


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def display_attention_map(attention_scores, x_labels=None, y_labels=None, title="Attention Map", figsize=(40, 40), cmap='viridis'):
    """
    Displays an attention map using a heatmap.

    Parameters:
    - attention_scores (np.ndarray or torch.Tensor): 2D matrix of attention scores.
    - x_labels (list of str): Labels for the columns (e.g., keys or target positions).
    - y_labels (list of str): Labels for the rows (e.g., queries or source positions).
    - title (str): Title of the heatmap.
    - figsize (tuple): Figure size.
    - cmap (str): Colormap for the heatmap.
    """
    if 'torch' in str(type(attention_scores)):
        attention_scores = attention_scores.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    sns.heatmap(attention_scores, xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, annot=False)
    plt.title(title)
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.show()


from tqdm import tqdm


def precompute_features_region(image_loader, save_path, device):
    all_features = []
    all_labels = []
    # global regions,batch_features,feat_tensors
    os.makedirs(save_path, exist_ok=True)
    for batch_idx, batch in enumerate(tqdm(image_loader, desc="Precomputing Features")):
        
        regions = crop_and_extract_features_batch(batch)
        
        batch_features = extract_features_from_region_batch(regions)
        
        for features in batch_features:
            # Process features
            feat_tensors = get_feature_representations_region(features)
            
            # Flatten all features in consistent order
            flat_features = []
            for name in feature_names:
                
                flat_features.append(feat_tensors[name].cpu().numpy())
            
            all_features.append(np.concatenate(flat_features))
            all_labels.append(features.get('label', 0))

    np.save(os.path.join(save_path, 'features.npy'), np.array(all_features))
    np.save(os.path.join(save_path, 'labels.npy'), np.array(all_labels))
    print(f"Features saved to {save_path}")


def create_ffn_(input_dim, hidden_dims=[128,64]): 
    layers = nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.Dropout(dropout),
        nn.Linear(hidden_dims[1], 2),
    )
    return layers


class AttentionMechanism2(nn.Module):
    """Cross-feature attention mechanism with batch support."""
    def __init__(self, feature_dim=2500, embed_dim=1024):########################dimension_change
        super().__init__()
        self.Q = nn.Linear(feature_dim, embed_dim)
        self.K = nn.Linear(feature_dim, embed_dim)
        self.V = nn.Linear(feature_dim, embed_dim)
        
    def forward(self, batched_feature_tensors):
        """
        batched_feature_tensors: list of dicts, each with 13 entries (from reshape_to_feature_dict)
        
        Returns:
            attended: (B, 13 * embed_dim)
            attention_weights: list of (13, 13) attention matrices
        """
        attended_list = []
        attn_weights_list = []

        for feature_tensors in batched_feature_tensors:
            all_features = torch.stack(list(feature_tensors.values()))  # (13, feature_dim)

            # Project to query, key, value
            queries = self.Q(all_features)  # (13, embed_dim)
            keys    = self.K(all_features)  # (13, embed_dim)
            values  = self.V(all_features)  # (13, embed_dim)

            scale = torch.sqrt(torch.tensor(self.Q.out_features, dtype=torch.float32, device=queries.device))
            scores = torch.matmul(queries, keys.T) / scale
            attention = F.softmax(scores, dim=-1)            # (13, 13)
            attended = torch.matmul(attention, values)       # (13, embed_dim)

            attended_list.append(attended.flatten())         # (13 * embed_dim,)
            attn_weights_list.append(attention)

        attended_batch = torch.stack(attended_list)          # (B, 13 * embed_dim)

        return attended_batch, attn_weights_list
from collections import defaultdict
import numpy as np

from collections import defaultdict
import numpy as np

def feat_to_region_attribute_scores(feature_importance):
    region_list = [
        'face', 'left_eye', 'right_eye', 'nose', 'mouth', 'forehead', 'chin',
        'upper_lip', 'lower_lip', 'left_cheek', 'right_cheek',
        'left_eyebrow', 'right_eyebrow', 'jawline_neck'
    ]

    feature_list = [
        'frequency', 'ycbcr_Y', 'ycbcr_Cb', 'ycbcr_Cr', 'hsv_H', 'hsv_S', 'hsv_V',
        'gradient_x', 'gradient_y', 'min_filter', 'max_filter', 'diff_h', 'diff_v',
        'log_edge', 'noise_resid', 'haar_HH'
    ]

    region_attr_scores = defaultdict(list)
    flat_region_attr_scores = {}  # (region_feature) -> score

    for key, value in feature_importance.items():
        matched_region = None
        matched_feature = None

        for region in region_list:
            if region in key:
                matched_region = region
                break

        for feature in feature_list:
            if feature in key:
                matched_feature = feature
                break

        if matched_region and matched_feature:
            region_attr_scores[matched_region].append((matched_feature, value))
            flat_key = f"{matched_region}_{matched_feature}"
            flat_region_attr_scores[flat_key] = value
        else:
            print(f"⚠️ No match found for key: {key}")

    # aggregated region scores (optional, keep for other use)
    region_score_dict = {
        region: np.mean([score for _, score in attr_scores])
        for region, attr_scores in region_attr_scores.items()
    }

    return region_attr_scores, region_score_dict, flat_region_attr_scores



def load_model(attention, ffn, optimizer, checkpoint_path, device='cuda'):
    # Load the checkpoint
    # global checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Restore model and optimizer states
    attention.load_state_dict(checkpoint['attention_state_dict'])
    ffn.load_state_dict(checkpoint['ffn_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded from", checkpoint_path)
    return attention, ffn, optimizer

def create_ffn(input_dim, hidden_dims=[256,128,64], dropout=0.2, leaky_relu=0.2):
    layers = nn.Sequential(
        nn.Linear(input_dim, hidden_dims[0]),
        # nn.BatchNorm1d(hidden_dims[0]),
        nn.LeakyReLU(leaky_relu),
        nn.Dropout(0.5),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        # nn.BatchNorm1d(hidden_dims[0]),
        nn.LeakyReLU(leaky_relu),
        # nn.Dropout(0.4),
        # nn.Linear(hidden_dims[1], hidden_dims[2]),
        # nn.BatchNorm1d(hidden_dims[1]),
        # nn.LeakyReLU(leaky_relu),
        nn.Dropout(dropout),
        nn.Linear(hidden_dims[1], 2),
        # nn.Sigmoid()
        nn.Softmax(dim=1)
    )
    return layers
class FeatureDataset(Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

def gen_outputs_region(attention, ffn):

    test_img_path = test_data_dir+"/test_image/" + os.listdir(test_data_dir+"/test_image/")[0]

    # Define transformations for data augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 (adjustable)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)

    # Get class names
    class_names = dataset.classes

    test_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    precompute_features_region(test_loader, test_save_path, device)

    test_feature_loader = DataLoader(
        FeatureDataset(test_load_path+'/features.npy', test_load_path+'/labels.npy'),
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    feat_tensors_r, attended, attended_region_feature, attention_scores, attention_diagonals, feature_importance, prob_score = test_model_computed_inpaint_batch((attention, ffn), test_feature_loader, device, thresh=0.5)



    def extract_region_from_feature_name(feature_name):
        # for feature_name in feature_names:
        feature_list = ['face', 'left_eye', 'right_eye', 'nose', 'mouth', 'forehead', 'chin',
                    'upper_lip', 'lower_lip', 'left_cheek', 'right_cheek', 'left_eyebrow', 'right_eyebrow', 'jawline_neck']
        attri_list = ['left', 'right', 'upper', 'lower', 'jawline']

        if feature_name.split('_')[0] in feature_list:
            region = feature_name.split('_')[0]
            feature = '_'.join(feature_name.split('_')[1:])
        elif feature_name.split('_')[0] in attri_list:
            if '_'.join(feature_name.split('_')[2:]) == 'frequency':
                feature = '_'.join(feature_name.split('_')[2:])
                region = '_'.join(feature_name.split('_')[:-1])

            else:
                region = '_'.join(feature_name.split('_')[:-2])
                feature = '_'.join(feature_name.split('_')[2:])

        return feature, region 



    def highlight_top_attention_regions_with_colorbar(
        image_path,
        region_score_dict,
        top_k=3,
        shape_predictor_path='shape_predictor_68_face_landmarks.dat',
        alpha=0.4,
        blur_kernel=(21, 21),
        blur_sigma=10
    ):
        global preds_binary
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(shape_predictor_path)
        dets = detector(image_rgb, 1)
        if len(dets) == 0:
            print("No face detected.")
            return
        shape = predictor(image_rgb, dets[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

        # Facial region indices
        region_indices = {
                "left_eye": list(range(36, 42)),
                "right_eye": list(range(42, 48)),
                "nose": list(range(27, 36)),
                "mouth": list(range(48, 68)),
                "forehead": list(range(17, 27)),  # Eyebrows plus forehead
                "chin": list(range(4, 13)),  # Combine jawline and lips for chin
                # "nose_bridge": list(range(27, 31)),  # Bridge of the nose
                "upper_lip": list(range(48, 55)),  # Upper lip
                "lower_lip": list(range(55, 60)),  # Lower lip
                "left_cheek": list(range(1, 7)),  # Left cheek
                "right_cheek": list(range(10, 16)),  # Right cheek
                "left_eyebrow": list(range(17, 22)),  # Left eyebrow
                # "left_eyebrow": list(range(22, 27)),  # Left eyebrow  
                "right_eyebrow": list(range(22, 27)),  # Right eyebrow,
                "jawline_neck": list(range(4, 14))  # Jawline plus neck region
            }
        region_attr_scores, region_score_dict, flat_region_attr_scores = feat_to_region_attribute_scores(feature_importance)

        # Sort region-attribute pairs directly
        sorted_region_attrs = sorted(flat_region_attr_scores.items(), key=lambda x: x[1], reverse=True)

        # Remove unwanted regions
        filtered_region_attrs = []
        for key, score in sorted_region_attrs:
            if not any(x in key for x in ["lower_lip", "upper_lip", "left_eyebrow", "right_eyebrow"]):
                filtered_region_attrs.append((key, score))

        top_region_attrs = filtered_region_attrs[:top_k]


        scores = np.array([score for _, score in top_region_attrs])
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


        top_regions = top_region_attrs

        cmap = cm.get_cmap("jet")
        mask = np.zeros_like(image_rgb, dtype=np.uint8)


        drawn_regions = set()  # To track drawn regions

        for (region_name, score), norm_score in zip(top_regions, norm_scores):
            feature_def, regionn = extract_region_from_feature_name(region_name)

            if regionn not in region_indices or regionn in drawn_regions:
                continue  # Skip if already colored

            indices = region_indices[regionn]
            points = landmarks[indices]

            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)

            if regionn == "forehead":
                y_min = max(0, y_min - 50)
            elif regionn == "jawline_neck":
                y_max = min(image.shape[0], y_max + 50)
            else:
                x_max = min(image.shape[1], x_max + 10)
                x_min = max(0, x_min - 10)
                y_max = min(image.shape[0], y_max + 10)
                y_min = max(0, y_min - 10)
            if regionn == "lower_lip" or regionn == "upper_lip" or regionn == "left_eyebrow" or regionn == "right_eyebrow":
                continue
            rgba = cmap(norm_score)
            rgb_color = tuple(int(255 * c) for c in rgba[:3])

            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), rgb_color, thickness=-1)

            drawn_regions.add(regionn)  

        mask = cv2.GaussianBlur(mask, blur_kernel, blur_sigma)
        blended = cv2.addWeighted(image_rgb, 1.0, mask, alpha, 0)

        # Plotting
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(blended)
        region_list = ", ".join([r for r, _ in top_regions])
        for i, (region, score) in enumerate(top_regions):
            print(f"Region {i + 1}: {region}, Score: {score}")
        descriptions = []

        preds_binary = torch.argmax(prob_score, dim=1, keepdim=True)
        if preds_binary == 1:
            print("Classified as: Real")
            descriptions.append("Classified as: Real")
        else:
            print("Classified as: Inpainted")
            descriptions.append("Classified as: Inpainted")
        if preds_binary == 1:
            print("No regions found with inconsistency")
            descriptions.append("No regions found with inconsistency")

        else:
            for i, (region, score) in enumerate(top_regions):
                feature_def, regionn = extract_region_from_feature_name(region)

                if regionn == "left_eye":
                    regionnn = "Left Eye"
                elif regionn == "right_eye":
                    regionnn = "Right Eye"
                elif regionn == "left_eyebrow":
                    regionnn = "Left Eyebrow"
                elif regionn == "right_eyebrow":
                    regionnn = "Right Eyebrow"
                elif regionn == "left_cheek":
                    regionnn = "Left Cheek"
                elif regionn == "right_cheek":
                    regionnn = "Right Cheek"
                elif regionn == "upper_lip":
                    regionnn = "Upper Lip"
                elif regionn == "lower_lip":
                    regionnn = "Lower Lip"
                elif regionn == "nose":
                    regionnn = "Nose"
                elif regionn == "mouth":
                    regionnn = "Mouth"
                elif regionn == "forehead":
                    regionnn = "Forehead"
                elif regionn == "chin":
                    regionnn = "Chin"
                elif regionn == "jawline_neck":
                    regionnn = "Jawline and Neck"
                elif regionn == "face":
                    regionnn = "Face"
                print(regionnn,"has inconsistency",feature_def)
                descriptions.append(f"{regionnn} has inconsistency {feature_def}")

        for i, description in enumerate(descriptions):
            ax.text(
                1.62, 
                0.7 - i * 0.1,  # Adjust vertical position for each line
                description, 
                transform=ax.transAxes, 
                fontsize=24.5, 
                verticalalignment='center', 
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            )

        ax.axis("off")

        norm = plt.Normalize(vmin=norm_scores.min(), vmax=norm_scores.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm,ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attention Score", rotation=270, labelpad=15)

        output_path = 'output/'+str(os.listdir(test_data_dir+"/test_image/")[0])+"output_image.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=1000)
        plt.tight_layout()
        plt.show()


    sample_image = read_image_to_tensor(test_img_path)# Get a sample image
    ft = convert_image_to_features_region(sample_image, device)
    if preds_binary == 0:
        highlight_top_attention_regions_with_colorbar(test_img_path, feature_importance,alpha=0.8, top_k=3,blur_sigma=0)

    return feat_tensors_r, attended, attended_region_feature, attention_scores, attention_diagonals, feature_importance, prob_score, test_feature_loader
def test_model_computed_inpaint_batch(model, data_loader, device,thresh=0.5):
    """
    Evaluate model on validation data and return metrics
    
    Args:
        model: Tuple containing (attention_model, ffn_model)
        data_loader: Validation DataLoader
        device: torch device (cpu or cuda)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # global feat_tensors_r, feat_tensors,attn_diag
    attention_model, ffn_model = model
    attention_model.eval()
    ffn_model.eval()
    feature_importance = {}
    attended_region_feature = {}
    all_preds = []
    all_labels = []
    all_probs = []
    epoch_acc = 0
    # global y_true, y_pred, y_probs,label
    # with torch.no_grad():
    for batch in data_loader:
        batch_features, labels = batch
        batch_features = batch_features.to(device)
        labels = labels.to(device)
        feat_tensors_r = reshape_to_feature_dict_batch(batch_features)
        attended, attn_weights = attention_model(feat_tensors_r)
        
        # attn_diag = torch.diag(attn_weights[0]).detach().cpu()
        preds = ffn_model(attended)
        preds_binary = torch.argmax(preds, dim=1, keepdim=True)
        print(preds)

        if preds_binary == 1:
            plabel = "Real"
        else:
            plabel = "Inpainted"

        print('Predicted Label:', plabel)
        
        print('Confidence Score:', preds[0][preds_binary].cpu().item())

        attended_copy = attended.clone().detach()  # detached copy
        attn_weights_copy = attn_weights[0].clone()  # detached copy

        attn_diag = torch.diag(attn_weights_copy).detach().cpu()
        for i, name in enumerate(feature_names):
            feature_importance[name] = attn_diag[i].item()

        for i, name in enumerate(feature_names):
            attended_region_feature[name] = attended_copy[0].reshape(224, -1)[i].cpu().detach().numpy()
            

    return feat_tensors_r, attended, attended_region_feature, attn_weights, attn_diag, feature_importance, preds