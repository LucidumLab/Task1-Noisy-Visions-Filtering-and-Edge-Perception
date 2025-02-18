import numpy as np
from utils import convolve
def add_uniform_noise(img, intensity=50):
    """
    Adds uniform noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param intensity: Maximum noise level (default=50).
    :return: Noisy image.
    """
    noise = np.random.uniform(-intensity, intensity, img.shape)  # Generate noise
    noisy_img = img.astype(np.int16) + noise  # Convert to int16 to avoid overflow
    noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values remain in range
    return noisy_img.astype(np.uint8)  # Convert back to uint8

def add_gaussian_noise(img, mean=0, std=25):
    """
    Adds Gaussian noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param mean: Mean of the Gaussian noise (default=0).
    :param std: Standard deviation of the noise (default=25).
    :return: Noisy image.
    """
    noise = np.random.normal(mean, std, img.shape)  # Generate Gaussian noise
    noisy_img = img.astype(np.int16) + noise  # Convert to int16 to prevent overflow
    noisy_img = np.clip(noisy_img, 0, 255)  # Ensure pixel values remain in range
    return noisy_img.astype(np.uint8)  # Convert back to uint8

def add_salt_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02):
    """
    Adds salt & pepper noise to an image using only NumPy.
    
    :param img: Input grayscale image (NumPy array).
    :param salt_prob: Probability of salt noise (default=0.02).
    :param pepper_prob: Probability of pepper noise (default=0.02).
    :return: Noisy image.
    """
    noisy_img = np.copy(img)  # Copy original image
    
    # Generate random mask for salt (white) noise
    salt_mask = np.random.rand(*img.shape) < salt_prob
    noisy_img[salt_mask] = 255  # Assign maximum intensity
    
    # Generate random mask for pepper (black) noise
    pepper_mask = np.random.rand(*img.shape) < pepper_prob
    noisy_img[pepper_mask] = 0  # Assign minimum intensity
    
    return noisy_img


def apply_average_filter(img, kernel_size=3):
    """
    Applies an average filter using a kernel of size `kernel_size x kernel_size`.

    :param img: Input grayscale image (NumPy array).
    :param kernel_size: Size of the averaging kernel (default=3).
    :return: Filtered image.
    """
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return convolve(img, kernel)

def gaussian_kernel(size, sigma=1.0):
    """
    Generates a Gaussian kernel.

    :param size: Kernel size (odd number).
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: Normalized Gaussian kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)  # Normalize kernel to sum to 1

def apply_gaussian_filter(img, kernel_size=3, sigma=1.0):
    """
    Applies a Gaussian filter.

    :param img: Input grayscale image (NumPy array).
    :param kernel_size: Size of the Gaussian kernel (default=3).
    :param sigma: Standard deviation (default=1.0).
    :return: Filtered image.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve(img, kernel)

def apply_median_filter(img, kernel_size=3):
    """
    Applies a median filter.

    :param img: Input grayscale image (NumPy array).
    :param kernel_size: Size of the median filter kernel (default=3).
    :return: Filtered image.
    """
    pad = kernel_size // 2
    img_padded = np.pad(img, pad, mode='edge')
    output = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_padded[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.median(region)  # Apply median filtering

    return output
