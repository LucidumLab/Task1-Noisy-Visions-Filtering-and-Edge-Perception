import numpy as np
import cv2

def convert_to_grayscale(image):
    """
    Converts a BGR image to grayscale using the weighted sum method.

    :param image: Input color image (BGR format).
    :return: Grayscale image as NumPy array.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Apply grayscale conversion formula
    gray_image = 0.299 * r + 0.587 * g + 0.114 * b
    return gray_image.astype(np.uint8)

def convolve(img, kernel):
    """
    Applies convolution to an image using a given kernel (without using convolve2d).

    :param img: Input grayscale image (NumPy array).
    :param kernel: Filter kernel (NumPy array).
    :return: Convolved image.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2  # Padding for kernel centering
    
    # Pad the image (edge mode to preserve boundary information)
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    
    # Create an empty output image
    output = np.zeros_like(img, dtype=np.float32)
    
    # Perform convolution
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = img_padded[i:i+kernel_height, j:j+kernel_width]  # Extract region
            output[i, j] = np.sum(region * kernel)  # Apply filter
    
    return np.clip(output, 0, 255).astype(np.uint8)  # Normalize output to valid range