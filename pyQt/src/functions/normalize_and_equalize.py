import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions.histogram_functions import compute_histogram, compute_cdf, normalize, apply_histogram_equalization

# def calculate_histogram(image):
#     """Calculates the histogram of a grayscale image."""
#     height, width = image.shape
#     histogram = np.zeros(256, dtype=int)

#     # Calculate histogram
#     for y in range(height):
#         for x in range(width):
#             intensity = image[y, x]
#             histogram[intensity] += 1
#     return histogram

# def calculate_cdf(histogram):
#     """Calculates the Cumulative Distribution Function (CDF) from a histogram."""
#     cdf = np.zeros(256, dtype=float)
#     cumulative_sum = 0

#     # Calculate running sum of histogram values (CDF)
#     for i in range(256):
#         cumulative_sum += histogram[i]
#         cdf[i] = cumulative_sum
#     return cdf

# def normalize_cdf(cdf, total_pixels):
#     """Normalizes the CDF to the range 0-255."""
#     cdf_normalized = (cdf * 255) / total_pixels
#     return cdf_normalized.astype(np.uint8)

# def apply_equalization(image, cdf_normalized):
#      """Applies histogram equalization using the normalized CDF."""
#      # Map each pixel's intensity to its new, equalized intensity
#      equalized_img = cdf_normalized[image]
#      return equalized_img

def equalize_grayscale_image(image):
    """Equalizes a grayscale image."""
    histogram = compute_histogram(image)
    cdf = compute_cdf(histogram)
    #total_pixels = image.shape[0] * image.shape[1]
    cdf_normalized = normalize(cdf)
    equalized_image = apply_histogram_equalization(image, cdf_normalized)
    return equalized_image

def equalize_color_image(image):
    """Equalizes a color image (BGR)."""
    height, width, channels = image.shape
    equalized_image = np.zeros_like(image)

    # Process each color channel independently
    for c in range(channels):
        channel = image[:, :, c]
        equalized_channel = equalize_grayscale_image(channel) 
        equalized_image[:, :, c] = equalized_channel
    return equalized_image

def normalize_grayscale_image(image):
    """Normalizes a grayscale image."""
    min_val = np.min(image)
    max_val = np.max(image)
    if min_val != max_val:
        normalized_image = ((image - min_val) * 255.0) / (max_val - min_val)
    else:
        normalized_image = image
    return normalized_image.astype(np.uint8)


def normalize_color_image(image):
    """Normalizes a color image (BGR)."""
    normalized_image = np.zeros_like(image, dtype=np.float32)

    # Process each color channel independently
    for channel in range(3):
        channel_data = image[:, :, channel]
        normalized_channel = normalize_grayscale_image(channel_data)  
        normalized_image[:, :, channel] = normalized_channel
    return normalized_image.astype(np.uint8)


def process_image(image, equalize=True, normalize=True):
    """Processes an image (grayscale or color) with optional equalization and normalization."""
    # Handle both grayscale and color images
    if len(image.shape) == 2:  # Grayscale
        processed_image = image.copy() 
        if equalize:
            processed_image = equalize_grayscale_image(processed_image)
        if normalize:
            processed_image = normalize_grayscale_image(processed_image)
        return processed_image
    elif len(image.shape) == 3:  # Color
        processed_image = image.copy()
        if equalize:
            processed_image = equalize_color_image(processed_image)
        if normalize:
            processed_image = normalize_color_image(processed_image)
        return processed_image


def display_all_versions(image_path):
    """
    Displays original, normalized, equalized, and combined versions of an image.
    Args:
        image_path: Path to the input image
    """
    # Load image
    color_image = cv2.imread(image_path)
    
    # Process images
    normalized_image = process_image(color_image, equalize=False, normalize=True)
    equalized_image = process_image(color_image, equalize=True, normalize=False)
    equalized_and_normalized = process_image(color_image, equalize=True, normalize=True)
    
    # Display results
    plt.figure(figsize=(20, 5))
    
    # Create subplots for each version
    images = {
        'Original Image': color_image,
        'Normalized Image': normalized_image,
        'Equalized Image': equalized_image,
        'Equalized & Normalized': equalized_and_normalized
    }
    
    for idx, (title, img) in enumerate(images.items(), 1):
        plt.subplot(1, 4, idx)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage

# # Grayscale Image
# image_path_gray = r"C:\Users\Compu House\Desktop\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\pyQt\resources\low contrast grayscale.jfif"
# display_all_versions(image_path_gray)

# # Colored Image
# image_path_color = r"C:\Users\Compu House\Desktop\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\pyQt\resources\low contrast colored .jfif"
# display_all_versions(image_path_color)
