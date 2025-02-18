import numpy as np
import matplotlib.pyplot as plt
from utils import convert_to_grayscale 

def compute_histogram(image):
    """_summary_

    Args:
        image (ndarray): 2d array of grayscale image or one channel image

    Returns:
        hist: histogram list of pixels intensities
    """
    shape = image.shape
    size = shape[0]*shape[1]
    hist = np.zeros(size, dtype=int)
    for pixel in image.flatten():
        hist[pixel] += 1

    return hist

def cumsum(arr):
    """
    Computes the cumulative sum of an array.

    :param arr: List of numbers.
    :return: List containing the cumulative sum.
    """
    cumulative_sum = [0] * len(arr)  # Initialize output array with zeros
    
    cumulative_sum[0] = arr[0]  # First element remains the same

    for i in range(1, len(arr)):  # Start from the second element
        cumulative_sum[i] = cumulative_sum[i - 1] + arr[i]  # Add previous sum

    return cumulative_sum

def compute_cdf(hist):
    cdf = cumsum(hist)
    return cdf

def normalize(cdf):
    cdf_min = cdf[cdf > 0].min()  # Get the first nonzero value
    cdf_normalized = ((cdf - cdf_min) / (cdf.max() - cdf_min)) * 255
    return np.round(cdf_normalized).astype(np.uint8)  # Convert to uint8

def apply_histogram_equalization(image, cdf_normalized):
    """
    Applies histogram equalization using the normalized CDF.

    :param image: Input grayscale image.
    :param cdf_normalized: Normalized CDF for pixel mapping.
    :return: Equalized image.
    """
    equalized_image = cdf_normalized[image]  # Map original pixels to new values
    return equalized_image


    
def compute_distribution_curve(data):
    """
    Computes the normal distribution curve manually without SciPy.

    :param data: Flattened grayscale or color channel image.
    :return: Tuple (x_values, y_values).
    """
    mean, std = np.mean(data), np.std(data)
    x_values = np.linspace(data.min(), data.max(), 100)
    
    # Manually compute normal distribution (Gaussian function)
    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
    
    return x_values, y_values

def draw_histo_and_distribution_curve(data, bins):
    """
    Plots a histogram with an overlaid normal distribution curve.
    """
    plt.figure(figsize=(8, 6))
    
    # Plot histogram
    plt.hist(data, bins=bins, density=True, edgecolor='black', alpha=0.7, label='Histogram')
    
    # Compute normal distribution curve
    mean, std = np.mean(data), np.std(data)
    x = np.linspace(min(data), max(data), 100)
    y = norm.pdf(x, mean, std)
    
    # Overlay normal distribution curve
    plt.plot(x, y, color='red', label='Normal Distribution')
    
    # Add labels and title
    plt.title('Histogram with Distribution Curve')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    
    # Show plot
    plt.show()
    
    return x, y

def draw_image_histogram_and_distribution(image, bins=256):
    """
    takes an image and plots its histogram with an overlaid normal distribution curve.
    
    Returns:
    x (numpy array): X values for the normal distribution curve.
    y (numpy array): Y values for the normal distribution curve.
    """
    flattened_data = image.flatten()
    return draw_histo_and_distribution_curve(flattened_data, bins=bins)


