import numpy as np
import matplotlib.pyplot as plt
from utils import convert_to_grayscale
from functions.histogram_functions import compute_histogram, compute_cdf, normalize, apply_histogram_equalization

class HistogramProcessor:
    """Handles histogram computation, CDF calculation, normalization, and equalization for grayscale and color images."""

    def __init__(self):
        self.image = None
        self.histograms = None
        self.cdfs = None
        self.cdf_normalized = None
        self.equalized_image = None
        self.is_color = False

    def set_image(self, image):
        """
        Sets the image and computes histograms for grayscale or each RGB channel.

        :param image: Input grayscale or color image.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid image input. Expected a NumPy array.")
        
        self.image = image
        self.is_color = len(image.shape) == 3  # Check if the image has 3 channels

        if self.is_color:
            self.histograms = {color: compute_histogram(image[:, :, idx]) 
                               for idx, color in enumerate(["Red", "Green", "Blue"])}
            self.cdfs = {color: compute_cdf(hist) for color, hist in self.histograms.items()}
            self.cdf_normalized = {color: normalize(cdf) for color, cdf in self.cdfs.items()}
        else:
            self.histograms = compute_histogram(image)
            self.cdfs = compute_cdf(self.histograms)
            self.cdf_normalized = normalize(self.cdfs)

    def transform_to_grayscale(self):
        """
        Converts a color image to grayscale using a weighted sum method.
        
        :return: Grayscale image.
        """
        if not self.is_color:
            raise ValueError("Image is already grayscale.")
        
        grayscale_image = convert_to_grayscale(self.image)
        return grayscale_image

    def apply_histogram_equalization(self):
        """
        Applies histogram equalization using the normalized CDF.

        :return: Equalized grayscale or color image.
        """
        if self.image is None:
            raise ValueError("Image not set. Please call set_image() first.")

        if self.is_color:
            # Apply equalization separately for R, G, B channels
            equalized_channels = [self.cdf_normalized[color][self.image[:, :, idx]] 
                                  for idx, color in enumerate(["Red", "Green", "Blue"])]
            self.equalized_image = np.stack(equalized_channels, axis=-1)
        else:
            self.equalized_image = apply_histogram_equalization(self.image, self.cdf_normalized)
        
        return self.equalized_image

    def plot_histogram_with_distribution(self, channel="Grayscale"):
        """
        Plots histogram and its distribution function (CDF).

        :param channel: 'Grayscale' or 'Red', 'Green', 'Blue'.
        """
        if self.histograms is None:
            raise ValueError("No histogram available. Set an image first.")
        
        if self.is_color and channel not in ["Red", "Green", "Blue"]:
            raise ValueError("Invalid channel. Choose 'Red', 'Green', or 'Blue'.")

        # Select histogram and CDF based on the channel
        if self.is_color:
            hist = self.histograms[channel]
            cdf = self.cdfs[channel]
            color = channel.lower()
        else:
            hist = self.histograms
            cdf = self.cdfs
            color = "black"

        # Plot histogram
        fig, ax1 = plt.subplots()
        ax1.bar(range(256), hist, color=color, alpha=0.6, label=f'{channel} Histogram')
        ax1.set_xlabel("Pixel Intensity")
        ax1.set_ylabel("Frequency", color=color)
        
        # Plot CDF on the same graph
        ax2 = ax1.twinx()
        ax2.plot(range(256), cdf / cdf.max(), color="blue", label="CDF")  # Normalize CDF
        ax2.set_ylabel("Cumulative Distribution", color="blue")

        plt.title(f"{channel} Histogram & Distribution")
        plt.legend()
        plt.show()

    def plot_all_histograms(self):
        """
        Plots histograms and CDFs for all channels (Grayscale or RGB).
        """
        if self.is_color:
            for channel in ["Red", "Green", "Blue"]:
                self.plot_histogram_with_distribution(channel)
        else:
            self.plot_histogram_with_distribution()
