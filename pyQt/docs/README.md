## Noise Addition Functions

### `add_uniform_noise(img: np.ndarray, intensity: int = 50) -> np.ndarray`
Adds uniform noise to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `intensity` (int): Intensity of the noise.
- **Returns:** np.ndarray: Image with uniform noise added.

### `add_gaussian_noise(img: np.ndarray, mean: float = 0, std: float = 25) -> np.ndarray`
Adds Gaussian noise to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `mean` (float): Mean of the Gaussian noise.
  - `std` (float): Standard deviation of the Gaussian noise.
- **Returns:** np.ndarray: Image with Gaussian noise added.

### `add_salt_pepper_noise(img: np.ndarray, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray`
Adds salt and pepper noise to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `salt_prob` (float): Probability of salt noise.
  - `pepper_prob` (float): Probability of pepper noise.
- **Returns:** np.ndarray: Image with salt and pepper noise added.

## Filtering Functions

### `apply_average_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray`
Applies an average filter to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `kernel_size` (int): Size of the kernel.
- **Returns:** np.ndarray: Image with average filter applied.

### `apply_gaussian_filter(img: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray`
Applies a Gaussian filter to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `kernel_size` (int): Size of the kernel.
  - `sigma` (float): Standard deviation of the Gaussian kernel.
- **Returns:** np.ndarray: Image with Gaussian filter applied.

### `apply_median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray`
Applies a median filter to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
  - `kernel_size` (int): Size of the kernel.
- **Returns:** np.ndarray: Image with median filter applied.

### `apply_filters(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
Applies all three filters (average, Gaussian, and median) to the image.
- **Parameters:**
  - `img` (np.ndarray): Input image.
- **Returns:** tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing images with average, Gaussian, and median filters applied.

## Visualization Function

### `show_images(images: list[np.ndarray], titles: list[str]) -> None`
Displays a list of images with their corresponding titles using Matplotlib.
- **Parameters:**
  - `images` (list[np.ndarray]): List of images to display.
  - `titles` (list[str]): List of titles for the images.
- **Returns:** None. Displays the images in a grid format.