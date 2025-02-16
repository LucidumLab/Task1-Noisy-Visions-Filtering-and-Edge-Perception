import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

def threshold(image, thresh, maxval, type):
    if type != cv2.THRESH_BINARY:
        raise ValueError("Only cv2.THRESH_BINARY is currently implemented.")

    thresholded_image = np.zeros_like(image, dtype=np.uint8)  

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] > thresh:
                thresholded_image[y, x] = maxval
            else:
                thresholded_image[y, x] = 0  

    return thresh, thresholded_image

def gaussian_filter(image, sigma, kernel_size):
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create the Gaussian kernel
    x, y = np.meshgrid(np.arange(-(kernel_size // 2), kernel_size // 2 + 1),
                       np.arange(-(kernel_size // 2), kernel_size // 2 + 1))
    
	# The Gaussian function
    kernel = np.exp(-((x**2 + y**2) / (2 * sigma**2))) / (2 * np.pi * sigma**2)

    # Normalize the kernel 
    kernel = kernel / np.sum(kernel)

    # Apply the filter (convolution)
    filtered_image = np.zeros_like(image, dtype=float)  

    height, width = image.shape
    k_height, k_width = kernel.shape

    for i in range(height):
        for j in range(width):
            sum_val = 0
            for kx in range(k_width):
                for ky in range(k_height):
                    # Translate the kernel's coordinates to the original image's coordinates
                    x_idx = j - k_width // 2 + kx
                    y_idx = i - k_height // 2 + ky

					# Boundary check!
                    if 0 <= x_idx < width and 0 <= y_idx < height:  
                        sum_val += image[y_idx, x_idx] * kernel[ky, kx]

            filtered_image[i, j] = sum_val

    return filtered_image.astype(image.dtype) 

def convolve(image, kernel):
    # Get dimensions
    height, width = image.shape
    k_height, k_width = kernel.shape
    
    # Calculate padding needed
    pad_y = k_height // 2
    pad_x = k_width // 2
    
    # Initialize output array
    edges = np.zeros((height, width), dtype=np.float64)
    
    # Manual convolution
    for y in range(pad_y, height - pad_y):
        for x in range(pad_x, width - pad_x):
            # Extract region matching kernel size exactly
            region = image[y-pad_y:y+pad_y+(k_height%2), x-pad_x:x+pad_x+(k_width%2)]
            edges[y, x] = np.sum(region * kernel)
            
    return edges

def magnitude(x, y):
    return np.sqrt(x**2 + y**2)

# Sobel Edge Detection
def sobel_edge_detection(image):
    # Convert to grayscale as sobel operator works on single-channel images (not coloed images)
    gray_image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian smoothing (optional) to reduce noise
    blurred_image = gaussian_filter(gray_image, 0, 3)

    # # Sobel operators
    # Gx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    # Gy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) 

    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) 

    # Apply Sobel kernels
    Gx = convolve(blurred_image, kernel_x)
    Gy = convolve(blurred_image, kernel_y)

    # Compute gradient magnitude
    G = magnitude(Gx, Gy)

    # Normalize to range 0-255 for better visualization
    Gx = np.uint8(255 * np.abs(Gx) / np.max(Gx))
    Gy = np.uint8(255 * np.abs(Gy) / np.max(Gy))
    # G = np.uint8(255 * np.abs(G) / np.max(G))

    return G 

# Canny Edge Detection
def canny_edge_detection(img, low_threshold=None, high_threshold=None):
    # Conversion of image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Noise reduction step 
    img = gaussian_filter(img, 1.4, 5)
    
	# # Calculating the gradients 
    # gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
    # gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]) 

    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]]) 

    # Calculating the gradients 
    Gx = convolve(img, kernel_x)
    Gy = convolve(img, kernel_y)

	# # Conversion of Cartesian coordinates to polar coordinates
    # mag, ang = cv2.cartToPolar(Gx, Gy, angleInDegrees = True)

    # Manual Calculation of Magnitude and Angle 
    mag = magnitude(Gx, Gy)
    ang = np.arctan2(Gy, Gx) * 180 / np.pi
    
	# Setting the minimum and maximum thresholds for double thresholding
    max_mag = np.max(mag)
    if not low_threshold:
        low_threshold = max_mag * 0.1
    if not high_threshold:
        high_threshold = max_mag * 0.5
        
	# Get the dimentions of the input image
    height, width = img.shape
    
	# Loop though every pixel in the grayscale image
    for x_i in range(width):
        for y_i in range(height):
            grad_ang = ang[y_i, x_i]
            grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)
            
			# Select the neighbours of the target pixel according to the gradient direction
            
			# In the x direction
            if grad_ang <= 22.5:
                neighb_1_x, neighb_1_y = x_i - 1, y_i 
                neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
			# In the top right (diagonal-1) direction
            elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                neighb_1_x, neighb_1_y = x_i - 1, y_i - 1 
                neighb_2_x, neighb_2_y = x_i + 1, y_i + 1
                
			# In the y direction
            elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                neighb_1_x, neighb_1_y = x_i , y_i - 1 
                neighb_2_x, neighb_2_y = x_i , y_i + 1
                
			# In the top left (diagonal-2) direction
            elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                neighb_1_x, neighb_1_y = x_i - 1, y_i + 1 
                neighb_2_x, neighb_2_y = x_i + 1, y_i - 1
                
			# Now it restarts the cycle
            elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                neighb_1_x, neighb_1_y = x_i - 1, y_i  
                neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
			# Non-maximum suppression step
            if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                if mag[y_i, x_i] < mag[neighb_1_y, neighb_1_x]:
                    mag[y_i, x_i] = 0
                    continue
            if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                if mag[y_i, x_i] < mag[neighb_2_y, neighb_2_x]:
                    mag[y_i, x_i] = 0
                    

	# Hysteresis Thresholding (NOT IMPLEMENTED YET)
    weak_ids = np.zeros_like(img) 
    strong_ids = np.zeros_like(img)               
    ids = np.zeros_like(img) 
		
	# double thresholding step 
    for i_x in range(width): 
        for i_y in range(height): 
              
            grad_mag = mag[i_y, i_x] 
              
            if grad_mag < low_threshold: 
                mag[i_y, i_x]= 0
            elif high_threshold > grad_mag >= low_threshold: 
                ids[i_y, i_x]= 1
            else: 
                ids[i_y, i_x]= 2
                
    return mag

# Prewitt Edge Detection
def prewitt_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply horizontal Prewitt kernel
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    horizontal_edges = convolve(gray_image, kernel_x)
    
    # Apply vertical Prewitt kernel
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    vertical_edges = convolve(gray_image, kernel_y)

    # Ensure both arrays have the same data type
    horizontal_edges = np.float32(horizontal_edges)
    vertical_edges = np.float32(vertical_edges)
    
    # Compute gradient magnitude
    gradient_magnitude = magnitude(horizontal_edges, vertical_edges)
    
    # Optional: Apply thresholding to highlight edges
    thresh_value, edges = threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)
    
    return edges

# Roberts Edge Detection
def roberts_edge_detection(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roberts_cross_v = np.array( [[1, 0 ], 
                                [0,-1 ]]) 

    roberts_cross_h = np.array( [[ 0, 1 ], 
                                [ -1, 0 ]]) 
    
    # Apply the filter (convolution)
    vertical = convolve(gray_image, roberts_cross_v ) 
    horizontal = convolve(gray_image, roberts_cross_h ) 

    # Compute gradient magnitude
    gradient_mag = magnitude(horizontal, vertical)

    return gradient_mag

def display_edge_detection(image_path, edge_detection_func, title):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Apply edge detection
    edges = edge_detection_func(image)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Detected Edges
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'{title} Edge Detection')
    plt.axis('off')
    
    plt.show()

# # Usage examples:
display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', sobel_edge_detection, 'Sobel')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', canny_edge_detection, 'Canny')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', prewitt_edge_detection, 'Prewitt')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', roberts_edge_detection, 'Roberts')
