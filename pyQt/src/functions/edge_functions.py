import numpy as np 
from utils import convolve, magnitude, convert_to_grayscale
from functions.thresholding_functions import globalthresholding
from functions.noise_functions import apply_gaussian_filter
import cv2 
import matplotlib.pyplot as plt 



def smooth_image(image, kernel_size=5, sigma=1.4):
    """Smooths the image using a Gaussian filter."""
    return apply_gaussian_filter(image, kernel_size, sigma)

def compute_sobel_gradients(image):
    """
    Computes the Sobel gradients of an image.
    
    :param image: Grayscale image.
    :return: Tuple (Gx, Gy) representing horizontal and vertical gradients.
    """
    # Define Sobel kernels
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    
    Gx = convolve(image, kernel_x)
    Gy = convolve(image, kernel_y)
    return Gx, Gy


def non_maximum_suppression(G, theta):
    height, width = G.shape
    nms = np.zeros_like(G, dtype=np.uint8)
    # WHY DOES WE DO IT AGAIN IN THIS FUNCTION?? (WE HAVE DONE IT ONCE IN THE FUNCTION OF CANNY)
    theta = theta % 180

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            angle = theta[y, x]
            if (0 <= angle < 22.5) or (157.5 <= angle < 180):
                p1, p2 = G[y, x-1], G[y, x+1]
            elif 22.5 <= angle < 67.5:
                p1, p2 = G[y-1, x+1], G[y+1, x-1]
            elif 67.5 <= angle < 112.5:
                p1, p2 = G[y-1, x], G[y+1, x]
            elif 112.5 <= angle < 157.5:
                p1, p2 = G[y-1, x-1], G[y+1, x+1]

            if G[y, x] >= p1 and G[y, x] >= p2:
                nms[y, x] = G[y, x]

    return nms

# min_edgeval SHOULD BE 128 FOR EXAMPLE
def apply_double_thresholding(image, low_threshold, high_threshold, max_edge_val=255, min_edge_val=0):
    strong_edges = np.zeros_like(image)
    weak_edges = np.zeros_like(image)

    strong_edges[image >= high_threshold] = max_edge_val
    weak_edges[(image >= low_threshold) & (image < high_threshold)] = min_edge_val
    return strong_edges, weak_edges

def apply_hysteresis(strong_edges, weak_edges, max_edge_val=255):
    height, width = strong_edges.shape
    final_edges = np.copy(strong_edges)
    stack = list(zip(*np.where(strong_edges == max_edge_val)))

    while stack:
        y, x = stack.pop()
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                # THIS CONDITION IS INCORRECT, IT WILL DETECT PIXELS THAT ARE NOT WEAK EDGES TO BE STRONG EDGES (weak_edges[ny, nx] == 128)
                if 0 <= ny < height and 0 <= nx < width and weak_edges[ny, nx] != 0:
                    final_edges[ny, nx] = max_edge_val
                    weak_edges[ny, nx] = 0
                    stack.append((ny, nx))

    return final_edges

def sobel_edge_detection(image, kernel_size = 3, sigma = 1.0):
    """
    Detects edges using the Sobel operator.
    
    :param image: Input BGR or grayscale image.
    :return: Sobel edge map (normalized to 0-255).
    """
    gray = convert_to_grayscale(image)
    # Optional Gaussian smoothing before gradient calculation:
    blurred = smooth_image(gray, kernel_size, sigma)
    Gx, Gy = compute_sobel_gradients(blurred)
    G = magnitude(Gx, Gy)
    return np.uint8(255 * (G / np.max(G)))

def prewitt_edge_detection(image, threshold, value):
    """
    Detects edges using the Prewitt operator.
    
    :param image: Input BGR or grayscale image.
    :return: Prewitt edge map.
    """
    gray = convert_to_grayscale(image)
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [ 0,  0,  0],
                         [ 1,  1,  1]])
    Gx = convolve(gray, kernel_x)
    Gy = convolve(gray, kernel_y)
    
    Gx = np.float32(Gx)
    Gy = np.float32(Gy)
    
    G = np.uint8(magnitude(Gx, Gy))
    G = globalthresholding(G, threshold, value)
    return G
    
    
def roberts_edge_detection(image):
    """
    Detects edges using the Roberts operator.
    
    :param image: Input BGR or grayscale image.
    :return: Roberts edge map.
    """
    gray = convert_to_grayscale(image)
    kernel_x = np.array([[1, 0],
                         [0, -1]])
    kernel_y = np.array([[0, 1],
                         [-1, 0]])
    Gx = convolve(gray, kernel_x)
    Gy = convolve(gray, kernel_y)
    return np.int8(magnitude(Gx, Gy))

def canny_edge_detection(image, low_threshold=10, high_threshold=30, max_edge_val=255, min_edge_val=0):
    gray = convert_to_grayscale(image)
    smooth = apply_gaussian_filter(gray)
    Gx, Gy = compute_sobel_gradients(smooth)
    G = magnitude(Gx, Gy)
    G_normalized = np.uint8(G / G.max() * 255)

    theta = np.rad2deg(np.arctan2(Gy, Gx)) % 180
    nms = non_maximum_suppression(G_normalized, theta)

    max_val = np.max(nms)
    low_threshold = low_threshold if low_threshold is not None else max_val * 0.1
    high_threshold = high_threshold if high_threshold is not None else max_val * 0.5

    strong_edges, weak_edges = apply_double_thresholding(nms, low_threshold, high_threshold, max_edge_val, min_edge_val)
    final_edges = apply_hysteresis(strong_edges, weak_edges)

    return  final_edges



# # Canny Edge Detection
# def canny_edge_detection(img, low_threshold=None, high_threshold=None):
#     # Conversion of image to grayscale
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 	# Noise reduction step 
#     img = apply_gaussian_filter(img, 5, 1.4)
    
# 	# # Calculating the gradients 
#     # gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3) 
#     # gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

#     # Define Sobel kernels
#     kernel_x = np.array([[-1, 0, 1],
#                         [-2, 0, 2],
#                         [-1, 0, 1]]) 

#     kernel_y = np.array([[-1, -2, -1],
#                         [0, 0, 0],
#                         [1, 2, 1]]) 

#     # Calculating the gradients 
#     Gx = convolve(img, kernel_x)
#     Gy = convolve(img, kernel_y)

# 	# # Conversion of Cartesian coordinates to polar coordinates
#     # mag, ang = cv2.cartToPolar(Gx, Gy, angleInDegrees = True)

#     # Manual Calculation of Magnitude and Angle 
#     mag = magnitude(Gx, Gy)
#     ang = np.arctan2(Gy, Gx) * 180 / np.pi
    
# 	# Setting the minimum and maximum thresholds for double thresholding
#     max_mag = np.max(mag)
#     if not low_threshold:
#         low_threshold = max_mag * 0.1
#     if not high_threshold:
#         high_threshold = max_mag * 0.5
        
# 	# Get the dimentions of the input image
#     height, width = img.shape
    
# 	# Loop though every pixel in the grayscale image
#     for x_i in range(width):
#         for y_i in range(height):
#             grad_ang = ang[y_i, x_i]
#             grad_ang = abs(grad_ang-180) if abs(grad_ang) > 180 else abs(grad_ang)
            
# 			# Select the neighbours of the target pixel according to the gradient direction
            
# 			# In the x direction
#             if grad_ang <= 22.5:
#                 neighb_1_x, neighb_1_y = x_i - 1, y_i 
#                 neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
# 			# In the top right (diagonal-1) direction
#             elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
#                 neighb_1_x, neighb_1_y = x_i - 1, y_i - 1 
#                 neighb_2_x, neighb_2_y = x_i + 1, y_i + 1
                
# 			# In the y direction
#             elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
#                 neighb_1_x, neighb_1_y = x_i , y_i - 1 
#                 neighb_2_x, neighb_2_y = x_i , y_i + 1
                
# 			# In the top left (diagonal-2) direction
#             elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
#                 neighb_1_x, neighb_1_y = x_i - 1, y_i + 1 
#                 neighb_2_x, neighb_2_y = x_i + 1, y_i - 1
                
# 			# Now it restarts the cycle
#             elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
#                 neighb_1_x, neighb_1_y = x_i - 1, y_i  
#                 neighb_2_x, neighb_2_y = x_i + 1, y_i 
                
# 			# Non-maximum suppression step
#             if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
#                 if mag[y_i, x_i] < mag[neighb_1_y, neighb_1_x]:
#                     mag[y_i, x_i] = 0
#                     continue
#             if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
#                 if mag[y_i, x_i] < mag[neighb_2_y, neighb_2_x]:
#                     mag[y_i, x_i] = 0
                    

# 	# Hysteresis Thresholding (NOT IMPLEMENTED YET)
#     weak_ids = np.zeros_like(img) 
#     strong_ids = np.zeros_like(img)               
#     ids = np.zeros_like(img) 
		
# 	# double thresholding step 
#     for i_x in range(width): 
#         for i_y in range(height): 
              
#             grad_mag = mag[i_y, i_x] 
              
#             if grad_mag < low_threshold: 
#                 mag[i_y, i_x]= 0
#             elif high_threshold > grad_mag >= low_threshold: 
#                 ids[i_y, i_x]= 1
#             else: 
#                 ids[i_y, i_x]= 2
                
#     return mag

# Prewitt Edge Detection
# def prewitt_edge_detection(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply horizontal Prewitt kernel
#     kernel_x = np.array([[-1, 0, 1],
#                          [-1, 0, 1],
#                          [-1, 0, 1]])
#     horizontal_edges = convolve(gray_image, kernel_x)
    
#     # Apply vertical Prewitt kernel
#     kernel_y = np.array([[-1, -1, -1],
#                          [0, 0, 0],
#                          [1, 1, 1]])
#     vertical_edges = convolve(gray_image, kernel_y)

#     # Ensure both arrays have the same data type

    
#     # Compute gradient magnitude
#     gradient_magnitude = magnitude(horizontal_edges, vertical_edges)
    
#     # Optional: Apply thresholding to highlight edges
#     thresh_value, edges = globalthresholding(gradient_magnitude, 50, 255)
    
#     return edges

# # Roberts Edge Detection
# def roberts_edge_detection(image):
#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     roberts_cross_v = np.array( [[1, 0 ], 
#                                 [0,-1 ]]) 

#     roberts_cross_h = np.array( [[ 0, 1 ], 
#                                 [ -1, 0 ]]) 
    
#     # Apply the filter (convolution)
#     vertical = convolve(gray_image, roberts_cross_v ) 
#     horizontal = convolve(gray_image, roberts_cross_h ) 

#     # Compute gradient magnitude
#     gradient_mag = magnitude(horizontal, vertical)

#     return gradient_mag

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
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', sobel_edge_detection, 'Sobel')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', canny_edge_detection, 'Canny')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', prewitt_edge_detection, 'Prewitt')
# display_edge_detection(r'E:\Rawan\Projects\Projects\ThirdYearBiomedical\Second Term\Computer Vision\Task1FilteringAndEdgeDetection\Task1-Noisy-Visions-Filtering-and-Edge-Perception\flower.png', roberts_edge_detection, 'Roberts')
