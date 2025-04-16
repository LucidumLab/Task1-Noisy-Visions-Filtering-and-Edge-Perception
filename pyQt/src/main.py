import sys
import cv2
import numpy as np
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

from processor_factory import ProcessorFactory
from classes.histogram_processor import HistogramVisualizationWidget
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame, QTabWidget, QSpacerItem, QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QCheckBox,
    QStackedWidget, QGridLayout
)

from processor_factory import ProcessorFactory
# from functions.hough_transform_functions import detect_lines,detect_circles
# from functions.active_contour_functions import initialize_snake, external_energy



from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,QComboBox, QSpinBox,QDoubleSpinBox, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
import cv2

from tabs.ActiveContourTab import ActiveContourTab
from tabs.HoughTransformTab import HoughTransformTab
from tabs.NoiseFilterTab import NoiseFilterTab
from tabs.EdgeDetectionTab import EdgeDetectionTab
from tabs.ThresholdingTab import ThresholdingTab
from tabs.FrequencyFilterTab import FrequencyFilterTab
from tabs.HybridImageTab import HybridImageTab

class Feature_DetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        # Edge Detector Selection
        edge_detector_group = QFrame()
        edge_detector_group.setObjectName("edge_detector_group")
        edge_detector_layout = QVBoxLayout(edge_detector_group)

        # === Edge Type Frame ===
        self.edgeTypeFrame = QFrame()
        edge_type_layout = QHBoxLayout(self.edgeTypeFrame)
        edge_type_layout.addWidget(QLabel("Edge Detector"))
        self.edgeType = QComboBox()
        self.edgeType.addItems(["canny", "sobel", "prewitt", "roberts"])
        self.edgeType.currentTextChanged.connect(self.update_edge_params_visibility)
        edge_type_layout.addWidget(self.edgeType)
        edge_detector_layout.addWidget(self.edgeTypeFrame)

        # === Kernel Size Frame ===
        self.kernelSizeFrame = QFrame()
        kernel_size_layout = QHBoxLayout(self.kernelSizeFrame)
        kernel_size_layout.addWidget(QLabel("Kernel Size"))
        self.kernelSize = QSpinBox()
        self.kernelSize.setRange(1, 15)
        self.kernelSize.setValue(3)
        kernel_size_layout.addWidget(self.kernelSize)
        edge_detector_layout.addWidget(self.kernelSizeFrame)

        # === Low Threshold Frame ===
        self.lowThresholdFrame = QFrame()
        low_threshold_layout = QHBoxLayout(self.lowThresholdFrame)
        low_threshold_layout.addWidget(QLabel("Low Threshold"))
        self.lowThreshold = QSpinBox()
        self.lowThreshold.setRange(0, 255)
        self.lowThreshold.setValue(50)
        low_threshold_layout.addWidget(self.lowThreshold)
        edge_detector_layout.addWidget(self.lowThresholdFrame)

        # === High Threshold Frame ===
        self.highThresholdFrame = QFrame()
        high_threshold_layout = QHBoxLayout(self.highThresholdFrame)
        high_threshold_layout.addWidget(QLabel("High Threshold"))
        self.highThreshold = QSpinBox()
        self.highThreshold.setRange(0, 255)
        self.highThreshold.setValue(150)
        high_threshold_layout.addWidget(self.highThreshold)
        edge_detector_layout.addWidget(self.highThresholdFrame)

        main_layout.addWidget(edge_detector_group)

        # Feature Detector Selection
        feature_detector_group = QFrame()
        feature_detector_group.setObjectName("feature_detector_group")
        feature_detector_layout = QVBoxLayout(feature_detector_group)

        # === Feature Type Frame ===
        self.featureTypeFrame = QFrame()
        feature_type_layout = QHBoxLayout(self.featureTypeFrame)
        feature_type_layout.addWidget(QLabel("Feature Detector"))
        self.featureType = QComboBox()
        self.featureType.addItems(["MOPs", "SIFT", "Harris"])
        self.featureType.currentTextChanged.connect(self.update_feature_params_visibility)
        feature_type_layout.addWidget(self.featureType)
        feature_detector_layout.addWidget(self.featureTypeFrame)

        # === Harris Controls Frame ===
        self.harrisFrame = QFrame()
        harris_layout = QHBoxLayout(self.harrisFrame)
        self.harrisKLabel = QLabel("Harris K")
        self.harrisK = QDoubleSpinBox()
        self.harrisK.setRange(0.01, 0.1)
        self.harrisK.setSingleStep(0.01)
        self.harrisK.setValue(0.04)
        harris_layout.addWidget(self.harrisKLabel)
        harris_layout.addWidget(self.harrisK)
        feature_detector_layout.addWidget(self.harrisFrame)

        # === SIFT Controls Frame ===
        self.siftFrame = QFrame()
        sift_layout = QHBoxLayout(self.siftFrame)
        self.siftFeaturesLabel = QLabel("SIFT Features")
        self.siftFeatures = QSpinBox()
        self.siftFeatures.setRange(1, 5000)
        self.siftFeatures.setValue(500)
        sift_layout.addWidget(self.siftFeaturesLabel)
        sift_layout.addWidget(self.siftFeatures)
        feature_detector_layout.addWidget(self.siftFrame)

        main_layout.addWidget(feature_detector_group)

        # Update visibility initially
        self.update_edge_params_visibility()
        self.update_feature_params_visibility()

    def update_edge_params_visibility(self):
        edge_type = self.edgeType.currentText()
        is_canny = edge_type == "canny"

        # Show/Hide Low and High Threshold frames together
        self.lowThresholdFrame.setVisible(is_canny)
        self.highThresholdFrame.setVisible(is_canny)

    def update_feature_params_visibility(self):
        feature_type = self.featureType.currentText()

        self.harrisFrame.setVisible(feature_type == "Harris")
        self.siftFrame.setVisible(feature_type == "SIFT")

class Feature_Detection_Frame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(grid_layout)

        
        self.original_image = QLabel("Original Image")
        self.original_image.setObjectName("original_label")
        self.original_image.setFixedSize(600, 425)
        self.original_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.original_image, 0, 0)

        
        self.template_image = QLabel("Template Image")
        self.template_image.setObjectName("template_label")
        self.template_image.setFixedSize(600, 425)
        self.template_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.template_image, 0, 1)

        
        self.roi_image = QLabel("ROI Image")
        self.roi_image.setObjectName("roi_label")
        self.roi_image.setFixedSize(600, 425)
        self.roi_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.roi_image, 1, 0)

        
        self.confusion_image = QLabel("Confusion Matrix")
        self.confusion_image.setObjectName("confusion_label")
        self.confusion_image.setFixedSize(600, 425)
        self.confusion_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.confusion_image, 1, 1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(50, 50, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        self.init_ui(self.main_layout)

        # Single data structure to store all parameters
        self.params = {
            "noise_filter": {},
            "filtering": {},
            "edge_detection": {},
            "thresholding": {},
            "frequency_filter": {},
            "hybrid_image": {},
            "shape_detection":{},
            "active_contour":{}
            
        }
        
        self.connect_signals()
        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None
        self.processors = {key: ProcessorFactory.create_processor(key) for key in ['noise', 'edge_detector', 'thresholding', 'frequency', 'histogram', 'image', 'active_contour']}

    def run_active_contour(self):
        """
        Runs the Active Contour (Snake) algorithm on the loaded image.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return
        contour_params = self.params["active_contour"]

        # Initialize the snake
        snake, history = self.processors["active_contour"].detect_contour(**contour_params)

        self.processors["active_contour"].visualize_contour()

        # Clear the layout before adding a new widget
        # while self.image_display_layout.count():
        #     item = self.image_display_layout.takeAt(0)
        #     if item.widget():
        #         item.widget().deleteLater()

        # Create and add the VisualizationPanel
        # self.active_contour_viewer = VisualizationPanel(self, self.processors["active_contour"])
        # self.active_contour_viewer.set_history(history)

        # self.image_display_layout.addWidget(self.active_contour_viewer)  # Add widget to layout
        # plt.imshow(self.image, cmap='gray')
        # plt.plot(snake[:, 0], snake[:, 1], 'r-', label="Snake Contour")
        # plt.title("Active Contour Result")
        # plt.legend()
        # plt.show()
        # Ensure the widget is displayed
        # self.active_contour_viewer.show()


    def init_ui(self, main_layout):
        left_frame = QFrame()
        left_frame.setObjectName("left_frame")
        left_layout = QVBoxLayout(left_frame)
        
        tab_widget = QTabWidget()
        tab_widget.setObjectName("tab_widget")

        # Noise and Filter Tab
        self.noise_filter_tab = NoiseFilterTab(self)
        tab_widget.addTab(self.noise_filter_tab, "Noise & Filter")

        # Edge Detection Tab
        self.edge_detection_tab = EdgeDetectionTab(self)
        tab_widget.addTab(self.edge_detection_tab, "Edge Detection")

        # Thresholding Tab
        self.thresholding_tab = ThresholdingTab(self)
        tab_widget.addTab(self.thresholding_tab, "Thresholding")

        # Frequency Filter Tab
        self.frequency_filter_tab = FrequencyFilterTab(self)
        tab_widget.addTab(self.frequency_filter_tab, "Frequency Filter")

        # Hybrid Image Tab
        self.hybrid_image_tab = HybridImageTab(self)
        tab_widget.addTab(self.hybrid_image_tab, "Hybrid Image")

        # Hough Transform Tab
        self.hough_transform_tab = HoughTransformTab(self)
        tab_widget.addTab(self.hough_transform_tab, "Hough Transform")

        self.active_contour_tab = ActiveContourTab(self)
        tab_widget.addTab(self.active_contour_tab, "Active Contour")

        # Feature Detection Tab
        self.feature_detection_tab = Feature_DetectionTab(self)
        tab_widget.addTab(self.feature_detection_tab, "Feature Detection")

        tab_widget.currentChanged.connect(self.on_tab_changed)

        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)
        
        #? Right Frame with Control Buttons and Image Display
        self.right_frame = QFrame()
        self.right_frame.setObjectName("right_frame")
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)

        
        # Control Buttons Frame
        control_frame = QFrame()
        control_frame.setMaximumHeight(100)
        control_layout = QHBoxLayout(control_frame)
        self.right_layout.addWidget(control_frame)

        # Control Buttons Frame
        control_buttons_frame = QFrame()
        control_buttons_layout = QHBoxLayout(control_buttons_frame)

        self.btn_histogram = QPushButton()
        self.btn_histogram.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/diagram-bar-3.png')))
        self.btn_histogram.setIconSize(QSize(32, 32))
        self.btn_histogram.clicked.connect(self.show_histogram)
        control_buttons_layout.addWidget(self.btn_histogram)

        self.btn_equalize = QPushButton()
        self.btn_equalize.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/equalizer-solid.png')))
        self.btn_equalize.setIconSize(QSize(32, 32))
        self.btn_equalize.clicked.connect(self.equalize)
        control_buttons_layout.addWidget(self.btn_equalize)

        self.btn_normalize = QPushButton()
        self.btn_normalize.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/gaussain-curve.png')))
        self.btn_normalize.setIconSize(QSize(32, 32))
        self.btn_normalize.clicked.connect(self.normalize)
        control_buttons_layout.addWidget(self.btn_normalize)

        control_layout.addWidget(control_buttons_frame)

        control_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        # Image Control Buttons Frame
        image_control_buttons_frame = QFrame()
        image_control_buttons_layout = QHBoxLayout(image_control_buttons_frame)

        self.btn_confirm = QPushButton()
        self.btn_confirm.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/confirm.png')))
        self.btn_confirm.setIconSize(QSize(28, 28))
        self.btn_confirm.clicked.connect(self.confirm_edit)
        image_control_buttons_layout.addWidget(self.btn_confirm)

        self.btn_discard = QPushButton()
        self.btn_discard.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/discard.png')))
        self.btn_discard.setIconSize(QSize(28, 28))
        self.btn_discard.clicked.connect(self.discard_edit)
        image_control_buttons_layout.addWidget(self.btn_discard)

        self.btn_reset = QPushButton()
        self.btn_reset.setIcon(QIcon(os.path.join(os.path.dirname(__file__), '../resources/reset.png')))
        self.btn_reset.setIconSize(QSize(28, 28))
        self.btn_reset.clicked.connect(self.reset_image)
        image_control_buttons_layout.addWidget(self.btn_reset)
        control_layout.addWidget(image_control_buttons_frame)

        # Add the control frame to the right layout
        self.content_stack = QStackedWidget()
        self.right_layout.addWidget(self.content_stack)

        # Image Display Frame
        self.image_display_frame = QFrame()  # Use self.image_display_frame instead of image_display_frame
        self.image_display_frame.setFixedSize(1390, 880)
        self.image_display_layout = QVBoxLayout(self.image_display_frame)

        self.lbl_image = QLabel("No Image Loaded")
        self.lbl_image.setObjectName("lbl_image")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.image_display_layout.addWidget(self.lbl_image)
        self.lbl_image.mouseDoubleClickEvent = self.on_image_label_double_click

        # Add the image display frame to the content stack
        self.content_stack.addWidget(self.image_display_frame)

        # Feature Detection Frame
        self.feature_detection_frame = Feature_Detection_Frame()
        self.content_stack.addWidget(self.feature_detection_frame)

        # Add the right frame to the main layout
        main_layout.addWidget(self.right_frame)

    def on_tab_changed(self, index):
        """
        Switch the content of the right frame based on the selected tab.
        """
        if index == 7:  # Assuming "Feature Detection" is the 8th tab (index 7)
            self.content_stack.setCurrentWidget(self.feature_detection_frame)
        else:
            self.content_stack.setCurrentWidget(self.image_display_frame)
    
    def connect_signals(self):
        # Active contour tab
        
        active_contour_ui = {
            "center" : (self.active_contour_tab.centerX, self.active_contour_tab.centerY),
            "radius" : self.active_contour_tab.radius,
            "alpha" : self.active_contour_tab.alpha,
            "beta" : self.active_contour_tab.beta,
            "gamma": self.active_contour_tab.gamma,
            "iterations" : self.active_contour_tab.iterations,
            "points": self.active_contour_tab.points,
            "w_edge" : self.active_contour_tab.w_edge,
            "convergence" : self.active_contour_tab.convergence
            
         } 
        
        for widget in active_contour_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("active_contour", active_contour_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("active_contour", active_contour_ui))      

        # Hough Transform and Shape Detection Tab
        shape_detection_ui = {
            'num_rho': self.hough_transform_tab.numRho,
            'num_theta': self.hough_transform_tab.numTheta,
            'hough_threshold_ratio': self.hough_transform_tab.houghThresholdRatio,
            'r_min': self.hough_transform_tab.rMin,
            'r_max': self.hough_transform_tab.rMax,
            'num_thetas': self.hough_transform_tab.numThetas,
            'min_d': self.hough_transform_tab.aMin,
            'max_d': self.hough_transform_tab.aMax,
            'step_size': self.hough_transform_tab.thetaStep,
            'threshold_factor': self.hough_transform_tab.ellipseThresholdRatio,
            # Shared Canny Detector Parameters
            'canny_low_threshold': self.hough_transform_tab.cannyLowThreshold,
            'canny_high_threshold': self.hough_transform_tab.cannyHighThreshold,
            'canny_blur_ksize': self.hough_transform_tab.cannyBlurKSize
        }

        for widget in shape_detection_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("shape_detection", shape_detection_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("shape_detection", shape_detection_ui))      

        
        # Noise & Filter Tab
        noise_filter_ui = {
            "noise_type": self.noise_filter_tab.noiseType,
            "noise_intensity": self.noise_filter_tab.noiseIntensity,
            "gaussian_mean": self.noise_filter_tab.gaussianMean,
            "gaussian_std": self.noise_filter_tab.gaussianStd,
            "salt_prob": self.noise_filter_tab.saltProb,
            "pepper_prob": self.noise_filter_tab.pepperProb
        }
        
        for widget in noise_filter_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("noise_filter", noise_filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("noise_filter", noise_filter_ui))      
        
        
        # Noise & Filter Tab - Filter UI Components
        filter_ui = {
            "filter_type": self.noise_filter_tab.filterType,
            "kernel_size": self.noise_filter_tab.kernelSize,
            "sigma_value": self.noise_filter_tab.sigmaValue
        }
        for widget in filter_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("filter", filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("filter", filter_ui))

        # Edge Detection Tab
        self.edge_detection_ui = {
            "detector_method": self.edge_detection_tab.edgeType,
            "kernel_size": self.edge_detection_tab.sobelKernelSize,
            "sigma": self.edge_detection_tab.sobelSigma,
            "low_threshold": self.edge_detection_tab.cannyLowThreshold,
            "high_threshold": self.edge_detection_tab.cannyHighThreshold,
            "max_edge_val": self.edge_detection_tab.cannyMaxEdgeVal,
            "min_edge_val": self.edge_detection_tab.cannyMinEdgeVal,
            "threshold": self.edge_detection_tab.prewittThreshold,
            "value": self.edge_detection_tab.prewittValue
        }
        # Edge Detection Tab
        for widget in self.edge_detection_ui.values():
            if isinstance(widget, QComboBox):
                # Connect QComboBox's currentTextChanged signal
                widget.currentTextChanged.connect(lambda: self.update_params("edge_detection", self.edge_detection_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                # Connect QSpinBox/QDoubleSpinBox's valueChanged signal
                widget.valueChanged.connect(lambda: self.update_params("edge_detection", self.edge_detection_ui))
 
        # Thresholding Tab
        self.thresholding_ui = {
            "threshold_type": self.thresholding_tab.thresholdType,
            "T": self.thresholding_tab.globalThreshold,
            "kernal": self.thresholding_tab.kernelSizeThreshold,
            "k": self.thresholding_tab.kValue
        }
        for widget in self.thresholding_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("thresholding", self.thresholding_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("thresholding", self.thresholding_ui))

        # Frequency Filter Tab
        self.frequency_filter_ui = {
            "filter_type": self.frequency_filter_tab.freqType,
            "radius": self.frequency_filter_tab.freqRadius
        }
        for widget in self.frequency_filter_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("frequency_filter", self.frequency_filter_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("frequency_filter", self.frequency_filter_ui))

        # Hybrid Image Tab
        self.hybrid_image_ui = {
            "cutoff1": self.hybrid_image_tab.cutoff1,
            "cutoff2": self.hybrid_image_tab.cutoff2,
            "type1": self.hybrid_image_tab.type1,
            "type2": self.hybrid_image_tab.type2
        }
        for widget in self.hybrid_image_ui.values():
            if isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(lambda: self.update_params("hybrid_image", self.hybrid_image_ui))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda: self.update_params("hybrid_image", self.hybrid_image_ui))    
        # Connect apply buttons
        self.noise_filter_tab.btn_noise.clicked.connect(self.apply_noise)
           # Connect the "Apply Filter" button
        self.noise_filter_tab.btn_filter.clicked.connect(self.apply_filter)
        # self.edge_detection_tab.btn_edge_detection.clicked.connect(lambda: self.process_image("detect_edges", **self.params["edge_detection"]))
        # self.thresholding_tab.btn_threshold.clicked.connect(lambda: self.process_image("apply_thresholding", **self.params["thresholding"]))
        # self.frequency_filter_tab.btn_freq_filter.clicked.connect(lambda: self.process_image("apply_frequency_filter", **self.params["frequency_filter"]))
        # self.hybrid_image_tab.btn_hybrid.clicked.connect(lambda: self.process_image("create_hybrid_image", **self.params["hybrid_image"]))
    
    def update_params(self, tab_name, ui_components):
        """
        Update the parameters for a specific tab based on the UI components.
        
        Args:
            tab_name (str): The name of the tab (e.g., "noise_filter").
            ui_components (dict): A dictionary of UI components and their keys.
        """
        print("Updating params for", tab_name)
        self.params[tab_name] = {}
        for key, widget in ui_components.items():
            if isinstance(widget, (QComboBox, QLineEdit)):
                self.params[tab_name][key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.params[tab_name][key] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.params[tab_name][key] = widget.isChecked()
        
        print(self.params[tab_name])        

    def on_image_label_double_click(self, event):
        self.load_image()
    

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio))

    def apply_noise(self):
        """
        Applies noise to the image based on the selected noise type and parameters from the UI.
        """
        # Retrieve noise parameters from the params dictionary
        noise_params = self.params["noise_filter"]

        # Call the add_noise function with the retrieved parameters
        self._add_noise(**noise_params)
        print("Applying noise:", noise_params)
        self.display_image(self.modified_image)
        
    def _add_noise(self, **kwargs):
        """
        Adds noise to the image based on the specified noise type and parameters.

        Args:
            noise_type (str): Type of noise to add. Options: "uniform", "gaussian", "salt_pepper".
            **kwargs: Additional parameters for the noise (e.g., intensity, mean, std, salt_prob, pepper_prob).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying new noise

        if self.image is not None:
            # Call the noise processor with the specified noise type and parameters
            noisy_image = self.processors['noise'].add_noise(**kwargs)
            self.modified_image = noisy_image
            self.display_image(self.modified_image, modified=True)
        else:
            raise ValueError("No image loaded. Please load an image before applying noise.")
            
    def apply_filter(self):
        """
        Applies a filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve filter parameters from the params dictionary
        filter_params = self.params.get("filtering", {})
        # Call the apply_filters function with the retrieved parameters
        self._apply_filters(**filter_params)
        print("Applying filter:", filter_params)
        self.display_image(self.modified_image)

    def _apply_filters(self, **kwargs):
        """
        Applies a filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of filter to apply. Options: "average", "gaussian", "median".
            **kwargs: Additional parameters for the filter (e.g., kernel_size, sigma).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying a new filter

        if self.image is not None:
            # Apply the filter using the specified parameters
            filtered_image = self.processors['noise'].apply_filters( **kwargs)
            self.modified_image = filtered_image.get(kwargs.get("filter_type", "median")) # Default to "median" filter
        else:
            raise ValueError("No image loaded. Please load an image before applying a filter.")
    def detect_edges(self):
        """
        Detects edges in the image based on the selected edge detection type and parameters from the UI.
        """
        # Retrieve edge detection parameters from the params dictionary
        edge_params = self.params.get("edge_detection", {})

        # Call the _detect_edges function with the retrieved parameters
        self._detect_edges(**edge_params)
        print("Detecting edges:", edge_params)
        
        self.display_image(self.modified_image, modified=True)

    def _detect_edges(self, **kwargs):
        """ 
        Detects edges in the image based on the specified edge detection type and parameters.

        Args:
            edge_type (str): Type of edge detection to apply. Options: "sobel", "canny", "prewitt", "roberts".
            **kwargs: Additional parameters for the edge detection (e.g., kernel_size, sigma, thresholds).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying edge detection

        if self.image is not None:
            # Apply edge detection using the specified parameters
            edge_map = self.processors['edge_detector'].detect_edges(**kwargs)
            self.modified_image = edge_map
        else:
            raise ValueError("No image loaded. Please load an image before detecting edges.")
    
    def apply_thresholding(self):
        """
        Applies thresholding to the image based on the selected thresholding type and parameters from the UI.
        """
        # Retrieve thresholding parameters from the params dictionary
        threshold_params = self.params.get("thresholding", {})
        # Call the _apply_thresholding function with the retrieved parameters
        print("Applying thresholding:", threshold_params)

        self._apply_thresholding( **threshold_params)
        self.display_image(self.modified_image, modified=True)
        

    def _apply_thresholding(self,  **kwargs):
        """
        Applies thresholding to the image based on the specified thresholding type and parameters.

        Args:
            threshold_type (str): Type of thresholding to apply. Options: "global", "local".
            **kwargs: Additional parameters for the thresholding (e.g., threshold_value, kernel_size, k_value).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying thresholding

        if self.image is not None:
            # Apply thresholding using the specified parameters
            thresholded_image = self.processors['thresholding'].apply_thresholding(**kwargs)
            self.modified_image = thresholded_image
        else:
            raise ValueError("No image loaded. Please load an image before applying thresholding.")
    
    def show_histogram(self):
        
        # self.processors['histogram'].plot_all_histograms()
        self.processors['histogram'].set_image(self.image)

        self.visualization_widget = HistogramVisualizationWidget(processor  = self.processors['histogram'])
        self.visualization_widget.show()
    
    def apply_frequency_filter(self):
        """
        Applies a frequency filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve frequency filter parameters from the params dictionary
        frequency_params = self.params.get("frequency_filter", {})
        # Call the _apply_frequency_filter function with the retrieved parameters
        self._apply_frequency_filter(**frequency_params)
        print("Applying frequency filter:", frequency_params)
        self.display_image(self.modified_image, modified=True)
        

    def _apply_frequency_filter(self, **kwargs):
        """
        Applies a frequency filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of frequency filter to apply. Options: "low_pass", "high_pass".
            **kwargs: Additional parameters for the frequency filter (e.g., radius).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying frequency filtering

        if self.image is not None:
            # Apply frequency filter using the specified parameters
            filtered_image = self.processors['frequency'].apply_filter(**kwargs)
            self.modified_image = filtered_image
        else:
            raise ValueError("No image loaded. Please load an image before applying frequency filtering.")
   
    def equalize(self):
        self.modified_image = self.processors['image'].get_equalized_image()
        self.display_image(self.modified_image)

    def normalize(self):
        self.modified_image = self.processors['image'].get_normalized_image()
        self.display_image(self.modified_image)

    def load_image(self, hybird = False):
        """
        Load an image from disk and display it in the UI.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path and hybird == False:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.original_image = self.image
            if self.image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.display_image(self.image)
        elif hybird == True:
            self.extra_image = cv2.imread(file_path)
            if self.extra_image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return

            self.display_image(self.extra_image, hybird = True)
        else:
            QMessageBox.information(self, "Info", "No file selected.")

    def display_image(self, img, hybrid=False, modified=False):
        """
        Convert a NumPy BGR image to QImage and display it in lbl_image.
        """
        if len(img.shape) == 3:
            # Convert BGR to RGB
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            h, w = img.shape
            # Ensure the image is in uint8 format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Convert the NumPy array to bytes
            img_bytes = img.tobytes()
            qimg = QImage(img_bytes, w, h, w, QImage.Format_Indexed8)
        
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(
            self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio
        ))
    

    def equalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
            
        if self.image is not None:
            equalized_image = self.processors['image'].get_equalized_image() 
            self.modified_image = equalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    def normalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
        if self.image is not None:
            normalized_image = self.processors['image'].get_normalized_image() 
            self.modified_image = normalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def confirm_edit(self):
        """
        Confirm the edit.
        """
        if self.modified_image is not None:
            self.image = self.modified_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def discard_edit(self):
        """
        Discard the edit.
        """
        if self.modified_image is not None:
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    def reset_image(self):
        """
        Reset the image to the original.
        """
        if self.original_image is not None:
            self.image = self.original_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No original image available. Load an image first.")

    def detect_lines(self):
        """
        Detects lines in the image using the Hough Transform based on the selected parameters from the UI.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        shape_params = self.params["shape_detection"]
    
        
        self.modified_image = self.processors['edge_detector'].detect_shape(
            shape_type = 'line',
            **shape_params
        )

        self.display_image(self.modified_image)

    def detect_circles(self):
        """
        Detects lines in the image using the Hough Transform based on the selected parameters from the UI.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        shape_params = self.params["shape_detection"]

        
        self.modified_image = self.processors['edge_detector'].detect_shape(
            shape_type = 'circle',
            **shape_params
        )

        self.display_image(self.modified_image)

    def detect_ellipses(self):
        """
        Detects lines in the image using the Hough Transform based on the selected parameters from the UI.
        """
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        shape_params = self.params["shape_detection"]

        
        self.modified_image = self.processors['edge_detector'].detect_shape(
            shape_type = 'ellipse',
            **shape_params
        )

        self.display_image(self.modified_image)

    # def detect_circles(self):
    #     """
    #     Detects circles in the image using the Hough Transform based on the selected parameters from the UI.
    #     """
    #     if self.image is None:
    #         QMessageBox.warning(self, "Warning", "Please load an image first.")
    #         return

    #     try:
    #         min_edge_threshold = self.hough_transform_tab.minEdgeThreshold.value()
    #         max_edge_threshold = self.hough_transform_tab.maxEdgeThreshold.value()
    #         r_min = self.hough_transform_tab.rMin.value()
    #         r_max = self.hough_transform_tab.rMax.value()
    #         delta_r = self.hough_transform_tab.deltaR.value()
    #         num_thetas = self.hough_transform_tab.numThetas.value()
    #         bin_threshold = self.hough_transform_tab.binThreshold.value()

    #         self.modified_image = detect_circles(
    #             self.image,
    #             min_edge_threshold=min_edge_threshold,
    #             max_edge_threshold=max_edge_threshold,
    #             r_min=r_min,
    #             r_max=r_max,
    #             delta_r=delta_r,
    #             num_thetas=num_thetas,
    #             bin_threshold=bin_threshold
    #         )

    #         self.display_image(self.modified_image)
    #     except Exception as e:
    #         print(f"Error in detect_circles: {e}")
    #         QMessageBox.critical(self, "Error", f"Failed to detect circles: {e}")


def main():
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qss_path = os.path.join(script_dir, "../resources/styles.qss")
    
    with open(qss_path, "r") as file:
        app.setStyleSheet(file.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

