import tkinter as tk
from tkinter import ttk, filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
import glob

# Import ADAS class from index-developed_adas.py using importlib
spec = importlib.util.spec_from_file_location("adas_module", "index-developed_adas.py")
adas_module = importlib.util.module_from_spec(spec)
sys.modules["adas_module"] = adas_module
spec.loader.exec_module(adas_module)
ADAS = adas_module.ADAS  # Access the ADAS class

class ADASDashboard:
    def __init__(self, root, seg_model_path, lane_det_model_path):
        self.root = root
        self.root.title("SadakSahayak: Advanced Driver Assistance System")
        self.root.geometry("1200x800")
        
        # Set base assets path 
        self.base_assets_path = "C:/IOMP/vision-based-adas-main/vision-based-adas-main/assets"
        
        # Initialize ADAS backend
        try:
            print(f"Initializing ADAS with models:\nSegmentation: {seg_model_path}\nLane Detection: {lane_det_model_path}")
            self.adas = ADAS(seg_model_path, lane_det_model_path)
            print("ADAS initialized successfully")
        except Exception as e:
            print(f"Error initializing ADAS: {e}")
            raise
        
        # Initialize image paths
        self.image_paths = []
        self.current_image_index = -1
        # Overlay and threshold parameters
        self.seg_overlay_alpha = tk.DoubleVar(value=0.3)
        #self.lane_departure_thresh = tk.DoubleVar(value=0.15)  # as fraction of frame width
        self.fcw_distance_thresh = tk.IntVar(value=500)        # in pixels           
        # Create UI elements
        self.create_main_frame()
        self.create_control_panel()
        self.create_visualization_area()     
    
    def create_main_frame(self):
        # Main container frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_control_panel(self):
        # Control panel on the right side
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # Image source selection
        ttk.Label(self.control_frame, text="Image Source:").pack(anchor=tk.W, padx=5, pady=5)
        
        # Select images directory button
        ttk.Button(self.control_frame, text="Select Images Directory", 
                  command=self.select_image_directory).pack(padx=5, pady=5)
               
        # Select single image button
        ttk.Button(self.control_frame, text="Select Single Image", 
                  command=self.select_single_image).pack(padx=5, pady=5)
        
        # Navigation buttons
        self.nav_frame = ttk.Frame(self.control_frame)
        self.nav_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(self.nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.nav_frame, text="Next", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        
        # Image counter
        self.counter_var = tk.StringVar(value="No images loaded")
        ttk.Label(self.control_frame, textvariable=self.counter_var).pack(pady=5)
        
        # Feature toggles
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        ttk.Label(self.control_frame, text="Display Mode:").pack(anchor=tk.W, padx=5, pady=5)
        
        # Use radio buttons for exclusive selection
        self.display_mode = tk.StringVar(value="original")
        
        ttk.Radiobutton(self.control_frame, text="Original Image", 
                        variable=self.display_mode, value="original",
                        command=self.process_current_image).pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(self.control_frame, text="Lane Detection", 
                        variable=self.display_mode, value="lane_detection",
                        command=self.process_current_image).pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(self.control_frame, text="Road Segmentation", 
                        variable=self.display_mode, value="segmentation",
                        command=self.process_current_image).pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(self.control_frame, text="Forward Collision Warning", 
                        variable=self.display_mode, value="fcw",
                        command=self.process_current_image).pack(anchor=tk.W, padx=20)
              # Blind‑spot detection mode
        ttk.Radiobutton(self.control_frame, text="Blind Spot Detection", 
                     variable=self.display_mode, value="blind_spot",
                     command=self.process_current_image).pack(anchor=tk.W, padx=20)
        
        ttk.Radiobutton(self.control_frame, text="All Features", 
                        variable=self.display_mode, value="all",
                        command=self.process_current_image).pack(anchor=tk.W, padx=20)
      
        # Remove all features button
        ttk.Button(self.control_frame, text="Remove All Features", 
                   command=self.remove_all_features).pack(padx=5, pady=10)
        
        # Load example images buttons
        ttk.Separator(self.control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(self.control_frame, text="Load Lane Detection Examples", 
                  command=lambda: self.load_example_images("lane-detection/sample images from the val set/fcn-preds")).pack(padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="Load Road Segmentation Examples", 
                  command=lambda: self.load_example_images("road-segmentation")).pack(padx=5, pady=5)
        
        ttk.Button(self.control_frame, text="Load FCW Examples", 
                  command=lambda: self.load_example_images("FCW-images")).pack(padx=5, pady=5)
               
        # Capture screenshot button
        ttk.Button(self.control_frame, text="Capture Screenshot", 
                  command=self.capture_screenshot).pack(padx=5, pady=5)

        # --- Overlay & Threshold Controls ---
        #ttk.Label(self.control_frame, text="Overlay/Threshold Controls:").pack(anchor=tk.W, padx=5, pady=(15, 0))

        # Segmentation overlay alpha
        #ttk.Label(self.control_frame, text="Segmentation Overlay Alpha").pack(anchor=tk.W, padx=15)
        #ttk.Scale(self.control_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.seg_overlay_alpha,
         #         command=lambda e: self.process_current_image(), length=150).pack(anchor=tk.W, padx=20)

        # Lane departure threshold
        #ttk.Label(self.control_frame, text="Lane Departure Threshold (% width)").pack(anchor=tk.W, padx=15)
        #ttk.Scale(self.control_frame, from_=0.05, to=0.5, orient=tk.HORIZONTAL, variable=self.lane_departure_thresh,
         #         command=lambda e: self.process_current_image(), length=150).pack(anchor=tk.W, padx=20)

        # FCW distance threshold
        #ttk.Label(self.control_frame, text="FCW Distance Threshold (px)").pack(anchor=tk.W, padx=15)
        #ttk.Scale(self.control_frame, from_=100, to=1000, orient=tk.HORIZONTAL, variable=self.fcw_distance_thresh,
         #         command=lambda e: self.process_current_image(), length=150).pack(anchor=tk.W, padx=20)                  
    def create_visualization_area(self):
        # Main visualization area
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image display
        self.image_label = ttk.Label(self.viz_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status bar
        self.status_frame = ttk.Frame(self.viz_frame)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        # FCW warning indicator
        self.warning_indicator = ttk.Label(self.status_frame, text="SAFE", 
                                          background="green", foreground="white", width=10)
        self.warning_indicator.pack(side=tk.LEFT, padx=5)

      # Blind‑spot indicator
        self.blind_spot_indicator = ttk.Label(self.status_frame, text="NO BLIND SPOT",
                                           background="green", foreground="white", width=15)
        self.blind_spot_indicator.pack(side=tk.LEFT, padx=5)
        
        # Image info label
        self.info_label = ttk.Label(self.status_frame, text="No image loaded")
        
        # Image info label
        self.info_label = ttk.Label(self.status_frame, text="No image loaded")
        self.info_label.pack(side=tk.RIGHT, padx=5)
    
    def select_image_directory(self):
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.image_paths.extend(glob.glob(os.path.join(directory, ext)))
            
            if self.image_paths:
                self.current_image_index = 0
                self.update_counter()
                self.load_and_process_image()
            else:
                self.counter_var.set("No images found in directory")
    
    def select_single_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.update_counter()
            self.load_and_process_image()
    
    def load_example_images(self, folder_name):
        # Use the absolute path to the examples
        directory = os.path.join(self.base_assets_path, folder_name)
        print(f"Looking for examples in: {directory}")
        
        if os.path.exists(directory):
            # Check if there's a sample subfolder
            sample_dir = os.path.join(directory, "sample images from the val set")
            if os.path.exists(sample_dir):
                directory = sample_dir
                print(f"Using sample directory: {directory}")
            
            self.image_paths = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                self.image_paths.extend(glob.glob(os.path.join(directory, ext)))
            
            print(f"Found {len(self.image_paths)} images")
            
            if self.image_paths:
                self.current_image_index = 0
                self.update_counter()
                self.load_and_process_image()
            else:
                self.counter_var.set("No example images found")
        else:
            self.counter_var.set(f"Example directory not found: {directory}")
    
    def next_image(self):
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
        self.update_counter()
        self.load_and_process_image()
    
    def prev_image(self):
        if not self.image_paths:
            return
        
        self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
        self.update_counter()
        self.load_and_process_image()
    
    def update_counter(self):
        if self.image_paths:
            self.counter_var.set(f"Image {self.current_image_index + 1} of {len(self.image_paths)}")
    
    def load_and_process_image(self):
        """Load the current image and process it with ADAS"""
        if not self.image_paths or self.current_image_index < 0:
            return
        
        try:
            img_path = self.image_paths[self.current_image_index]
            self.process_image(img_path)
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
    
    def process_current_image(self):
        """Process the current image again (used when toggles change)"""
        if not self.image_paths or self.current_image_index < 0:
            return
            
        self.load_and_process_image()
    
    def remove_all_features(self):
        """Switch to original image mode (no features)"""
        self.display_mode.set("original")
        self.process_current_image()
    
    def process_image(self, img_path):
        """Process an image with ADAS and display the results"""
        self.info_label.configure(text=os.path.basename(img_path))
        
        try:
            # Get the original image for reference
            original_img = cv2.imread(img_path)
            # --- BLIND‑SPOT DETECTION ---
            h, w = original_img.shape[:2]
            left_w = int(0.1 * w)
            right_x = int(0.9 * w)
            gray   = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            edges  = cv2.Canny(gray, 50, 150)
            # Find contours in left ROI
            left_roi = edges[:, :left_w]
            contours_left, _ = cv2.findContours(left_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area_left = max([cv2.contourArea(cnt) for cnt in contours_left], default=0)

            # Find contours in right ROI
            right_roi = edges[:, right_x:]
            contours_right, _ = cv2.findContours(right_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area_right = max([cv2.contourArea(cnt) for cnt in contours_right], default=0)

            # Threshold: only flag if a large object is present (tune 500 as needed)
            area_thresh = 500
            flag_left = max_area_left > area_thresh
            flag_right = max_area_right > area_thresh
            blind_spot_flag = flag_left or flag_right

            if blind_spot_flag:
                self.blind_spot_indicator.configure(text="BLIND SPOT!", background="red")
            else:
                self.blind_spot_indicator.configure(text="NO BLIND SPOT", background="green")
            
            # Run ADAS on the selected image
            flag, image, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi = self.adas.run(img_path)
            
            # Update warning indicator based on FCW flag
            if flag and (self.display_mode.get() == "fcw" or self.display_mode.get() == "all"):
                self.warning_indicator.configure(text="WARNING", background="red")
            else:
                self.warning_indicator.configure(text="SAFE", background="green")
            
            # Select visualization based on display mode
            display_mode = self.display_mode.get()
# Show/hide blind spot indicator based on mode
            if display_mode == "blind_spot" or display_mode == "all":
                self.blind_spot_indicator.pack(side=tk.LEFT, padx=5)
            else:
                self.blind_spot_indicator.pack_forget()

            if display_mode == "original":
                visualization = original_img.copy()

            elif display_mode == "lane_detection":
                visualization = original_img.copy()
                if lane_mask is not None:
                    h, w = visualization.shape[:2]
                    if lane_mask.shape[:2] != (h, w):
                        lane_mask_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        lane_mask_resized = lane_mask
                    lane_overlay = np.zeros_like(visualization)
                    lane_overlay[lane_mask_resized > 0] = [0, 255, 255]
                    visualization = cv2.addWeighted(visualization, 0.7, lane_overlay, 0.3, 0)

            elif display_mode == "segmentation":
                if fcw_mask is not None:
                    visualization = fcw_mask
                else:
                    visualization = original_img.copy()

            elif display_mode == "fcw":
                if fcw_image_roi is not None:
                    visualization = fcw_image_roi
                else:
                    visualization = original_img.copy()

            elif display_mode == "all":
                # Start with the original image as base
                visualization = original_img.copy()
                
                # Add lane detection overlay if available
                if lane_mask is not None:
                    h, w = visualization.shape[:2]
                    if lane_mask.shape[:2] != (h, w):
                        lane_mask_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        lane_mask_resized = lane_mask
                    lane_overlay = np.zeros_like(visualization)
                    lane_overlay[lane_mask_resized > 0] = [0, 255, 255]  # Yellow
                    visualization = cv2.addWeighted(visualization, 0.7, lane_overlay, 0.3, 0)
                
                # Add segmentation overlay if available
                if fcw_mask is not None and len(fcw_mask.shape) == 3:
                    seg_mask = fcw_mask.copy()
                    if seg_mask.shape[:2] != (h, w):
                        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Add these lines BEFORE the cv2.addWeighted call in line 375:
                    if visualization.dtype != np.uint8:
                        visualization = (visualization * 255).astype(np.uint8) if np.max(visualization) <= 1.0 else visualization.astype(np.uint8)
                    if seg_mask.dtype != np.uint8:
                        seg_mask = (seg_mask * 255).astype(np.uint8) if np.max(seg_mask) <= 1.0 else seg_mask.astype(np.uint8)
                    # Blend with lower opacity to not overwhelm the image
                    visualization = cv2.addWeighted(visualization, 0.7, seg_mask, 0.3, 0)
                
                # If FCW warning is active, highlight it in this view
                if flag and fcw_image_roi is not None:
                    h, w = visualization.shape[:2]
                    if fcw_image_roi.shape[:2] != (h, w):
                        roi = cv2.resize(fcw_image_roi, (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        roi = fcw_image_roi.copy()
                    if len(roi.shape) == 3:
                        # Extract pixels that are predominantly red
                        is_red = (roi[:,:,2] > 200) & (roi[:,:,0] < 50) & (roi[:,:,1] < 50)
                        visualization[is_red] = [0, 0, 255]  # Make them bright red
                
                # Add blind spot ROIs
                cv2.rectangle(visualization, (0, 0), (left_w, h), (255, 0, 0), 2)
                cv2.rectangle(visualization, (right_x, 0), (w, h), (255, 0, 0), 2)
                
                # Add edge overlay with very low opacity
                edge_color = np.dstack([edges] * 3)
                if edge_color.shape[:2] != visualization.shape[:2]:
                    edge_color = cv2.resize(edge_color, (visualization.shape[1], visualization.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
                
                # Ensure both are uint8 before blending
                if visualization.dtype != np.uint8:
                    visualization = (visualization * 255).astype(np.uint8) if np.max(visualization) <= 1.0 else visualization.astype(np.uint8)
                if edge_color.dtype != np.uint8:
                    edge_color = (edge_color * 255).astype(np.uint8) if np.max(edge_color) <= 1.0 else edge_color.astype(np.uint8)
                
                # Blend edges with very low opacity (0.1)
                visualization = cv2.addWeighted(visualization, 0.9, edge_color, 0.1, 0)

            elif display_mode == "blind_spot":
                visualization = original_img.copy()
                cv2.rectangle(visualization, (0, 0), (left_w, h), (255, 0, 0), 2)
                cv2.rectangle(visualization, (right_x, 0), (w, h), (255, 0, 0), 2)
                edge_color = np.dstack([edges] * 3)

                # Ensure edge_color matches visualization shape
                if edge_color.shape[:2] != visualization.shape[:2]:
                    edge_color = cv2.resize(edge_color, (visualization.shape[1], visualization.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Ensure both are uint8
                if visualization.dtype != np.uint8:
                    visualization = (visualization * 255).astype(np.uint8) if np.max(visualization) <= 1.0 else visualization.astype(np.uint8)
                if edge_color.dtype != np.uint8:
                    edge_color = (edge_color * 255).astype(np.uint8) if np.max(edge_color) <= 1.0 else edge_color.astype(np.uint8)

                visualization = cv2.addWeighted(visualization, 0.8, edge_color, 0.2, 0)

            else:
                visualization = original_img.copy()
            
            # Convert for display
            # Ensure visualization is in uint8 format
            if visualization.dtype != np.uint8:
                if np.max(visualization) <= 1.0:
                    visualization = (visualization * 255).astype(np.uint8)
                else:
                    visualization = visualization.astype(np.uint8)
            
            # Convert BGR to RGB for PIL
            visualization_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(visualization_rgb)
            
            # Resize for display
            display_width = self.viz_frame.winfo_width() - 20
            if display_width < 100:
                display_width = 800
            
            w, h = pil_img.size
            ratio = min(display_width / w, 600 / h)
            new_size = (int(w * ratio), int(h * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            
            # Update display
            img_tk = ImageTk.PhotoImage(image=pil_img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            print(f"Error in process_image: {e}")
            import traceback
            traceback.print_exc()
            
            # Display original image on error
            try:
                original_img = cv2.imread(img_path)
                if original_img is not None:
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(original_img)
                    display_width = 800
                    w, h = pil_img.size
                    ratio = min(display_width / w, 600 / h)
                    new_size = (int(w * ratio), int(h * ratio))
                    pil_img = pil_img.resize(new_size, Image.LANCZOS)
                    img_tk = ImageTk.PhotoImage(image=pil_img)
                    self.image_label.config(image=img_tk)
                    self.image_label.image = img_tk
                    self.warning_indicator.configure(text="ERROR", background="yellow")
            except Exception as e2:
                print(f"Could not display original image either: {e2}")
    
    def capture_screenshot(self):
        if not hasattr(self.image_label, 'image'):
            print("No image to capture")
            return
        
        try:
            os.makedirs(os.path.join(self.base_assets_path, "screenshots"), exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.base_assets_path, "screenshots", f"adas_screenshot_{timestamp}.png")
            
            if self.image_paths and self.current_image_index >= 0:
                # Get the current visualization being displayed
                display_mode = self.display_mode.get()
                img_path = self.image_paths[self.current_image_index]
                
                if display_mode == "original":
                    # Save original image
                    original_img = cv2.imread(img_path)
                    cv2.imwrite(filename, original_img)
                else:
                    # Run ADAS to get the correct visualization
                    flag, image, lane_mask, fcw_mask, fcw_image_mask, fcw_image_roi = self.adas.run(img_path)
                    
                    if display_mode == "lane_detection":
                        # Create lane detection visualization
                        original_img = cv2.imread(img_path)
                        if lane_mask is not None:
                            h, w = original_img.shape[:2]
                            lane_mask_resized = cv2.resize(lane_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            lane_overlay = np.zeros_like(original_img)
                            lane_overlay[lane_mask_resized > 0] = [0, 255, 255]
                            viz = cv2.addWeighted(original_img, 0.7, lane_overlay, 0.3, 0)
                            cv2.imwrite(filename, viz)
                        else:
                            cv2.imwrite(filename, original_img)
                    elif display_mode == "segmentation" and fcw_mask is not None:
                        cv2.imwrite(filename, fcw_mask)
                    elif display_mode == "fcw" and fcw_image_roi is not None:
                        cv2.imwrite(filename, fcw_image_roi)
                    elif display_mode == "all" and fcw_image_mask is not None:
                        cv2.imwrite(filename, fcw_image_mask)
                    else:
                        cv2.imwrite(filename, image)
                
                print(f"Screenshot saved as {filename}")
        except Exception as e:
            print(f"Error capturing screenshot: {e}")

# Main entry point
if __name__ == "__main__":
    # Get absolute paths to model files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Look for model files
    seg_model_path = None
    lane_det_model_path = None
    
    # Check in road segmentation directory
    seg_model_dir = os.path.join(root_dir, "models", "road-segmentation")
    if os.path.exists(seg_model_dir):
        for file in os.listdir(seg_model_dir):
            if file.endswith(".h5"):
                seg_model_path = os.path.join(seg_model_dir, file)
                break
    
    # Check in lane detection directory
    lane_det_dir = os.path.join(root_dir, "models", "lane-detection")
    if os.path.exists(lane_det_dir):
        lane_det_model_path = os.path.join(lane_det_dir, "save_at_32.h5")
        if not os.path.exists(lane_det_model_path):
            for file in os.listdir(lane_det_dir):
                if file.endswith(".h5"):
                    lane_det_model_path = os.path.join(lane_det_dir, file)
                    break
    
    # Print model paths
    print(f"Segmentation model: {seg_model_path}")
    print(f"Lane detection model: {lane_det_model_path}")
    
    # Start application
    root = tk.Tk()
    app = ADASDashboard(root, seg_model_path, lane_det_model_path)
    root.mainloop()