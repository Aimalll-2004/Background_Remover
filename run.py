# -*- coding: utf-8 -*-
# run.py file
"""
Created on Thu Apr 22 11:59:19 2021

@author: droes
"""

import keyboard # pip install keyboard
import cv2
import numpy as np
import mediapipe as mp # pip install mediapipe
import threading
import time

from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import *


class KeyboardHandler:
    def __init__(self):
        self.commands = {
            'filter_change': False,
            'background_toggle': False,
            'help': False,
            'quit': False
        }
        self.running = True
        self.thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.thread.start()
        
    def _keyboard_listener(self):
        while self.running:
            try:
                if keyboard.is_pressed('f'):
                    self.commands['filter_change'] = True
                    time.sleep(0.3)  # Prevent multiple triggers
                elif keyboard.is_pressed('b'):
                    self.commands['background_toggle'] = True
                    time.sleep(0.3)
                elif keyboard.is_pressed('h'):
                    self.commands['help'] = True
                    time.sleep(0.3)
                elif keyboard.is_pressed('q'):
                    self.commands['quit'] = True
                    time.sleep(0.3)
                time.sleep(0.05)  # Small delay to prevent high CPU usage
            except:
                # Fallback: if keyboard library fails, continue without keyboard input
                time.sleep(0.1)
                
    def get_and_reset_command(self, command):
        if command in self.commands:
            state = self.commands[command]
            self.commands[command] = False
            return state
        return False
        
    def stop(self):
        self.running = False


class FilterManager:
    def __init__(self):
        self.current_filter = 0
        self.filter_names = [
            "Original",
            "Linear Transform",
            "Histogram Equalization", 
            "Gaussian Blur"
        ]
        self.frame_counter = 0  
        self.stats_skip_counter = 0 
        
    def get_current_filter_name(self):
        return self.filter_names[self.current_filter]
    
    def next_filter(self):
        self.current_filter = (self.current_filter + 1) % len(self.filter_names)
        
    def apply_current_filter(self, img):
        if self.current_filter == 0:  # Original
            return img
        elif self.current_filter == 1:  # Linear Transform
            return linear_transformation(img, alpha=1.3, beta=20)
        elif self.current_filter == 2:  # Histogram Equalization
            return histogram_equalization(img)
        elif self.current_filter == 3:  # Gaussian Blur
            return apply_gaussian_blur(img, kernel_size=11)
        else:
            return img


class BackgroundRemover:
    def __init__(self):
        # Initialize MediaPipe Selfie Segmentation with model 0 (faster)
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)  # 0 is faster
        
        # Create backgrounds
        self.background_image = None
        self.background_enabled = False
        self.frame_skip = 0  # Skip some frames for background processing
        self.last_mask = None  # Cache last mask for skipped frames
        
    def create_gradient_background(self, height, width):
        # Create simpler gradient for better performance
        background = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simple horizontal gradient (faster than complex gradients)
        for x in range(width):
            ratio = x / width
            color_val = int(100 + ratio * 100)
            background[:, x] = [color_val, 50, 200]  # BGR format
            
        return background
    
    def remove_background(self, image):
        if not self.background_enabled:
            return image
            
        # Skip processing some frames to improve FPS
        self.frame_skip += 1
        if self.frame_skip < 3 and self.last_mask is not None:  # Process every 3rd frame
            # Use cached mask
            mask_3channel = self.last_mask
        else:
            self.frame_skip = 0
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image to get segmentation mask
            results = self.segmentation.process(rgb_image)
            mask = results.segmentation_mask
            
            # Create 3-channel mask and cache it
            mask_3channel = np.stack([mask] * 3, axis=-1)
            self.last_mask = mask_3channel
        
        # Create background if not exists or size changed
        height, width = image.shape[:2]
        if self.background_image is None or self.background_image.shape[:2] != (height, width):
            self.background_image = self.create_gradient_background(height, width)
        
        # Blend foreground and background
        output_image = (image * mask_3channel + 
                       self.background_image * (1 - mask_3channel)).astype(np.uint8)
        
        return output_image
    
    def toggle_background_removal(self):
        self.background_enabled = not self.background_enabled
        print(f"Background removal: {'ON' if self.background_enabled else 'OFF'}")


# Example function
# You can use this function to process the images from opencv
# This function must be implemented as a generator function
def custom_processing(img_source_generator):
    # Initialize components
    filter_manager = FilterManager()
    background_remover = BackgroundRemover()
    keyboard_handler = KeyboardHandler()
    
    # Use this figure to plot your histogram
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()
    
    # Display initial help
    print("Press 'f' to cycle through filters")
    print("Press 'b' to toggle background removal")
    print("Press 'h' for help")
    print("Press 'q' to quit")
    print("=========================")
    
    # Performance optimization variables
    stats_dict = {'R': {'mean': 0, 'std': 0, 'max': 255, 'min': 0, 'mode': 128},
                  'G': {'mean': 0, 'std': 0, 'max': 255, 'min': 0, 'mode': 128},
                  'B': {'mean': 0, 'std': 0, 'max': 255, 'min': 0, 'mode': 128}}
    entropy_dict = {'R': 7.0, 'G': 7.0, 'B': 7.0}
    
    try:
        for sequence in img_source_generator:
            # Handle keyboard input using our improved handler
            if keyboard_handler.get_and_reset_command('filter_change'):
                filter_manager.next_filter()
                print(f"Switched to filter: {filter_manager.get_current_filter_name()}")
                
            if keyboard_handler.get_and_reset_command('background_toggle'):
                background_remover.toggle_background_removal()
                
            if keyboard_handler.get_and_reset_command('help'):
                print("=== CONTROLS ===")
                print("Press 'f' to cycle through filters")
                print("Press 'b' to toggle background removal")
                print("Press 'h' for help")
                print("Press 'q' to quit")
                print("Current filter:", filter_manager.get_current_filter_name())
                
            if keyboard_handler.get_and_reset_command('quit'):
                print("Quitting...")
                break

            # SPECIAL TASK: Apply MediaPipe background removal 
            sequence = background_remover.remove_background(sequence)
            
            # Apply current image processing filter
            processed_sequence = filter_manager.apply_current_filter(sequence)
            
            # Calculate statistics ONLY every 5 frames for better performance
            filter_manager.stats_skip_counter += 1
            if filter_manager.stats_skip_counter >= 5:
                filter_manager.stats_skip_counter = 0
                stats_dict = calculate_basic_statistics_fast(processed_sequence)
                entropy_dict = calculate_entropy_fast(processed_sequence)
            
            ###
            ### Histogram overlay example (optimized)
            ###
            
            # Update histogram only every other frame for better performance
            if filter_manager.frame_counter % 2 == 0:
                # Load the histogram values using our optimized function
                r_bars, g_bars, b_bars = histogram_figure_fast(processed_sequence)        
                
                # Update the histogram with new data
                update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)
            
            # Use the figure to create the overlay
            processed_sequence = plot_overlay_to_image(processed_sequence, fig)
            
            ###
            ### END Histogram overlay example
            ###

            # Create informative text display (simplified for performance)
            display_text = [
                f"Filter: {filter_manager.get_current_filter_name()}",
                f"R: {stats_dict['R']['mean']:.0f}",
                f"G: {stats_dict['G']['mean']:.0f}",
                f"B: {stats_dict['B']['mean']:.0f}",
            ]
            
            # Add background removal status
            if background_remover.background_enabled:
                display_text.append("Background: REMOVED")
            else:
                display_text.append("Background: ORIGINAL")
                
            # Add controls info and auto-cycle status
            display_text.extend([
                "F=Filter, B=Background, H=Help",
            ])
            
            # Display text on image
            processed_sequence = plot_strings_to_image(processed_sequence, display_text)

            # Make sure to yield your processed image
            yield processed_sequence
            
    finally:
        # Clean up keyboard handler
        keyboard_handler.stop()


def main():
    print("- Fast OpenCV histogram calculation")
    print("- Reduced MediaPipe model complexity")
    print("- Frame skipping for expensive operations")
    print("- Simplified statistics calculation")
    print("========================================")
    
    # Reduced resolution for better performance - adjust as needed
    width = 1280  
    height = 720  
    fps = 30
    
    # Define your virtual camera
    vc = VirtualCamera(fps, width, height)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # Use camera stream (recommended for background removal)
            # vc.capture_cv_video(0, bgr_to_rgb=True)
            
            # Or use screen capture (comment out camera and uncomment this)
            vc.capture_screen()
        )
    )

if __name__ == "__main__":
    main()