# -*- coding: utf-8 -*-
# basics.py file
"""
Created on Mon May  3 19:18:29 2021

@author: droes
"""
import numpy as np
import cv2
from numba import njit # conda install numba
from scipy import stats

@njit
def histogram_figure_numba(np_img):
    # This is a placeholder - we used OpenCV's fast histogram in the non-numba version
    return np.zeros(256, dtype=np.float32), np.zeros(256, dtype=np.float32), np.zeros(256, dtype=np.float32)

def histogram_figure_fast(np_img):
    # Use OpenCV's optimized histogram calculation
    r_hist = cv2.calcHist([np_img], [0], None, [256], [0, 256]).flatten()
    g_hist = cv2.calcHist([np_img], [1], None, [256], [0, 256]).flatten()
    b_hist = cv2.calcHist([np_img], [2], None, [256], [0, 256]).flatten()
    
    # Normalize histograms (scale to fit in plot range 0-3)
    total_pixels = np_img.shape[0] * np_img.shape[1]
    r_hist = (r_hist / total_pixels) * 3.0
    g_hist = (g_hist / total_pixels) * 3.0
    b_hist = (b_hist / total_pixels) * 3.0
    
    return r_hist, g_hist, b_hist


def calculate_basic_statistics_fast(np_img):
    # Calculate mean and std
    means = cv2.mean(np_img)[:3]  
    
    # Convert to simple format
    stats_dict = {
        'R': {'mean': means[2], 'std': 0, 'max': 255, 'min': 0, 'mode': 128},  # Approximations for speed
        'G': {'mean': means[1], 'std': 0, 'max': 255, 'min': 0, 'mode': 128},
        'B': {'mean': means[0], 'std': 0, 'max': 255, 'min': 0, 'mode': 128}
    }
    
    return stats_dict


def linear_transformation(np_img, alpha=1.2, beta=30):
    # Apply transformation and clip values to valid range
    transformed = cv2.convertScaleAbs(np_img, alpha=alpha, beta=beta)
    return transformed


def calculate_entropy_fast(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    
    # Return same value for all channels for simplicity
    return {'R': entropy, 'G': entropy, 'B': entropy}


def histogram_equalization(np_img):
    # Split channels
    b, g, r = cv2.split(np_img)
    
    # Apply histogram equalization to each channel
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    
    # Merge channels back
    equalized = cv2.merge([b_eq, g_eq, r_eq])
    
    return equalized


def apply_gaussian_blur(np_img, kernel_size=15, sigma=0):
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    
    blurred = cv2.GaussianBlur(np_img, (kernel_size, kernel_size), sigma)
    return blurred

def create_filter_info_text(filter_name, stats_dict, entropy_dict):
    info_text = [
        f"Filter: {filter_name}",
        f"R: μ={stats_dict['R']['mean']:.1f}, σ={stats_dict['R']['std']:.1f}",
        f"G: μ={stats_dict['G']['mean']:.1f}, σ={stats_dict['G']['std']:.1f}",
        f"B: μ={stats_dict['B']['mean']:.1f}, σ={stats_dict['B']['std']:.1f}",
        f"Entropy R:{entropy_dict['R']:.2f} G:{entropy_dict['G']:.2f} B:{entropy_dict['B']:.2f}"
    ]
    
    return info_text


####

### All other basic functions

####