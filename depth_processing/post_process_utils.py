import os

import cv2
import numpy as np
import scipy.ndimage

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# labels based on ADE20K
FG = [4, 7, 10, 12, 15, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 30, 31, 32, 33, 
      34, 35, 36, 37, 38, 39, 41, 43, 44, 45, 47, 49, 50, 55, 56, 57, 58, 62, 
      63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 83, 85, 86, 87, 88, 89, 90, 92, 93, 95, 96, 97, 98, 99, 100, 102, 
      103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 
      118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 
      133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 
      147, 148, 149]
BG = [0, 1, 3, 5, 6, 8, 9, 11, 13, 14, 16, 25, 26, 29, 40, 42, 46, 48, 51, 52, 53, 54, 59, 60, 61, 68, 84, 91, 94, 101, 113, 128]
SKY = 2

def post_process(depth, semantic_map, depth_pho=None):
    if depth_pho is not None:
        ratio = depth / (depth_pho + 1.e-6)
        mask_swap = ratio > 1.1
        depth[mask_swap] = depth_pho[mask_swap]

    depth_dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    depth_dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    depth_grad = np.sqrt(depth_dx**2 + depth_dy**2)
    edge_mask = depth_grad > 50.
    depth[edge_mask] = 0.0

    depth_median_filtered = scipy.ndimage.median_filter(depth, size=7)
    unstable_score = np.max(np.concatenate([
        np.abs(depth / (depth_median_filtered + 1e-8))[..., None],
        np.abs(depth_median_filtered / (depth + 1e-8))[..., None]
    ], axis=-1), axis=-1)
    unstable_mask = unstable_score > 1.15
    depth[unstable_mask] = 0.0

    for label in np.unique(semantic_map):
        if label in BG:
            continue
        if label == SKY:
            continue
        mask = (semantic_map == label)
        depth_label = depth[mask]
        if depth_label.max() == 0.:
            continue
        labels_im, num_labels = scipy.ndimage.label(mask)
        for i in range(1, num_labels+1):
            component = (labels_im == i)
            depth_component = depth[component]
            if depth_component.max() == 0.:
                continue
            if np.sum(depth_component > 0.) / np.sum(component) < 0.5:
                depth[component] = 0.
    
    sky_mask = (semantic_map == SKY)
    depth[sky_mask] = 0.

    eroded_depth_mask = scipy.ndimage.binary_erosion(depth > 0., structure=np.ones((3,3))).astype(depth.dtype)
    labeled, num_labels = scipy.ndimage.label(eroded_depth_mask)
    cc_size = scipy.ndimage.sum(eroded_depth_mask, labeled, range(num_labels+1))
    mask_size = cc_size < 50
    depth[mask_size[labeled]] = 0.

    return depth

def ordinal_labeling(depth, semantic_map):
    H, W = depth.shape
    ordinal_map = np.zeros_like(depth, dtype=np.uint8)
    bg_mask = np.isin(semantic_map, BG)
    bg_mask = np.logical_and(bg_mask, (depth > 0.))

    if np.any(np.logical_and(bg_mask, (depth > 0.))):
        labeled, num_labels = scipy.ndimage.label(bg_mask)
        cc_size = scipy.ndimage.sum(bg_mask, labeled, range(num_labels+1))
        mask_size = cc_size <= (H*W)*0.05
        bg_mask_cc = bg_mask.copy()
        bg_mask_cc[mask_size[labeled]] = False

        last_quartile_thres = np.percentile(depth[np.logical_and(bg_mask, (depth > 0.))], 75)
        bg_mask_rank = np.logical_and(bg_mask, (depth >= last_quartile_thres))

        bg_mask = np.logical_and(bg_mask_cc, bg_mask_rank)

        ordinal_map[bg_mask] = 2
    
    for label in np.unique(semantic_map):
        if label in BG:
            continue
        if label == SKY:
            continue
        mask = (semantic_map == label)
        labels_im, num_labels = scipy.ndimage.label(mask)
        for i in range(1, num_labels+1):
            component = (labels_im == i)
            if np.sum(component) < (H*W)*0.05:
                continue
            component = scipy.ndimage.binary_erosion(component, structure=np.ones((3,3))).astype(component.dtype)
            ordinal_map[component] = 1

    return ordinal_map

def filter_with_monocular_prior(depth, mono_depth, mono_mask, filter_by_gradient=False, depth_error_thres=0.2, gradient_error_thres=0.2):
    out_depth = depth.copy()
    valid_mask = (depth > 0) & (mono_depth > 0) & mono_mask
    if np.any(valid_mask):
        depth_median = np.median(depth[valid_mask])
        mono_median = np.median(mono_depth[valid_mask])
        scale = mono_median / (depth_median + 1e-6)
        depth *= scale

    mono_depth[~valid_mask] = 0.0
    error = np.abs(depth - mono_depth) / (depth + 1e-6)

    filter_mask = error > depth_error_thres

    out_depth[filter_mask] = 0.0

    if not filter_by_gradient:
        return out_depth

    depth_dx_mvs = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=5)
    depth_dy_mvs = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=5)
    depth_grad_mvs = np.sqrt(depth_dx_mvs**2 + depth_dy_mvs**2) / np.clip(depth, 1e-3, None)

    depth_dx_mono = cv2.Sobel(mono_depth, cv2.CV_64F, 1, 0, ksize=5)
    depth_dy_mono = cv2.Sobel(mono_depth, cv2.CV_64F, 0, 1, ksize=5)
    depth_grad_mono = np.sqrt(depth_dx_mono**2 + depth_dy_mono**2) / np.clip(mono_depth, 1e-3, None)
    grad_error = np.abs(depth_grad_mvs - depth_grad_mono)
    grad_filter_mask = grad_error > gradient_error_thres

    out_depth[grad_filter_mask] = 0.0

    return out_depth
