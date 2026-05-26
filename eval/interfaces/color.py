import numpy as np
import cv2

def masked_color_normalization(images, masks, blend_sigma=30, blend_strength=0.8):
    """
    Normalize images based on masked regions, with soft blending to preserve background.

    Args:
        images (list[np.ndarray]): List of RGB images (H, W, 3), dtype uint8 or float32.
        masks  (list[np.ndarray]): List of binary masks (H, W).
        blend_sigma (float): Gaussian blur kernel for soft transitions.
        blend_strength (float): 0–1, how strongly to apply normalization inside mask.

    Returns:
        normalized_images (list[np.ndarray]): List of softly normalized RGB images.
    """
    assert len(images) == len(masks), "Number of images and masks must match."

    imgs_f32 = [img.astype(np.float32) for img in images]
    imgs_lab = [cv2.cvtColor(img, cv2.COLOR_RGB2LAB) for img in imgs_f32]

    means, stds = [], []
    for img_lab, mask in zip(imgs_lab, masks):
        mask = mask > 0
        vals = img_lab[mask].reshape(-1, 3)
        mean = vals.mean(axis=0)
        std = vals.std(axis=0) + 1e-6
        means.append(mean)
        stds.append(std)

    ref_mean = np.mean(np.stack(means), axis=0)
    ref_std = np.mean(np.stack(stds), axis=0)

    normalized_images = []
    for idx, (img_lab, mask, mean_src, std_src) in enumerate(zip(imgs_lab, masks, means, stds)):
        # Color normalization
        normalized = (img_lab - mean_src) * (ref_std / std_src) + ref_mean
        normalized[..., 0] = np.clip(normalized[..., 0], 0, 100)
        normalized[..., 1] = np.clip(normalized[..., 1], -128, 127)
        normalized[..., 2] = np.clip(normalized[..., 2], -128, 127)

        # Convert back to RGB
        norm_rgb = cv2.cvtColor(normalized.astype(np.float32), cv2.COLOR_LAB2RGB)
        norm_rgb = np.clip(norm_rgb, 0, 1)

        # Create soft blend mask (Gaussian blur + blend strength)
        # mask_f = mask.astype(np.float32)
        # mask_blur = cv2.GaussianBlur(mask_f, (0, 0), blend_sigma)
        # mask_blur = (mask_blur / mask_blur.max()) * blend_strength
        # # Blend normalized and original image
        # blended = norm_rgb * mask_blur[..., None] + imgs_f32[idx] * (1 - mask_blur[..., None])
        # blended = np.clip(blended, 0, 1)
        blended = norm_rgb
        normalized_images.append(blended)

    return np.stack(normalized_images, axis=0)
