import torchvision.transforms.functional as TF
import numpy as np
import os
import torch
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import block_reduce
from qwen_vl_utils import process_vision_info
from io import BytesIO
import base64

def encode_base64(image):
    """
    Encodes a PIL image to a base64 string.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):

    """
    Prepare the input for Qwen2.5VL.
    """

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    return inputs

def high_pass_filter(image, resolusion, km=7, kh=3, reduce=True):
    """
    Applies a high-pass filter to an image to highlight edges and fine details.
    
    This function resizes the image, applies a Gaussian blur to create a low-frequency version,
    subtracts it from the original to get high-frequency components, and then applies median filtering.
    
    Args:
        image: Input PIL image
        resolusion: Target resolution to resize the image to
        km: Kernel size for median filtering (default: 7)
        kh: Kernel size for Gaussian blur (default: 3)
        reduce: Whether to reduce the output size using block reduction (default: True)
        
    Returns:
        h_brightness: A 2D numpy array representing the high-frequency components of the image
    """

    image = TF.resize(image, (resolusion, resolusion))
    image = TF.to_tensor(image).unsqueeze(0)
    l = TF.gaussian_blur(image, kernel_size=(kh, kh)).squeeze().detach().cpu().numpy()
    h = image.squeeze().detach().cpu().numpy() - l
    h_brightness = np.sqrt(np.square(h).sum(axis=0))
    h_brightness = median_filter(h_brightness, size=km)
    if reduce:
        h_brightness = block_reduce(h_brightness, block_size=(14, 14), func=np.sum)

    return h_brightness

def safe_divide(numerator, denominator, eps=1e-6):
    """
    Divide two arrays while guarding against unstable small denominators.
    """

    if isinstance(numerator, torch.Tensor) or isinstance(denominator, torch.Tensor):
        numerator = numerator.to(torch.float32)
        denominator = denominator.to(torch.float32)
        safe_denominator = torch.where(
            denominator.abs() < eps,
            torch.full_like(denominator, eps),
            denominator,
        )
        result = numerator / safe_denominator
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    safe_denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    result = numerator / safe_denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

def importance_edge_mask(shape, border_ratio=0.1, min_edge_weight=0.55):
    """
    Build a soft edge-suppression mask to reduce common border artifacts.
    """

    height, width = shape
    border = max(1, int(round(min(height, width) * border_ratio)))

    y_dist = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    x_dist = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)

    y_weight = np.clip(y_dist / border, 0.0, 1.0)
    x_weight = np.clip(x_dist / border, 0.0, 1.0)

    mask = np.minimum.outer(y_weight, x_weight)
    return min_edge_weight + (1.0 - min_edge_weight) * mask

def postprocess_importance_map(att_map, smooth_sigma=1.0, clip_percentiles=(5, 99)):
    """
    Stabilize a raw importance map before box search.
    """

    importance_map = np.asarray(att_map, dtype=np.float32).squeeze()
    importance_map = np.nan_to_num(importance_map, nan=0.0, posinf=0.0, neginf=0.0)

    if importance_map.ndim != 2:
        raise ValueError(f"Expected a 2D importance map, got shape {importance_map.shape}")

    importance_map = importance_map - importance_map.min()

    low, high = np.percentile(importance_map, clip_percentiles)
    if high > low:
        importance_map = np.clip(importance_map - low, 0.0, high - low)

    if smooth_sigma > 0:
        importance_map = gaussian_filter(importance_map, sigma=smooth_sigma)

    importance_map = importance_map * importance_edge_mask(importance_map.shape)
    importance_map = np.clip(importance_map, 0.0, None)

    max_value = float(importance_map.max())
    if max_value > 0:
        importance_map = importance_map / max_value

    return importance_map.astype(np.float32)

def importance_map_summary(importance_map):
    """
    Compute lightweight confidence statistics for a normalized importance map.
    """

    flat = np.asarray(importance_map, dtype=np.float32).reshape(-1)
    total_mass = float(flat.sum())

    if flat.size == 0 or total_mass <= 0:
        return {
            "confidence": 0.0,
            "std": 0.0,
            "peak_value": 0.0,
            "top_mass": 0.0,
            "contrast": 0.0,
        }

    top_k = max(1, flat.size // 20)
    top_indices = np.argpartition(flat, flat.size - top_k)[-top_k:]
    top_mass = float(flat[top_indices].sum() / max(total_mass, 1e-6))
    std = float(flat.std())
    peak_value = float(flat.max())
    contrast = float(max(peak_value - flat.mean(), 0.0))

    confidence = (
        0.45 * np.clip(std / 0.18, 0.0, 1.0)
        + 0.35 * np.clip(top_mass / 0.30, 0.0, 1.0)
        + 0.20 * np.clip(contrast / 0.60, 0.0, 1.0)
    )

    return {
        "confidence": float(confidence),
        "std": std,
        "peak_value": peak_value,
        "top_mass": top_mass,
        "contrast": contrast,
    }

def candidate_window_score(importance_map, x, y, width, height):
    """
    Score a candidate crop window using mass, local contrast, and compactness.
    """

    window = importance_map[y:y+height, x:x+width]
    window_sum = float(window.sum())
    density = float(window_sum / max(window.size, 1))
    peak = float(window.max())

    ring_x1 = max(0, x - 1)
    ring_y1 = max(0, y - 1)
    ring_x2 = min(importance_map.shape[1], x + width + 1)
    ring_y2 = min(importance_map.shape[0], y + height + 1)

    ring = importance_map[ring_y1:ring_y2, ring_x1:ring_x2]
    ring_area = ring.size - window.size
    ring_sum = float(ring.sum() - window_sum)
    ring_mean = ring_sum / max(ring_area, 1)

    total_mass = float(importance_map.sum())
    mass_ratio = window_sum / max(total_mass, 1e-6)
    coverage = float(window.size / importance_map.size)
    local_contrast = max(density - ring_mean, 0.0)

    score = 0.55 * mass_ratio + 0.30 * local_contrast + 0.15 * peak - 0.10 * coverage

    return {
        "score": float(score),
        "mass_ratio": float(mass_ratio),
        "density": density,
        "local_contrast": float(local_contrast),
        "peak": peak,
        "coverage": coverage,
    }

def crop_box_from_grid(x, y, width, height, block_size, image_size, context_padding=0.08):
    """
    Convert a window on the importance-map grid into image pixel coordinates.
    """

    selected_width = width * block_size[0]
    selected_height = height * block_size[1]

    selected_width = min(selected_width * (1.0 + context_padding), image_size[0])
    selected_height = min(selected_height * (1.0 + context_padding), image_size[1])

    x_center = (x + width / 2) * block_size[0]
    y_center = (y + height / 2) * block_size[1]

    x_center = min(max(x_center, selected_width / 2), image_size[0] - selected_width / 2)
    y_center = min(max(y_center, selected_height / 2), image_size[1] - selected_height / 2)

    x1 = max(0, int(round(x_center - selected_width / 2)))
    y1 = max(0, int(round(y_center - selected_height / 2)))
    x2 = min(image_size[0], int(round(x_center + selected_width / 2)))
    y2 = min(image_size[1], int(round(y_center + selected_height / 2)))

    return x1, y1, x2, y2

def bbox_from_att_image_adaptive(att_map, image_size, bbox_size=336, return_metadata=False):
    """
    Generate a confidence-aware crop box from an importance map.
     
    The search uses stabilized importance maps, multiple crop scales, a small
    family of aspect ratios, and a confidence gate so weak maps fall back to
    the full image rather than forcing a brittle crop.
     
    Args:
        att_map: A 2D numpy array representing the raw importance map
        image_size: Tuple of (width, height) of the original image
        bbox_size: Base size for the bounding box (default: 336)
        return_metadata: Whether to return crop metadata alongside the box
         
    Returns:
        tuple: (x1, y1, x2, y2) coordinates of the bounding box in the original image
    """

    full_bbox = (0, 0, image_size[0], image_size[1])
    processed_map = postprocess_importance_map(att_map)
    map_stats = importance_map_summary(processed_map)

    metadata = {
        "did_crop": False,
        "reason": "low_confidence",
        "map_confidence": float(map_stats["confidence"]),
        "map_std": float(map_stats["std"]),
        "map_top_mass": float(map_stats["top_mass"]),
        "map_peak_value": float(map_stats["peak_value"]),
        "aspect_ratio": 1.0,
        "scale": 1.0,
        "window_score": 0.0,
        "grid_box": [0, 0, processed_map.shape[1], processed_map.shape[0]],
    }

    block_size = image_size[0] / processed_map.shape[1], image_size[1] / processed_map.shape[0]

    if map_stats["confidence"] < 0.30 or processed_map.sum() <= 0:
        if return_metadata:
            return full_bbox, metadata
        return full_bbox

    scales = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    aspect_ratios = [1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16]

    best_candidate = None
    seen_sizes = set()

    for scale in scales:
        for aspect_ratio in aspect_ratios:
            target_width = bbox_size * scale * np.sqrt(aspect_ratio)
            target_height = bbox_size * scale / np.sqrt(aspect_ratio)

            window_width = min(max(int(round(target_width / block_size[0])), 1), processed_map.shape[1])
            window_height = min(max(int(round(target_height / block_size[1])), 1), processed_map.shape[0])

            if (window_width, window_height) in seen_sizes:
                continue
            seen_sizes.add((window_width, window_height))

            for y in range(processed_map.shape[0] - window_height + 1):
                for x in range(processed_map.shape[1] - window_width + 1):
                    candidate = candidate_window_score(processed_map, x, y, window_width, window_height)
                    candidate["x"] = x
                    candidate["y"] = y
                    candidate["window_width"] = window_width
                    candidate["window_height"] = window_height
                    candidate["aspect_ratio"] = float(aspect_ratio)
                    candidate["scale"] = float(scale)

                    if best_candidate is None or candidate["score"] > best_candidate["score"]:
                        best_candidate = candidate

    if best_candidate is None:
        if return_metadata:
            return full_bbox, metadata
        return full_bbox

    bbox = crop_box_from_grid(
        best_candidate["x"],
        best_candidate["y"],
        best_candidate["window_width"],
        best_candidate["window_height"],
        block_size,
        image_size,
    )

    crop_width = bbox[2] - bbox[0]
    crop_height = bbox[3] - bbox[1]
    nearly_full = (
        crop_width >= 0.95 * image_size[0]
        and crop_height >= 0.95 * image_size[1]
    )

    metadata.update(
        {
            "did_crop": not nearly_full,
            "reason": "selected_window" if not nearly_full else "full_image_window",
            "aspect_ratio": float(best_candidate["aspect_ratio"]),
            "scale": float(best_candidate["scale"]),
            "window_score": float(best_candidate["score"]),
            "window_density": float(best_candidate["density"]),
            "window_mass_ratio": float(best_candidate["mass_ratio"]),
            "window_local_contrast": float(best_candidate["local_contrast"]),
            "grid_box": [
                int(best_candidate["x"]),
                int(best_candidate["y"]),
                int(best_candidate["window_width"]),
                int(best_candidate["window_height"]),
            ],
        }
    )

    if nearly_full:
        bbox = full_bbox

    if return_metadata:
        return bbox, metadata
    return bbox

def high_res_split_threshold(image, res_threshold=1024):
    """
    Splits a high-resolution image into smaller patches.
    
    This function divides a large image into smaller patches to process them individually,
    which is useful for handling high-resolution images that might be too large for direct processing.
    
    Args:
        image: Input PIL image
        res_threshold: Maximum resolution threshold before splitting (default: 1024)
        
    Returns:
        tuple: (split_images, vertical_split, horizontal_split)
            - split_images: List of PIL image patches
            - vertical_split: Number of vertical splits
            - horizontal_split: Number of horizontal splits
    """

    vertical_split = int(np.ceil(image.size[1] / res_threshold))
    horizontal_split = int(vertical_split * image.size[0] / image.size[1])

    split_num = (horizontal_split, vertical_split)
    split_size = int(np.ceil(image.size[0] / split_num[0])), int(np.ceil(image.size[1] / split_num[1]))
    
    split_images = []
    for j in range(split_num[1]):
        for i in range(split_num[0]):
            split_image = image.crop((i*split_size[0], j*split_size[1], (i+1)*split_size[0], (j+1)*split_size[1]))
            split_images.append(split_image)
    
    return split_images, vertical_split, horizontal_split

def high_res(map_func, image, prompt, general_prompt, model, processor):
    """
    Applies an attention mapping function to high-resolution images by splitting and recombining.
    
    This function splits a high-resolution image into smaller patches, applies the specified
    attention mapping function to each patch, and then recombines the results into a single
    attention map.
    
    Args:
        map_func: The attention mapping function to apply to each patch
        image: Input PIL image
        prompt: Text prompt for the attention function
        general_prompt: General text prompt for baseline comparison
        model: Model instance (LLaVA or BLIP)
        processor: Processor for the corresponding model
        
    Returns:
        block_att: A 2D numpy array representing the combined attention map for the entire image
    """

    split_images, num_vertical_split, num_horizontal_split = high_res_split_threshold(image)
    att_maps = []
    for split_image in split_images:
        att_map = map_func(split_image, prompt, general_prompt, model, processor)
        # att_map = att_map / att_map.mean()
        att_maps.append(att_map)
    block_att = np.block([att_maps[j:j+num_horizontal_split] for j in range(0, num_horizontal_split * num_vertical_split, num_horizontal_split)])

    return block_att
