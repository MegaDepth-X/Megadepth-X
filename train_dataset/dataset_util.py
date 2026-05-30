import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import math
import random
import numpy as np
import networkx as nx
from PIL import Image, ImageFile, ImageOps, ExifTags
import PIL

from .geometry import closed_form_inverse_se3

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


def crop_image_depth_and_intrinsic_by_pp(
    image, depth_map, intrinsic, target_shape, track=None, filepath=None, strict=False, depth_sfm_map=None
):
    original_size = np.array(image.shape)
    intrinsic = np.copy(intrinsic)

    if original_size[0] < target_shape[0]:
        raise AssertionError(
            f"Width check failed: original width {original_size[0]} is less than target width {target_shape[0]}."
        )

    if original_size[1] < target_shape[1]:
        raise AssertionError(
            f"Height check failed: original height {original_size[1]} is less than target height {target_shape[1]}."
        )

    cx = intrinsic[1, 2]
    cy = intrinsic[0, 2]

    if strict:
        half_x = min((target_shape[0] / 2), cx)
        half_y = min((target_shape[1] / 2), cy)
    else:
        half_x = min((target_shape[0] / 2), cx, original_size[0] - cx)
        half_y = min((target_shape[1] / 2), cy, original_size[1] - cy)

    start_x = math.floor(cx) - math.floor(half_x)
    start_y = math.floor(cy) - math.floor(half_y)

    assert start_x >= 0
    assert start_y >= 0

    if strict:
        end_x = start_x + target_shape[0]
        end_y = start_y + target_shape[1]
    else:
        end_x = start_x + 2 * math.floor(half_x)
        end_y = start_y + 2 * math.floor(half_y)

    image = image[start_x:end_x, start_y:end_y, :]
    if depth_map is not None:
        depth_map = depth_map[start_x:end_x, start_y:end_y]
    if depth_sfm_map is not None:
        depth_sfm_map = depth_sfm_map[start_x:end_x, start_y:end_y]

    intrinsic[1, 2] = intrinsic[1, 2] - start_x
    intrinsic[0, 2] = intrinsic[0, 2] - start_y

    if track is not None:
        track[:, 1] = track[:, 1] - start_x
        track[:, 0] = track[:, 0] - start_y

    if strict:
        if (image.shape[:2] != target_shape).any():
            current_h, current_w = image.shape[:2]
            target_h, target_w = target_shape[0], target_shape[1]
            pad_h = target_h - current_h
            pad_w = target_w - current_w
            if pad_h < 0 or pad_w < 0:
                raise ValueError(
                    f"The cropped image is bigger than the target shape: cropped=({current_h},{current_w}), "
                    f"target=({target_h},{target_w})."
                )
            image = np.pad(
                image,
                pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )
            if depth_sfm_map is not None:
                depth_sfm_map = np.pad(
                    depth_sfm_map,
                    pad_width=((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0,
                )

    if depth_sfm_map is not None:
        if depth_map is not None:
            assert image.shape[:2] == depth_map.shape[:2] == depth_sfm_map.shape[:2]
        return image, depth_map, intrinsic, track, depth_sfm_map

    return image, depth_map, intrinsic, track


def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    original_size,
    track=None,
    pixel_center=True,
    safe_bound=4,
    rescale_aug=True,
    depth_sfm_map=None,
):
    if rescale_aug:
        random_boundary = np.random.triangular(0, 0, 0.3)
        safe_bound = safe_bound + random_boundary * target_shape.max()

    resize_scales = (target_shape + safe_bound) / original_size
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    image = Image.fromarray(image)
    input_resolution = np.array(image.size)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = image.resize(tuple(output_resolution), resample=lanczos if max_resize_scale < 1 else bicubic)
    image = np.array(image)

    if depth_map is not None:
        depth_map = cv2.resize(
            depth_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )
    if depth_sfm_map is not None:
        depth_sfm_map = cv2.resize(
            depth_sfm_map,
            output_resolution,
            fx=max_resize_scale,
            fy=max_resize_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    actual_size = np.array(image.shape[:2])
    actual_resize_scale = np.max(actual_size / original_size)

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5

    intrinsic[:2, :] = intrinsic[:2, :] * actual_resize_scale

    if track is not None:
        track = track * actual_resize_scale

    if pixel_center:
        intrinsic[0, 2] = intrinsic[0, 2] - 0.5
        intrinsic[1, 2] = intrinsic[1, 2] - 0.5

    if depth_map is not None:
        assert image.shape[:2] == depth_map.shape[:2]
    if depth_sfm_map is not None:
        assert image.shape[:2] == depth_sfm_map.shape[:2]
        return image, depth_map, intrinsic, track, depth_sfm_map

    return image, depth_map, intrinsic, track


def threshold_depth_map(depth_map, max_percentile=99, min_percentile=1, max_depth=-1):
    if depth_map is None:
        return None

    depth_map = depth_map.astype(float, copy=True)

    if max_depth > 0:
        depth_map[depth_map > max_depth] = 0.0

    depth_max_thres = np.nanpercentile(depth_map, max_percentile) if max_percentile > 0 else None
    depth_min_thres = np.nanpercentile(depth_map, min_percentile) if min_percentile > 0 else None

    if depth_max_thres is not None and depth_max_thres > 0:
        depth_map[depth_map > depth_max_thres] = 0.0
    if depth_min_thres is not None and depth_min_thres > 0:
        depth_map[depth_map < depth_min_thres] = 0.0

    return depth_map


def depth_to_world_coords_points(depth_map, extrinsic, intrinsic, eps=1e-8, half_offset=False):
    if depth_map is None:
        return None, None, None

    point_mask = depth_map > eps
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic, half_offset=half_offset)

    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map, intrinsic, half_offset=False):
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    if half_offset:
        u = u + 0.5
        v = v + 0.5

    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def rotate_90_degrees(image, depth_map, extri_opencv, intri_opencv, clockwise=True, track=None, depth_sfm_map=None):
    image_height, image_width = image.shape[:2]

    rotated_image, rotated_depth_map, rotated_depth_sfm_map = rotate_image_and_depth_rot90(
        image, depth_map, clockwise, depth_sfm_map
    )
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)

    new_track = adjust_track_rot90(track, image_width, image_height, clockwise) if track is not None else None
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    if depth_sfm_map is not None:
        return (
            rotated_image,
            rotated_depth_map,
            new_extri_opencv,
            new_intri_opencv,
            new_track,
            rotated_depth_sfm_map,
        )

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
        new_track,
    )


def rotate_image_and_depth_rot90(image, depth_map, clockwise, depth_sfm_map=None):
    rotated_depth_map = None
    rotated_depth_sfm_map = None

    if clockwise:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=1)
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=1)
        if depth_sfm_map is not None:
            rotated_depth_sfm_map = np.transpose(depth_sfm_map, (1, 0))
            rotated_depth_sfm_map = np.flip(rotated_depth_sfm_map, axis=1)
    else:
        rotated_image = np.transpose(image, (1, 0, 2))
        rotated_image = np.flip(rotated_image, axis=0)
        if depth_map is not None:
            rotated_depth_map = np.transpose(depth_map, (1, 0))
            rotated_depth_map = np.flip(rotated_depth_map, axis=0)
        if depth_sfm_map is not None:
            rotated_depth_sfm_map = np.transpose(depth_sfm_map, (1, 0))
            rotated_depth_sfm_map = np.flip(rotated_depth_sfm_map, axis=0)

    rotated_image = np.copy(rotated_image)
    rotated_depth_map = np.copy(rotated_depth_map) if rotated_depth_map is not None else None
    rotated_depth_sfm_map = (
        np.copy(rotated_depth_sfm_map) if rotated_depth_sfm_map is not None else None
    )
    return rotated_image, rotated_depth_map, rotated_depth_sfm_map


def adjust_extrinsic_matrix_rot90(extri_opencv, clockwise):
    R = extri_opencv[:, :3]
    t = extri_opencv[:, 3]

    if clockwise:
        R_rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    else:
        R_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    new_R = np.dot(R_rotation, R)
    new_t = np.dot(R_rotation, t)
    new_extri_opencv = np.hstack((new_R, new_t.reshape(-1, 1)))
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise):
    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = np.eye(3)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    return new_intri_opencv


def adjust_track_rot90(track, image_width, image_height, clockwise):
    if track is None:
        return None

    if clockwise:
        return np.stack((track[:, 1], image_width - 1 - track[:, 0]), axis=-1)

    return np.stack((image_height - 1 - track[:, 1], track[:, 0]), axis=-1)


def rotate_image_pil(image, orientation):
    if orientation == 1:
        return image

    if orientation == 2:
        image = ImageOps.mirror(image)
    elif orientation == 3:
        image = image.rotate(180, expand=True)
    elif orientation == 4:
        image = ImageOps.flip(image)
    elif orientation == 5:
        image = ImageOps.mirror(image.rotate(-90, expand=True))
    elif orientation == 6:
        image = image.rotate(-90, expand=True)
    elif orientation == 7:
        image = ImageOps.mirror(image.rotate(90, expand=True))
    elif orientation == 8:
        image = image.rotate(90, expand=True)

    return image


def read_image_pil(img_path, orientation=1, random_orientation=False):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    image_exif = img.getexif()

    if orientation == 1:
        orientation = image_exif.get(ExifTags.Base.Orientation, 1)
    if orientation < 1 or orientation > 8:
        orientation = 1

    if random_orientation and random.random() < 0.2:
        if orientation == 1:
            orientation = random.choice([6, 8])
        elif orientation == 3:
            orientation = random.choice([6, 8])
        elif orientation == 6:
            orientation = random.choice([1, 3])
        elif orientation == 8:
            orientation = random.choice([1, 3])

    img = rotate_image_pil(img, orientation)
    return np.array(img), orientation, w, h


def rotate_depth(depth, orientation=1):
    if orientation == 1:
        return depth

    if orientation == 2:
        depth = np.fliplr(depth)
    elif orientation == 3:
        depth = np.rot90(depth, 2)
    elif orientation == 4:
        depth = np.flipud(depth)
    elif orientation == 5:
        depth = np.rot90(np.fliplr(depth), -1)
    elif orientation == 6:
        depth = np.rot90(depth, -1)
    elif orientation == 7:
        depth = np.rot90(np.fliplr(depth), 1)
    elif orientation == 8:
        depth = np.rot90(depth, 1)

    return depth


def get_exif_rotation_matrix_cv(orientation):
    mapping = {
        1: np.eye(3),
        3: np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        6: np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        8: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    }
    R = mapping.get(orientation, np.eye(3))
    H = np.eye(4)
    H[:3, :3] = R
    return H


def rotate_intrinsics(K, width, height, orientation=1):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    W0, H0 = int(width), int(height)

    if orientation == 1:
        pass
    elif orientation == 2:
        cx = W0 - cx
    elif orientation == 3:
        cx = W0 - cx
        cy = H0 - cy
    elif orientation == 4:
        cy = H0 - cy
    elif orientation == 5:
        fx, fy = fy, fx
        cx, cy = cy, cx
    elif orientation == 6:
        fx, fy = fy, fx
        cx, cy = H0 - cy, cx
    elif orientation == 7:
        fx, fy = fy, fx
        cx, cy = H0 - cy, W0 - cx
    elif orientation == 8:
        fx, fy = fy, fx
        cx, cy = cy, W0 - cx
    else:
        raise ValueError(f"Unsupported orientation: {orientation}")

    K_new = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    return K_new


def random_dfs_with_node_num_limit(G, max_nodes, start=None):
    if start is None:
        start = random.choice(list(G.nodes()))
    visited = set()
    traversal = []

    def dfs(node):
        if len(traversal) == max_nodes:
            return
        visited.add(node)
        traversal.append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor)
        return

    dfs(start)
    return traversal


def random_dfs_with_node_num_limit_multi_cc(G, max_nodes, start=None):
    ccs = list(nx.connected_components(G))
    ccs = sorted(ccs, key=lambda cc: len(cc), reverse=True)
    num_cc = len(ccs)

    if max_nodes < 2 * num_cc:
        num_cc = max_nodes // 2
        ccs = ccs[:num_cc]

    allocation = [2] * num_cc
    remaining = max_nodes - 2 * num_cc
    cc_sizes = [len(cc) for cc in ccs]
    total_extra = sum(max(s - 2, 0) for s in cc_sizes)

    if total_extra > 0 and remaining > 0:
        for i in range(num_cc):
            extra = max(cc_sizes[i] - 2, 0)
            share = int(remaining * extra / total_extra)
            allocation[i] += min(share, cc_sizes[i] - 2)

        allocated_so_far = sum(allocation)
        leftover = max_nodes - allocated_so_far
        indices = list(range(num_cc))
        random.shuffle(indices)
        for i in indices:
            if leftover <= 0:
                break
            can_add = cc_sizes[i] - allocation[i]
            add = min(can_add, leftover)
            allocation[i] += add
            leftover -= add

    traversal = []
    for i, cc in enumerate(ccs):
        subgraph = G.subgraph(cc)
        nodes_for_cc = allocation[i]
        result = random_dfs_with_node_num_limit(subgraph, nodes_for_cc, start=start)
        traversal.extend(result)
        start = None

    assert len(traversal) == max_nodes
    return traversal
