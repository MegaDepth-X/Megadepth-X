# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Iterable
import json
import os.path as osp
import os
import logging

import cv2
import random
import numpy as np
import networkx as nx

import torch
from torch.utils.data import Dataset
import torchvision.transforms as tvf

import time
from .dataset_util import *

to_tensor = tvf.ToTensor()

class MegaDepth_X(Dataset):
    def __init__(
        self,
        split: str = "test",
        MegaDepth_X_DIR: str = None,
        sample_DIR: str = None,
        load_img_size: int = 518,
        min_num_images: int = 24,
        sample_type: str = "sparse_only",
        sample_num = 5,
        limit_id_list = None,
    ):
        super().__init__()

        self.debug = False
        self.training = False
        self.split = split
        self.load_img_size = load_img_size
        self.sample_type = sample_type
        self.sample_num = sample_num
        self.MegaDepth_X_DIR = MegaDepth_X_DIR
        self.sample_DIR = sample_DIR

        if MegaDepth_X_DIR is None:
            raise ValueError("MegaDepth_X_DIR must be specified.")
        if sample_DIR is None:
            raise ValueError("sample_DIR must be specified.")
        self.invalid_sequence = [] # set any invalid sequence names here

        if limit_id_list is not None:
            with open(limit_id_list, 'r') as f:
                limit_ids = json.load(f)
            logging.info(f"Using limited id list with {len(limit_ids)} sequences.")
        
        scene_recon_list = []
        scene_recon_list_raw = list(json.load(open(osp.join(sample_DIR, f"{split}_recons.json"), 'r')))
        cleaned_scene_recon_list = []
        for scene_recon_name in scene_recon_list_raw:
            scene, recon = scene_recon_name.split(">")
            if os.path.exists(os.path.join(sample_DIR, scene, recon, f"{sample_type}.npz")) and scene_recon_name not in self.invalid_sequence:
                cleaned_scene_recon_list.append(scene_recon_name)
        for x in cleaned_scene_recon_list:
            for i in range(self.sample_num):
                if limit_id_list is not None:
                    if f"{x}>{i}" not in limit_ids:
                        continue
                scene_recon_list.append(f"{x}>{i}")

        logging.info(f"scene_recon_list: {scene_recon_list}")
        logging.info(f"scene_recon_list.len: {len(scene_recon_list)}")
        self.data_store = {}
        self.seqlen = None
        self.min_num_images = min_num_images

        for scene_recon_name_sample_idx in scene_recon_list:
            scene_recon_name, sample_idx = scene_recon_name_sample_idx.rsplit(">", 1)
            scene, recon = scene_recon_name.split(">")
            scene_recon_file = os.path.join(sample_DIR, scene, recon, f"{sample_type}.npz")
            sample_list = list(np.load(scene_recon_file, allow_pickle=True)['sampled_cases'])
            
            true_sample_list = []
            for sample in sample_list:
                if len(sample["imgs"].keys()) >= min_num_images:
                    true_sample_list.append(sample)
            self.data_store[scene_recon_name] = true_sample_list

        self.sequence_list = scene_recon_list

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MegaScenes Data dataset length: {len(self)}")

    def __len__(self):
        return len(self.sequence_list)

    def get_seq_framenum(self, index: Optional[int] = None, sequence_name: Optional[str] = None):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        sequence_name, sample_idx = sequence_name.rsplit(">", 1)
        return len(self.data_store[sequence_name][int(sample_idx)]["imgs"])
    
    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
        depth_sfm_map=None,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)
        if depth_sfm_map is not None:
            depth_sfm_map = np.copy(depth_sfm_map)
        aug_size = original_size

        if depth_sfm_map is not None:
            # Move principal point to the image center and crop if necessary
            image, depth_map, intri_opencv, track, depth_sfm_map = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
                depth_sfm_map=depth_sfm_map,
            )
        else:
            # Move principal point to the image center and crop if necessary
            image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
            )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape

        # Resize images and update intrinsics
        if depth_sfm_map is not None:
            image, depth_map, intri_opencv, track, depth_sfm_map = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=False,
                depth_sfm_map=depth_sfm_map,
            )
        else:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=False
            )

        # Ensure final crop to target shape
        if depth_sfm_map is not None:
            image, depth_map, intri_opencv, track, depth_sfm_map = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath,
                depth_sfm_map=depth_sfm_map,
            )
        else:
            image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
            )

        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        if depth_sfm_map is not None:
            return (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    track,
                    depth_sfm_map,
                )
        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def get_data(
            self,
            index: Optional[int] = None,
            sequence_name: Optional[str] = None,
            ids: Union[Iterable, None] = None,
        ):
        if sequence_name is None:
            if index is None:
                raise ValueError("Please specify either index or sequence_name")
            sequence_name = self.sequence_list[index]
        sequence_name, sample_idx = sequence_name.rsplit(">", 1)
        sample_idx = int(sample_idx)
        if len(self.data_store[sequence_name]) < sample_idx + 1:
            sample = self.data_store[sequence_name][0]
        else:
            # print(sequence_name, sample_idx)
            sample = self.data_store[sequence_name][sample_idx]
        if ids is None:
            ids = np.arange(len(sample["imgs"]))
        img_per_seq = len(sample["imgs"]) if ids is None else len(ids)


        selected_path = list(sample["imgs"].keys())
        
        target_image_shape = np.array([self.load_img_size, self.load_img_size])

        images = []
        depths = []
        sfm_depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        depth_paths = []
        original_sizes = []

        edge_mask = np.eye(img_per_seq, dtype=np.uint8)

        for current_idx, node in enumerate(selected_path):
            scene_name, recon = sequence_name.split(">")
            image_path = osp.join(self.MegaDepth_X_DIR, "depth_raw", scene_name, recon, "images_w_exif", node)
            image, orientation, w_o, h_o = read_image_pil(image_path)

            depth_path = osp.join(self.MegaDepth_X_DIR, "depth", scene_name, recon, "depth", sample["imgs"][node]["depth_path"])
            if not os.path.exists(depth_path):
                if ".npy" in depth_path:
                    depth_path = depth_path.replace(".npy", ".npz")
                    if os.path.exists(depth_path):
                        depth_map = np.load(depth_path)["depth"]
                        depth_map = rotate_depth(depth_map, orientation)
                    else:
                        logging.warning(f"[MegaDepth-X] Depth file not found: {depth_path}, using zeros.")
                        # Fallback: read image to get shape, then create zero depth
                        depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                else:
                    logging.warning(f"[MegaDepth-X] Depth file not found: {depth_path}, using zeros.")
                    depth_map = np.zeros(image.shape[:2], dtype=np.float32)
            else:
                depth_map = np.load(depth_path)
                depth_map = rotate_depth(depth_map, orientation)

            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=98
            )
            depth_sfm_map = np.zeros_like(depth_map)

            # original_size = np.array(image.shape[:2])
            extri_opencv = np.array(sample["imgs"][node]["c2w"]) @ get_exif_rotation_matrix_cv(orientation)
            extri_opencv = closed_form_inverse_se3(extri_opencv[None, ...])[0][:3, :4] # convert c2w to w2c
            intri_opencv = rotate_intrinsics(np.array(sample["imgs"][node]["K"]), w_o, h_o, orientation)
            intri_opencv[0, 2] = intri_opencv[0, 2] - 0.5
            intri_opencv[1, 2] = intri_opencv[1, 2] - 0.5 # colmap to opencv convention

            # pad images, depth based on longest side and change intrinsics accordingly
            longest_side = max(image.shape[:2])
            pad_h = (longest_side - image.shape[0]) // 2
            pad_w = (longest_side - image.shape[1]) // 2

            # print(f"Megascenes_dgpp, Before padding: image shape: {image.shape}.{image.dtype}, depth shape: {None if depth_map is None else depth_map.shape}, intri: {intri_opencv}")

            image = np.pad(
                image,
                ((pad_h, longest_side - image.shape[0] - pad_h),
                 (pad_w, longest_side - image.shape[1] - pad_w),
                 (0, 0)),
                mode='constant',
                constant_values=255
            )

            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    ((pad_h, longest_side - depth_map.shape[0] - pad_h),
                     (pad_w, longest_side - depth_map.shape[1] - pad_w)),
                    mode='constant',
                    constant_values=0
                )

            if depth_sfm_map is not None:
                depth_sfm_map = np.pad(
                    depth_sfm_map,
                    ((pad_h, longest_side - depth_sfm_map.shape[0] - pad_h),
                     (pad_w, longest_side - depth_sfm_map.shape[1] - pad_w)),
                    mode='constant',
                    constant_values=0
                )

            intri_opencv[0, 2] += pad_w
            intri_opencv[1, 2] += pad_h
            # print(f"Megascenes_dgpp, After padding: image shape: {image.shape}.{image.dtype}, depth shape: {None if depth_map is None else depth_map.shape}, intri: {intri_opencv}")


            # resize image, depth, intrinsics to self.img_size
            original_size = np.array(image.shape[:2])
            image, depth_map, intri_opencv, _, depth_sfm_map = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, np.array([self.load_img_size, self.load_img_size]), original_size, track=None,
                safe_bound=0, rescale_aug=False, depth_sfm_map=depth_sfm_map
            )
            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
                depth_sfm_map,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_path,
                depth_sfm_map=depth_sfm_map,
            )

            images.append(to_tensor(image))
            depths.append(depth_map)
            sfm_depths.append(depth_sfm_map)
            extrinsics.append(torch.tensor(extri_opencv))
            intrinsics.append(torch.tensor(intri_opencv))
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            depth_paths.append(depth_path)
            original_sizes.append(original_size)

        # print(f"edge_mask={edge_mask}")
        set_name = "MegaDepth-X"

        batch = {
            "seq_id": set_name + "_" + sequence_name,
            "seq_len": img_per_seq,
            "n": img_per_seq,
            "ids": torch.tensor(ids),
            "frame_num": len(extrinsics),
            "image_paths": image_paths,
            "images": torch.stack(images, dim=0),
            "images_processed": torch.stack(images, dim=0),
            "depth_paths": depth_paths,
            "depths": depths,
            "extrs": torch.stack(extrinsics),
            "intrs": torch.stack(intrinsics),
            "pointclouds": np.stack(world_points, axis=0),
            "valid_mask": np.stack(point_masks, axis=0),
            "original_sizes": original_sizes,
            "edge_mask": edge_mask,
            "sfm_depths": sfm_depths,
        }

        return batch
