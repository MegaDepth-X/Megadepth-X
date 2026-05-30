import json
import logging
import os
import random

import numpy as np
import networkx as nx

from .base_dataset import BaseDataset
from .dataset_util import (
    read_image_pil,
    rotate_depth,
    threshold_depth_map,
    get_exif_rotation_matrix_cv,
    rotate_intrinsics,
    resize_image_depth_and_intrinsic,
    random_dfs_with_node_num_limit_multi_cc,
)
from .geometry import closed_form_inverse_se3

DEFAULT_DATA_ROOT = ""
DEFAULT_SAMPLE_ROOT = ""


class MegaDepthXDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split="train",
        data_root=None,
        sample_root=None,
        max_num_images=24,
        sample_type="mix",
    ):
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random

        self.split = split
        self.sample_type = sample_type
        self.max_num_images = max_num_images

        self.data_root = data_root
        self.sample_root = sample_root

        if not os.path.isdir(self.data_root):
            raise ValueError(f"data_root not found: {self.data_root}")
        if not os.path.isdir(self.sample_root):
            raise ValueError(f"sample_root not found: {self.sample_root}")

        recons_path = os.path.join(self.sample_root, f"{split}_recons.json")
        if not os.path.isfile(recons_path):
            raise ValueError(f"recons file not found: {recons_path}")

        scene_recon_list = list(json.load(open(recons_path, "r")))
        cleaned_scene_recon_list = []

        for scene_recon_name in scene_recon_list:
            scene, recon = scene_recon_name.split(">")
            sample_file = os.path.join(
                self.sample_root, scene, recon, f"{max_num_images}_{sample_type}.npz"
            )
            if os.path.exists(sample_file):
                cleaned_scene_recon_list.append(scene_recon_name)

        scene_recon_list = cleaned_scene_recon_list

        if self.debug:
            scene_recon_list = scene_recon_list[:2]
            if split == "train":
                self.len_train = len(scene_recon_list) * 100
            else:
                self.len_train = len(scene_recon_list)
        else:
            self.len_train = len(scene_recon_list)

        self.invalid_sequence = []
        self.data_store = {}
        total_frame_num = 0

        for scene_recon_name in scene_recon_list:
            if scene_recon_name in self.invalid_sequence:
                continue
            scene, recon = scene_recon_name.split(">")
            sample_file = os.path.join(
                self.sample_root, scene, recon, f"{max_num_images}_{sample_type}.npz"
            )
            sample_list = list(np.load(sample_file, allow_pickle=True)["sampled_cases"])

            true_sample_list = []
            for sample in sample_list:
                if len(sample["imgs"].keys()) >= max_num_images:
                    total_frame_num += len(sample["imgs"].keys())
                    true_sample_list.append(sample)

            if true_sample_list:
                self.data_store[scene_recon_name] = true_sample_list

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        self.total_frame_num = total_frame_num

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: MegaDepth-X data size: {self.sequence_list_len}")
        logging.info(f"{status}: MegaDepth-X dataset length: {len(self)}")

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        aspect_ratio=1.0,
    ):
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        sample_list = self.data_store[seq_name]
        if not sample_list:
            raise ValueError(f"No samples found for sequence: {seq_name}")

        sample_index = random.randint(0, len(sample_list) - 1)
        if self.debug:
            sample_index = 0

        sample = sample_list[sample_index]
        num_nodes = len(sample["imgs"].keys())
        if img_per_seq is None:
            img_per_seq = min(num_nodes, self.max_num_images)
        else:
            img_per_seq = min(img_per_seq, num_nodes)

        G = nx.Graph()
        G.add_nodes_from(list(sample["imgs"].keys()))
        G.add_edges_from(list(sample["edges"]))

        selected_path = random_dfs_with_node_num_limit_multi_cc(G, img_per_seq)
        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        sfm_depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        edge_mask = np.eye(img_per_seq, dtype=np.uint8)

        for current_idx, node in enumerate(selected_path):
            scene_name, recon = seq_name.split(">")
            image_root = os.path.join(self.data_root, scene_name, recon, "images")
            depth_root = os.path.join(self.data_root, scene_name, recon, "depths")

            image_path = os.path.join(image_root, node)
            image, orientation, w_o, h_o = read_image_pil(
                image_path, random_orientation=self.training
            )

            if self.load_depth:
                depth_path = os.path.join(depth_root, sample["imgs"][node]["depth_path"])
                depth_map = None
                if os.path.exists(depth_path):
                    if depth_path.endswith(".npz"):
                        depth_map = np.load(depth_path)["depth"]
                    else:
                        depth_map = np.load(depth_path)
                else:
                    if depth_path.endswith(".npy"):
                        npz_path = depth_path[:-4] + ".npz"
                        if os.path.exists(npz_path):
                            depth_map = np.load(npz_path)["depth"]
                        else:
                            logging.warning(
                                f"[MegaDepth-X] Depth file not found: {depth_path}, using zeros."
                            )
                            depth_map = np.zeros(image.shape[:2], dtype=np.float32)
                    else:
                        logging.warning(
                            f"[MegaDepth-X] Depth file not found: {depth_path}, using zeros."
                        )
                        depth_map = np.zeros(image.shape[:2], dtype=np.float32)

                depth_map = rotate_depth(depth_map, orientation)
                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
                depth_sfm_map = np.zeros_like(depth_map)
            else:
                depth_map = None
                depth_sfm_map = None

            extri_opencv = np.array(sample["imgs"][node]["c2w"]) @ get_exif_rotation_matrix_cv(orientation)
            extri_opencv = closed_form_inverse_se3(extri_opencv[None, ...])[0][:3, :4]
            intri_opencv = rotate_intrinsics(np.array(sample["imgs"][node]["K"]), w_o, h_o, orientation)
            intri_opencv[0, 2] = intri_opencv[0, 2] - 0.5
            intri_opencv[1, 2] = intri_opencv[1, 2] - 0.5

            longest_side = max(image.shape[:2])
            pad_h = (longest_side - image.shape[0]) // 2
            pad_w = (longest_side - image.shape[1]) // 2

            image = np.pad(
                image,
                ((pad_h, longest_side - image.shape[0] - pad_h),
                 (pad_w, longest_side - image.shape[1] - pad_w),
                 (0, 0)),
                mode="constant",
                constant_values=255,
            )

            if depth_map is not None:
                depth_map = np.pad(
                    depth_map,
                    ((pad_h, longest_side - depth_map.shape[0] - pad_h),
                     (pad_w, longest_side - depth_map.shape[1] - pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            if depth_sfm_map is not None:
                depth_sfm_map = np.pad(
                    depth_sfm_map,
                    ((pad_h, longest_side - depth_sfm_map.shape[0] - pad_h),
                     (pad_w, longest_side - depth_sfm_map.shape[1] - pad_w)),
                    mode="constant",
                    constant_values=0,
                )

            intri_opencv[0, 2] += pad_w
            intri_opencv[1, 2] += pad_h

            original_size = np.array(image.shape[:2])
            if depth_sfm_map is not None:
                image, depth_map, intri_opencv, _, depth_sfm_map = resize_image_depth_and_intrinsic(
                    image,
                    depth_map,
                    intri_opencv,
                    np.array([self.img_size, self.img_size]),
                    original_size,
                    track=None,
                    safe_bound=0,
                    rescale_aug=False,
                    depth_sfm_map=depth_sfm_map,
                )
            else:
                image, depth_map, intri_opencv, _ = resize_image_depth_and_intrinsic(
                    image,
                    depth_map,
                    intri_opencv,
                    np.array([self.img_size, self.img_size]),
                    original_size,
                    track=None,
                    safe_bound=0,
                    rescale_aug=False,
                    depth_sfm_map=None,
                )
            original_size = np.array(image.shape[:2])

            if depth_sfm_map is not None:
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
            else:
                (
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    world_coords_points,
                    cam_coords_points,
                    point_mask,
                    _,
                ) = self.process_one_image(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    original_size,
                    target_image_shape,
                    filepath=image_path,
                    depth_sfm_map=None,
                )

            images.append(image)
            depths.append(depth_map)
            sfm_depths.append(depth_sfm_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

            neighbors = list(G.neighbors(node))
            for neighbor_idx, other in enumerate(selected_path):
                if other in neighbors:
                    edge_mask[current_idx, neighbor_idx] = 1

        batch = {
            "seq_name": "MegaDepth_X_" + seq_name + "_" + str(sample_index),
            "ids": np.array([0]),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "edge_mask": edge_mask,
            "sfm_depths": sfm_depths,
        }
        return batch
