import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .dataset_util import (
    crop_image_depth_and_intrinsic_by_pp,
    resize_image_depth_and_intrinsic,
    rotate_90_degrees,
    depth_to_world_coords_points,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    def __init__(self, common_conf):
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check
        self.training = common_conf.training

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        raise NotImplementedError("Subclasses must implement get_data().")

    def get_target_shape(self, aspect_ratio):
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size
        return np.array([short_size, self.img_size])

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
        half_offset=False,
    ):
        image = np.copy(image)
        depth_map = np.copy(depth_map) if depth_map is not None else None
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)
        if depth_sfm_map is not None:
            depth_sfm_map = np.copy(depth_sfm_map)

        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        if depth_sfm_map is not None:
            image, depth_map, intri_opencv, track, depth_sfm_map = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
                depth_sfm_map=depth_sfm_map,
            )
        else:
            image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
            )

        original_size = np.array(image.shape[:2])
        target_shape = target_image_shape

        rotate_to_portrait = False
        if self.landscape_check:
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        if self.rescale:
            if depth_sfm_map is not None:
                image, depth_map, intri_opencv, track, depth_sfm_map = resize_image_depth_and_intrinsic(
                    image, depth_map, intri_opencv, target_shape, original_size, track=track,
                    safe_bound=safe_bound,
                    rescale_aug=self.rescale_aug,
                    depth_sfm_map=depth_sfm_map,
                )
            else:
                image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                    image, depth_map, intri_opencv, target_shape, original_size, track=track,
                    safe_bound=safe_bound,
                    rescale_aug=self.rescale_aug
                )

        if depth_sfm_map is not None:
            image, depth_map, intri_opencv, track, depth_sfm_map = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath,
                depth_sfm_map=depth_sfm_map,
            )
        else:
            image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
                image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
            )

        if rotate_to_portrait:
            clockwise = np.random.rand() > 0.5
            if depth_sfm_map is not None:
                image, depth_map, extri_opencv, intri_opencv, track, depth_sfm_map = rotate_90_degrees(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    clockwise=clockwise,
                    track=track,
                    depth_sfm_map=depth_sfm_map,
                )
            else:
                image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                    image,
                    depth_map,
                    extri_opencv,
                    intri_opencv,
                    clockwise=clockwise,
                    track=track,
                )

        world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
            depth_map, extri_opencv, intri_opencv, half_offset=half_offset
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
