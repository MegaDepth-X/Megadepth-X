# MegaDepth-X training dataset (train_dataset)

This repository contains a PyTorch dataset implementation for MegaDepth-X at `megadepth_x_dataset.py`. The dataset is suitable for training multi-view reconstruction and relative-pose models.

1) Specifying `data_root` and `sample_root`

- `data_root`: path to the extracted MegaDepth-X dataset (contains per-scene directories).
- `sample_root`: path to the sampling files (for example `train_samples/`), which should contain `train_recons.json` and per-scene `.npz` sample files.

Example usage:
```python
from train_dataset.megadepth_x_dataset import MegaDepthXDataset
from train_dataset.base_dataset import CommonConfig

common_conf = CommonConfig(
    debug=False,
    training=True,
    load_depth=True,
    inside_random=True,
)
dataset = MegaDepthXDataset(
    common_conf=common_conf,
    split="train",
    data_root="/path/to/MegaDepth-X/",
    sample_root="/path/to/train_samples/",
    max_num_images=24,
    sample_type="mix",
)
```

- `split` accepts `"train"`, `"val"`, or `"test"` and corresponds to `train_recons.json`, `val_recons.json`, or `test_recons.json` under `sample_root`.
- `max_num_images` should match the sampling `npz` (e.g. 24).
- `sample_type` should match the npz naming (e.g. `mix`, `random`).

2) Integrating with vggt

- Add the `train_dataset/` directory to your Python path (or copy it into the vggt project) and import `MegaDepthXDataset` from your training script or dataloader.
- Replace the dataset/dataloader in `trainer.py` (or your custom loader) with `MegaDepthXDataset` to train on MegaDepth-X samples.
- vggt repository: https://github.com/facebookresearch/vggt/tree/main/vggt

Integration example:
```python
from train_dataset.megadepth_x_dataset import MegaDepthXDataset
# use dataset in your DataLoader / training loop
```

3) Dataset download and structure (HuggingFace)

- MegaDepth-X is released on HuggingFace with per-scene `tar.gz` archives: https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X
- Download and extract the archives locally and point `data_root` to the extracted dataset.
- The sampling files (`train_samples/`, `train_recons.json`, etc.) will also be provided on the HuggingFace release; download those and point `sample_root` to the extracted samples.

Example layout:
```
/path/to/MegaDepth-X/
  ├── Aachen_Cathedral/
  │    └── 1/
  │        ├── images/
  │        ├── depths/
  │        └── ...
  └── Zvartnots/
       └── 0/
           ├── images/
           ├── depths/
           └── ...
/path/to/train_samples/
  ├── train_recons.json
  └── Aachen_Cathedral/1/24_mix.npz
  └── Zvartnots/0/24_mix.npz
```