<div align="center">
  <h1>Long-Tail Internet Photo Reconstruction</h1>
  <p><strong>CVPR 2026</strong></p>
  <p>
    <a href="https://megadepth-x.github.io/">
      <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-brightgreen">
    </a>
    <a href="https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X">
      <img alt="Dataset" src="https://img.shields.io/badge/Dataset-HuggingFace-blue">
    </a>
  </p>
  <p>
    Yuan Li<sup>1</sup>, Yuanbo Xiangli<sup>1</sup>&dagger;, Hadar Averbuch-Elor<sup>1</sup>,
    Noah Snavely<sup>1</sup>, Ruojin Cai<sup>2</sup>&dagger;
  </p>
  <p><sup>1</sup>Cornell University &nbsp;&nbsp; <sup>2</sup>Kempner Institute, Harvard University</p>
</div>

This repository contains scripts for depth post-processing, presampling, and evaluation used in the MegaDepth-X pipeline.

## Overview
- depth_processing: filter COLMAP depth maps (*.geometric.bin) using semantic segmentation and a monocular prior, save .npy depth maps.
- sampling: build an image graph from COLMAP database or sparse reconstruction, filter images and depths, and sample multi-view cases.
- eval: hydra-based evaluation for relative pose (angular error) and multi-view reconstruction metrics.

## Dependencies
This project is based on Python 3.10. Install dependencies via requirements file:

```bash
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0 --no-cache-dir --no-build-isolation
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Data layout (per scene)
SfM is obtained via [MASt3R](https://github.com/naver/mast3r) and [Doppelgangers++](https://github.com/doppelgangers25/doppelgangers-plusplus).
MVS is obtained via [COLMAP-MVS](https://colmap.github.io/cli.html#example).

Example structure:

```
<scene>/
|-- <recon_id>/
|   |-- images/
|   |-- stereo/
|   |   `-- depth_maps/
|   `-- sparse/
`-- database.db
```

## Depth processing
Filter COLMAP depth maps with semantic segmentation and MoGe monocular prior:

```bash
python depth_processing/filter_colmap_depths.py \
  --colmap_root <scene_dir>/<recon_id> \
  --seg_ckpt <path/to/segformer_checkpoint.pth> \
  --seg_cfg <path/to/segformer_config.py> \
  --device cuda:0
```

Notes:
- By default, images are read from <colmap_root>/images and depth maps from <colmap_root>/stereo/depth_maps.
- Output is written to <colmap_root>/depth as .npy files.
- Use --job_idx and --n_jobs to split work across multiple jobs.

Segmentation model note:

- We use SegFormer from the mmsegmentation project for semantic segmentation. You can find the configs and checkpoints here: https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer
- Recommended config used in this pipeline: `segformer_mit-b5_8xb2-160k_ade20k-640x640.py`. Use the matching checkpoint for best results.

## Presampling
Generate multi-view samples for a scene:

```bash
python sampling/presampling.py \
  --scene_dir <scene_dir> \
  --recon_id all \
  --n_cases 64
```

Notes:
- If database.db exists under the scene, it is used to build the graph; otherwise it falls back to sparse reconstruction.
- Depth maps are read from <scene_dir>/<recon_id>/depth by default.
- Output is saved to <scene_dir>/samples/<recon_id>/24_mix.npz (DEFAULT_N_IMGS = 24).

## Evaluation
This evaluation code is based on [recons_eval](https://github.com/ZhouTimeMachine/recons_eval).
- Checkpoints for Pi3 and VGGT: [ckpts](https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/tree/main/ckpts). Please set `pretrained_model_name_or_path` in `eval/configs/model/default.yaml` to the local path of the pretrained or fine-tuned model.

Edit dataset paths in these configs first:
- eval/configs/data/relpose-angular.yaml
- eval/configs/data/mv_recon.yaml
- Pre-sampled test cases: [test_samples](https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/tree/main/test_samples). Please set `MegaDepth_X_DIR` and `sample_DIR` in `eval/configs/data/relpose-angular.yaml` and `eval/configs/data/mv_recon.yaml` to the local paths of MegaDepth-X and the test samples.

Relative pose evaluation (multi GPU, DDP):

```bash
cd eval
torchrun --nproc_per_node=<N> relpose/eval_angle_mp.py
```

Multi-view reconstruction evaluation (multi GPU, DDP):

```bash
cd eval
torchrun --nproc_per_node=<N> mv_recon/eval_mp.py
```

Outputs are written to:

```
eval/outputs/<name>/<model>/<dataset>/
```

where <name> is relpose-angular or mv_recon (see eval/configs/general/default.yaml). Per-sequence CSV metrics are saved under _seq_metrics.

## References
- [VGGT](https://github.com/facebookresearch/vggt/tree/main/vggt)
- [Pi3](https://github.com/yyfz/Pi3)
- [recons_eval](https://github.com/ZhouTimeMachine/recons_eval)

## Citation
```
@inproceedings{li2026longtail,
  title={Long-Tail Internet Photo Reconstruction},
  author={Li, Yuan and Xiangli, Yuanbo and Averbuch-Elor, Hadar and Snavely, Noah and Cai, Ruojin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```