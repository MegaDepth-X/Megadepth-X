import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from tqdm import tqdm

from mmseg.apis import inference_model, init_model
from moge.model.v2 import MoGeModel

from post_process_utils import filter_with_monocular_prior, post_process

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def rotate_depth_to_upright(depth, orientation):
    if orientation == 2:
        return np.fliplr(depth)
    if orientation == 3:
        return np.flipud(np.fliplr(depth))
    if orientation == 4:
        return np.flipud(depth)
    if orientation == 5:
        return np.fliplr(np.rot90(depth, k=-1))
    if orientation == 6:
        return np.rot90(depth, k=-1)
    if orientation == 7:
        return np.fliplr(np.rot90(depth, k=1))
    if orientation == 8:
        return np.rot90(depth, k=1)
    return depth


def rotate_depth_from_upright(depth, orientation):
    if orientation == 2:
        return np.fliplr(depth)
    if orientation == 3:
        return np.flipud(np.fliplr(depth))
    if orientation == 4:
        return np.flipud(depth)
    if orientation == 5:
        return np.flipud(np.rot90(depth, k=1))
    if orientation == 6:
        return np.rot90(depth, k=1)
    if orientation == 7:
        return np.flipud(np.rot90(depth, k=-1))
    if orientation == 8:
        return np.rot90(depth, k=-1)
    return depth



def visualize_depth(depth, out_path):
    import cv2
    valid_mask = depth > 0.
    if not valid_mask.any():
        dmin, dmax = 0, 1
    else:
        dmin, dmax = depth[valid_mask].min(), depth[valid_mask].max()
    if dmax == dmin:
        vis = np.zeros(depth.shape, dtype=np.uint8)
    else:
        vis = (255 * (depth - dmin) / (dmax - dmin))
        vis[valid_mask == 0] = 0
        vis = vis.astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_PLASMA)
        vis[valid_mask == 0] = 0
    cv2.imwrite(str(out_path), vis)

def build_image_path(images_dir, depth_rel):
    img_rel = Path(str(depth_rel).replace(".geometric.bin", ""))
    return images_dir / img_rel


def filter_depth_file(seg_model, depth_model, depth_path, image_path, device):
    img = Image.open(str(image_path)).convert("RGB")
    orientation = img.getexif().get(274, 1)
    img_rotated = ImageOps.exif_transpose(img)
    img_rgb = np.array(img_rotated)

    depth = read_array(str(depth_path))
    if orientation != 1:
        depth = rotate_depth_to_upright(depth, orientation)

    result = inference_model(seg_model, str(image_path))
    seg_map = result.pred_sem_seg.data[0].cpu().numpy().astype(np.int32)
    depth_1st_pass = post_process(depth, seg_map)

    input_image = torch.tensor(img_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    output = depth_model.infer(input_image)
    mono_depth = output["depth"].cpu().numpy()
    mono_mask = output["mask"].cpu().numpy()

    depth_filtered = filter_with_monocular_prior(depth_1st_pass, mono_depth, mono_mask)
    if orientation != 1:
        depth_filtered = rotate_depth_from_upright(depth_filtered, orientation)

    return depth_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter COLMAP geometric.bin depth maps with semantic and monocular priors."
    )
    parser.add_argument(
        "--colmap_root",
        required=True,
        help="Path to a COLMAP reconstruction root (e.g. debug/Aachen_Cathedral/1).",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Optional image directory (default: <colmap_root>/images_w_exif).",
    )
    parser.add_argument(
        "--depth_dir",
        default=None,
        help="Optional depth directory (default: <colmap_root>/stereo/depth_maps).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: <colmap_root>/filtered_depth).",
    )
    parser.add_argument("--job_idx", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seg_ckpt", required=True)
    parser.add_argument("--seg_cfg", required=True)
    parser.add_argument(
        "--moge_ckpt",
        default="Ruicheng/moge-2-vitl-normal",
        help="HuggingFace model id or local path.",
    )
    args = parser.parse_args()

    colmap_root = Path(args.colmap_root)
    images_dir = Path(args.images_dir) if args.images_dir else (colmap_root / "images")
    depth_dir = Path(args.depth_dir) if args.depth_dir else (colmap_root / "stereo" / "depth_maps")
    output_dir = Path(args.output_dir) if args.output_dir else (colmap_root / "depth")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    seg_model = init_model(args.seg_cfg, args.seg_ckpt, device=args.device)
    depth_model = MoGeModel.from_pretrained(args.moge_ckpt).to(device)

    depth_files = sorted(depth_dir.rglob("*.geometric.bin"))
    depth_files = depth_files[args.job_idx :: args.n_jobs]

    if not depth_files:
        print(f">> No depth files found under {depth_dir}")
        return

    failed = 0
    for depth_path in tqdm(depth_files, desc="Filtering depth"):
        try:
            rel_path = depth_path.relative_to(depth_dir)
            image_path = build_image_path(images_dir, rel_path)
            if not image_path.exists():
                print(f">> Missing image for {rel_path}, expected {image_path}")
                failed += 1
                continue

            depth_filtered = filter_depth_file(
                seg_model, depth_model, depth_path, image_path, device
            )

            out_path = output_dir / (str(rel_path).replace(".geometric.bin", "") + ".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth_filtered)
            # vis_path = output_dir / (str(rel_path).replace('.geometric.bin', '') + '_vis.png')
            # visualize_depth(depth_filtered, vis_path)            
        except Exception as exc:
            print(f">> Error processing {depth_path}: {exc}")
            failed += 1

    print(f">> Done. Total: {len(depth_files)}, Failed: {failed}")


if __name__ == "__main__":
    main()
