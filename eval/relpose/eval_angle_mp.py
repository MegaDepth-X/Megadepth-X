
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist
import hydra
import logging
import json
import open3d as o3d
from scipy.ndimage import label
import imageio

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.messages import set_default_arg, write_csv, gather_csv_and_write, make_csvsdir_and_remove_history_csvs
from utils.vis_utils import save_image_grid_auto, predictions_to_glb, depth_edge
from relpose.metric import se3_to_relative_pose_error, calculate_auc_np
from interfaces.color import masked_color_normalization
from models.vggt.utils.geometry import closed_form_inverse_se3
import cv2

# os.environ["NCCL_P2P_DISABLE"]="1"

# os.environ["NCCL_DEBUG"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# os.environ["OMP_NUM_THREADS"]="1"
# os.environ["MKL_NUM_THREADS"]="1"

def save_ply(points, colors, filename):               
    if torch.is_tensor(points):
        points_visual = points.reshape(-1, 3).cpu().numpy()
    else:
        points_visual = points.reshape(-1, 3)
    if torch.is_tensor(colors):
        points_visual_rgb = colors.reshape(-1, 3).cpu().numpy()
    else:
        points_visual_rgb = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_visual.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(points_visual_rgb.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)

def save_log_files(images, pcds, output_dir, conf=None):
    if conf is not None:
        conf = torch.sigmoid(conf)[..., 0]
        conf_mask = conf > 0.2
    pcd_output_dir = os.path.join(output_dir, "pcds")
    os.makedirs(pcd_output_dir, exist_ok=True)
    for idx in range(len(images)):
        if conf is not None:
            to_save_pcds = pcds[idx][conf_mask[idx]]
            to_save_rgbs = images[idx].permute(1, 2, 0)[conf_mask[idx]]
        else:
            to_save_pcds = pcds[idx]
            to_save_rgbs = images[idx].permute(1, 2, 0)
        save_ply(
            to_save_pcds.reshape(-1, 3).float(), 
            to_save_rgbs.reshape(-1, 3).float(), 
            os.path.join(pcd_output_dir, f"{idx}_world_pcd.ply")
        )

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    if not torch.cuda.is_available() or hydra_cfg.device != "cuda":
        raise EnvironmentError("Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage")
    # Get local rank from torchrun environment (guaranteed to be set)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # Map local_rank to a device index (robust to different GPU counts)
    device_idx = local_rank % torch.cuda.device_count()
    # Set the CUDA device for this process BEFORE doing any CUDA work or init_process_group
    torch.cuda.set_device(device_idx)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device_id_val = None
    if torch.cuda.is_available():
        device_id_val = torch.device("cuda", device_idx)
    # Initialize process group and tell NCCL which device this process will use
    # use env:// so torchrun-managed env vars are used
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device_id_val)
    # dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # device = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_idx}")
    print(f"Starting rank={rank} (local_rank={local_rank}) on device={device} world_size={world_size}", flush=True)
    
    # torch.cuda.set_device(device)
    # print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/relpose-angular.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-angular.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]

        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model_logger = logging.getLogger(f"relpose-angle-{model_keyname}-rank{rank}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        # 0.3 route the correct infer function for the model
        # output_root = osp.join(hydra_cfg.log.output_dir, model_name)
        infer_func_cfg = model_info.get(
            "infer_cameras_w2c",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_cameras_w2c',
                '_partial_': True,
            })
        )
        infer_cameras_w2c = hydra.utils.instantiate(infer_func_cfg)
        
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            save_dir = osp.join(hydra_cfg.output_dir, model_keyname)
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            if rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                for file in os.listdir(save_dir):
                    if file.endswith(".npy"):
                        os.remove(osp.join(save_dir, file))
                os.makedirs(output_root, exist_ok=True)
                os.makedirs(osp.join(output_root, "_seq_metrics"), exist_ok=True)
            dist.barrier()  # wait for all ranks to finish this sequence
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            dataset = hydra.utils.instantiate(dataset_info.cfg)

            # 2. ready to read, and look up sampled ids from sequence name
            model.eval()
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map = json.load(f)

            # 3. prepare for metrics
            rError = []
            tError = []
            metric_dict: dict = {}
            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Start evaluating {dataset_name} with {model_keyname}...")

            seq_list_this_rank = dataset.sequence_list[rank::dist.get_world_size()]
            tbar = tqdm(
                seq_list_this_rank,
                desc=f"[Rank {rank}] {model_keyname}-{dataset_name}"
            )
            save_csv_flag = False
            for seq_name in tbar:
                # 4. decide sampling strategy to choose sample frames, from all frames (seq_num_frames) of a sequence
                if seq_name not in seq_id_map:
                    model_logger.warning(f"Sequence {seq_name} not found in seq_id_map, skipping...")
                    continue
                ids = seq_id_map[seq_name]

                # 5. load data sample (only extrinsics are used)
                batch = dataset.get_data(sequence_name=seq_name, ids=ids)
                gt_extrs = batch["extrs"]
                
                seq_name = seq_name.replace("/", "-")
                with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                    rets = infer_cameras_w2c(batch['image_paths'], model, hydra_cfg, batch.get("images_processed", None))

                    # 7. compute metrics
                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                        # pred_se3   = pred_extrs,
                        pred_se3   = rets["pred_extrs"],
                        gt_se3     = gt_extrs,
                        # num_frames = num_frames,
                        num_frames = len(ids),
                    )

                # 8. update metric for a sequence
                tbar.set_postfix_str(f"seq {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")

                rError.extend(rel_rangle_deg.cpu().numpy())
                tError.extend(rel_tangle_deg.cpu().numpy())
                
                # 11. save metrics to csv
                if "images" in batch:
                    write_csv(osp.join(output_root, "_seq_metrics", f"{seq_name.replace('/', '-')}-{rel_rangle_deg.mean().item():.2f}-{rel_tangle_deg.mean().item():.2f}-rank{rank}.csv"), {
                        "seq":    seq_name.replace('/', '-'),
                        "rError_mean":  rel_rangle_deg.mean().item(),
                        "tError_mean":   rel_tangle_deg.mean().item(),
                        "rError_mid":  rel_rangle_deg.median().item(),
                        "tError_mid":   rel_tangle_deg.median().item(),
                        "Racc_5": np.mean(rel_rangle_deg.cpu().numpy() < 5).item() * 100,
                        "Tacc_5": np.mean(rel_tangle_deg.cpu().numpy() < 5).item() * 100,
                        "Racc_15": np.mean(rel_rangle_deg.cpu().numpy() < 15).item() * 100,
                        "Tacc_15": np.mean(rel_tangle_deg.cpu().numpy() < 15).item() * 100,
                        "Racc_30": np.mean(rel_rangle_deg.cpu().numpy() < 30).item() * 100,
                        "Tacc_30": np.mean(rel_tangle_deg.cpu().numpy() < 30).item() * 100,
                    })          
                    save_csv_flag = True      

            np.save(osp.join(output_root, f"rError-rank{rank}.npy"), rError)
            np.save(osp.join(output_root, f"tError-rank{rank}.npy"), tError)
            dist.barrier()  # wait for all ranks to finish this sequence
            
            if rank == 0:
                # 8.5 gather all rError and tError from all ranks
                rErrors = []
                tErrors = []
                for rk in range(dist.get_world_size()):
                    load_rError = np.load(osp.join(output_root, f"rError-rank{rk}.npy"))
                    load_tError = np.load(osp.join(output_root, f"tError-rank{rk}.npy"))
                    rErrors.extend(load_rError)
                    tErrors.extend(load_tError)
                rErrors = np.array(rErrors)
                tErrors = np.array(tErrors)
                if save_csv_flag:
                    df = gather_csv_and_write(
                        input_root=osp.join(output_root, "_seq_metrics"),
                        output_file=osp.join(output_root, "_seq_metrics.csv")
                    )

                # 9. arrange all intermediate results to metrics
                for threshold in dataset_info.metric_thresholds:
                    metric_dict[f"Racc_{threshold}"] = np.mean(rErrors < threshold).item() * 100
                    metric_dict[f"Tacc_{threshold}"] = np.mean(tErrors < threshold).item() * 100
                    Auc, _ = calculate_auc_np(rErrors, tErrors, max_threshold=threshold)
                    metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100
                metric_dict["Rerr_mean"] = rErrors.mean().item()
                metric_dict["Terr_mean"] = tErrors.mean().item()
                metric_dict["Rerr_median"] = np.median(rErrors).item()
                metric_dict["Terr_median"] = np.median(tErrors).item()

                model_logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

                # 9. save evaluation metrics to csv
                statistics_data = {"model": model_keyname, **metric_dict}
                statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
                if getattr(hydra_cfg, "save_suffix", None) is not None:
                    statistics_file += f"-{hydra_cfg.save_suffix}"
                statistics_file += ".csv"
                write_csv(statistics_file, statistics_data)

        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating model {model_keyname} on all datasets.")
        dist.barrier()
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
