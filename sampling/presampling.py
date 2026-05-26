import argparse
import copy
import os
import random
import time
from functools import partial
from multiprocessing import Pool

import networkx as nx
import numpy as np
from tqdm import tqdm

from colmap_utils import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)
from sample_utils_release import *

DEFAULT_N_IMGS = 24
EDGE_COMMON_PTS_MIN = 50
MATCH_COUNT_THRESHOLD = 16
DEFAULT_MAX_N_START = 4


class Sample_State:
    graph = None
    img_meta = None
    cnt = None
    name_id_dict = None
    imdata = None
    camdata = None
    points3D = None
    mean_scale = None


def _optional_path(value):
    return value if value else None


def _first_existing_dir(*candidates):
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def _resolve_scene_paths(scene_dir, recon_id, images_dir=None, sparse_dir=None, depth_dir=None):
    recon_dir = os.path.join(scene_dir, str(recon_id))
    img_dir = images_dir or _first_existing_dir(
        os.path.join(recon_dir, "images")
    )
    sparse_dir = sparse_dir or _first_existing_dir(
        os.path.join(recon_dir, "sparse_manhattan"),
        os.path.join(recon_dir, "sparse"),
    )
    depth_dir = depth_dir or _first_existing_dir(
        os.path.join(recon_dir, "depth"),
    )
    return recon_dir, img_dir, sparse_dir, depth_dir


def _list_recon_ids(scene_dir):
    recon_ids = []
    for name in sorted(os.listdir(scene_dir)):
        path = os.path.join(scene_dir, name)
        if not os.path.isdir(path):
            continue
        if _first_existing_dir(
            os.path.join(path, "images"),
            os.path.join(path, "sparse"),
            os.path.join(path, "sparse_manhattan"),
            os.path.join(path, "depth"),
        ):
            recon_ids.append(name)
    return recon_ids


def img_filter_worker(node, src_img_dir):
    if filter_image(os.path.join(src_img_dir, node)):
        return (True, node)
    else:
        return (False, node)


def depth_filter_worker(node, src_depth_dir):
    depth_path = os.path.join(src_depth_dir, node + ".npy")
    if not os.path.exists(depth_path):
        return (True, node)
    depth = np.load(os.path.join(src_depth_dir, node + ".npy"))
    if np.sum(depth > 0) / (depth.shape[0] * depth.shape[1]) < 0.02:
        return (True, node)
    else:
        return (False, node)


def edge_filter_worker(image_pair):
    name_i, name_j = image_pair
    id_i = Sample_State.name_id_dict[name_i]
    id_j = Sample_State.name_id_dict[name_j]
    common_pts_num = compute_common_pts(Sample_State.imdata[id_i], Sample_State.imdata[id_j], Sample_State.points3D)
    if common_pts_num < EDGE_COMMON_PTS_MIN:
        return (False, image_pair, 0, 0)
    else:
        c2w_i = Sample_State.img_meta["img_meta"][f"{name_i}_extrin"]
        c2w_j = Sample_State.img_meta["img_meta"][f"{name_j}_extrin"]
        distance = np.linalg.norm(c2w_i[:3, 3] - c2w_j[:3, 3], axis=0) / Sample_State.mean_scale
        return (True, image_pair, common_pts_num, distance)


def sample_N_images(
    N_imgs,
    dst_scene_dir,
    N_cases=64,
    max_N_start=DEFAULT_MAX_N_START,
):
    os.makedirs(dst_scene_dir, exist_ok=True)
    sampled_cases = []
    cnt = 0
    graph = Sample_State.img_meta["graph"]

    if graph.number_of_nodes() == 0:
        return

    start = time.time()
    comms_raw = nx.community.louvain_communities(graph)
    end = time.time()
    print(f"Community detection takes {end - start} seconds, the graph is partitioned into {len(comms_raw)} communities.")

    for _ in range(N_cases * 2):
        sampled_nodes = steiner_tree_sampling(
            register_G=graph,
            comms_raw=comms_raw,
            max_depth=random.randint(5, N_imgs),
            N_nodes=N_imgs,
            idx=cnt,
            N_starts=random.randint(1, max_N_start),
        )
        
        if len(sampled_nodes) >= N_imgs:
            sampled_item = {"imgs": {}}
            for img_name in sampled_nodes:
                c2w = Sample_State.img_meta["img_meta"][f"{img_name}_extrin"]
                single_meta = {
                    "c2w": c2w.tolist(),
                    "K": Sample_State.img_meta["img_meta"][f"{img_name}_intrin"].tolist(),
                    "depth_path": img_name + ".npy"
                }
                sampled_item["imgs"][img_name] = single_meta
            sampled_item["edges"] = list(graph.subgraph(sampled_nodes).edges)
            cnt += 1
            
            sampled_cases.append(sampled_item)
        else:
            print(f"Sampled {len(sampled_nodes)} nodes, less than {N_imgs}, re-sample ...")
        if cnt >= N_cases:
            break

    if sampled_cases:
        np.savez_compressed(os.path.join(dst_scene_dir, f"{N_imgs}_mix.npz"), sampled_cases=sampled_cases)


def sample_scene(
    scene_dir,
    recon_id,
    n_cases=None,
    depth_dir=None,
    db_path=None,
    output_dir=None,
    images_dir=None,
    sparse_dir=None,
):
    scene_dir = os.path.abspath(scene_dir)
    scene_name = os.path.basename(scene_dir)
    recon_dir, src_img_dir, src_sparse_dir, depth_dir = _resolve_scene_paths(
        scene_dir,
        recon_id,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        depth_dir=depth_dir,
    )

    if not src_img_dir:
        print(f"Scene {scene_name}, image directory not found under {recon_dir}, skip.")
        return
    if not src_sparse_dir:
        print(f"Scene {scene_name}, sparse directory not found under {recon_dir}, skip.")
        return

    db_path = db_path or os.path.join(scene_dir, "database.db")
    if not os.path.exists(db_path):
        print(f"Scene {scene_name}, database not found at {db_path}, skip.")
        return

    output_dir = output_dir or os.path.join(scene_dir, "samples")
    dst_scene_dir = os.path.join(output_dir, str(recon_id))

    if os.path.exists(os.path.join(dst_scene_dir, f"{DEFAULT_N_IMGS}_mix.npz")):
        try:
            _ = np.load(os.path.join(dst_scene_dir, f"{DEFAULT_N_IMGS}_mix.npz"), allow_pickle=True)
            print(f"Scene {scene_name}, already processed, skip.")
            return
        except Exception:
            print(f"Scene {scene_name}, npz file corrupted, re-process.")

    try:
        start_db = time.time()
        pairs_raw, matches_raw, _, _, _, _ = read_two_view_from_db(db_path)
        end_db = time.time()
        print(f"Reading from database takes {end_db - start_db} seconds.")

        pairs = []
        matches_num = []
        for pair, match in zip(pairs_raw, matches_raw):
            if len(match) > MATCH_COUNT_THRESHOLD:
                pairs.append(pair)
                matches_num.append(len(match))

        G = build_image_graph(pairs, matches_num)
    except Exception as e:
        print(f"Scene {scene_name}, failed to read database ({e}), try to read from sfm reconstruction.")
        if os.path.exists(os.path.join(src_sparse_dir, "images.txt")):
            imdata = read_images_text(os.path.join(src_sparse_dir, "images.txt"))
            camdata = read_cameras_text(os.path.join(src_sparse_dir, "cameras.txt"))
            points3D = read_points3D_text(os.path.join(src_sparse_dir, "points3D.txt"))
        else:
            imdata = read_images_binary(os.path.join(src_sparse_dir, "images.bin"))
            camdata = read_cameras_binary(os.path.join(src_sparse_dir, "cameras.bin"))
            points3D = read_points3D_binary(os.path.join(src_sparse_dir, "points3D.bin"))

        imnames = [imdata[k].name for k in imdata]

        name_id_dict = {imdata[k].name: k for k in imdata}

        pairs = []
        matches_num = []
        for i in tqdm(range(len(imnames))):
            for j in range(i + 1, len(imnames)):
                id_i = name_id_dict[imnames[i]]
                id_j = name_id_dict[imnames[j]]
                common_pts_num = compute_common_pts(imdata[id_i], imdata[id_j], points3D)
                if common_pts_num >= EDGE_COMMON_PTS_MIN:
                    pairs.append((imnames[i], imnames[j]))
                    matches_num.append(common_pts_num)

        G = build_image_graph(pairs, matches_num)

    exist_imgs = []
    for node in G.nodes():
        if os.path.exists(os.path.join(src_img_dir, node)):
            exist_imgs.append(node)
    G = nx.Graph(copy.deepcopy(G.subgraph(exist_imgs)))

    node_list = list(G.nodes())
    local_img_filter_worker = partial(img_filter_worker, src_img_dir=src_img_dir)
    with Pool(processes=4) as pool:
        _raw_filter_list = list(pool.imap(local_img_filter_worker, node_list))

    filter_list = []
    for pair in _raw_filter_list:
        if pair[0]:
            filter_list.append(pair[1])

    G.remove_nodes_from(filter_list)
    print(f"{len(filter_list)} images are filtered out due to fisheye or panorama effect.")

    node_list = list(G.nodes())
    local_depth_filter_worker = partial(depth_filter_worker, src_depth_dir=depth_dir)
    with Pool(processes=4) as pool:
        _raw_filter_list = list(pool.imap(local_depth_filter_worker, node_list))

    filter_list = []
    for pair in _raw_filter_list:
        if pair[0]:
            filter_list.append(pair[1])

    G.remove_nodes_from(filter_list)
    print(f"{len(filter_list)} images are filtered out due to sparse depth maps.")

    if len(G.nodes()) == 0:
        return

    if os.path.exists(os.path.join(src_sparse_dir, "images.txt")):
        Sample_State.imdata = read_images_text(os.path.join(src_sparse_dir, "images.txt"))
        Sample_State.camdata = read_cameras_text(os.path.join(src_sparse_dir, "cameras.txt"))
        Sample_State.points3D = read_points3D_text(os.path.join(src_sparse_dir, "points3D.txt"))
    else:
        Sample_State.imdata = read_images_binary(os.path.join(src_sparse_dir, "images.bin"))
        Sample_State.camdata = read_cameras_binary(os.path.join(src_sparse_dir, "cameras.bin"))
        Sample_State.points3D = read_points3D_binary(os.path.join(src_sparse_dir, "points3D.bin"))
    img_names = [Sample_State.imdata[k].name for k in Sample_State.imdata if Sample_State.imdata[k].name in exist_imgs]
    Sample_State.name_id_dict = {}
    for key in Sample_State.imdata:
        Sample_State.name_id_dict[Sample_State.imdata[key].name] = key

    bottom = np.array([[0, 0, 0, 1.]])
    w2c_mats = []
    intrinsics = []
    for img_name in img_names:
        im = Sample_State.imdata[Sample_State.name_id_dict[img_name]]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]

        cam = Sample_State.camdata[im.camera_id]
        intrinsics.append(COLMAP_intrinsic_to_K(cam.model, cam.params))
    w2c_mats = np.stack(w2c_mats, 0)
    intrinsics = np.stack(intrinsics, 0)
    poses = np.linalg.inv(w2c_mats)
    Sample_State.mean_scale = np.linalg.norm(poses[..., :3, 3], axis=-1).mean()

    Sample_State.img_meta = {}
    Sample_State.img_meta["graph"] = nx.Graph(G.subgraph(img_names))
    Sample_State.img_meta["img_meta"] = {}
    Sample_State.img_meta["cnt"] = len(img_names)
    for img_name, pose, intrinsic in zip(img_names, poses, intrinsics):
        if Sample_State.img_meta["graph"].has_node(img_name):
            Sample_State.img_meta["img_meta"][f"{img_name}_extrin"] = pose
            Sample_State.img_meta["img_meta"][f"{img_name}_intrin"] = intrinsic

    if Sample_State.img_meta["graph"].number_of_nodes() == 0:
        return

    components = nx.connected_components(Sample_State.img_meta["graph"])
    largest_cc = max(components, key=len)
    num_nodes_max = len(largest_cc)
    sub_graph_edges = list(Sample_State.img_meta["graph"].edges)
    original_edge_num = len(sub_graph_edges)
    print(
        f"Before pruning, sub_graph contains {Sample_State.img_meta['graph'].number_of_nodes()} nodes, "
        f"{original_edge_num} edges and {nx.number_connected_components(Sample_State.img_meta['graph'])} "
        f"connected components, the largest component has {num_nodes_max} nodes."
    )
    start = time.time()
    if not (
        Sample_State.img_meta["graph"].number_of_nodes() < DEFAULT_N_IMGS
        or original_edge_num < (DEFAULT_N_IMGS - 1)
    ):
        print(
            f"Scene {scene_name}, only {Sample_State.img_meta['graph'].number_of_nodes()} nodes "
            f"and {original_edge_num} edges, skip."
        )
        return

    with Pool(processes=4) as pool:
        _results = list(pool.imap(edge_filter_worker, sub_graph_edges))

    for triple in _results:
        name_i, name_j = triple[1]
        if triple[0]:
            Sample_State.img_meta["graph"][name_i][name_j]["common_pts"] = triple[2]
            Sample_State.img_meta["graph"][name_i][name_j]["distance"] = triple[3]
        else:
            Sample_State.img_meta["graph"].remove_edge(name_i, name_j)
    sub_graph_edges = list(Sample_State.img_meta["graph"].edges)
    after_edge_num = len(sub_graph_edges)
    if original_edge_num > 0:
        print(f"{after_edge_num / original_edge_num * 100}% edges are kept.")

    components = nx.connected_components(Sample_State.img_meta["graph"])
    largest_cc = max(components, key=len)
    num_nodes_max = len(largest_cc)
    end = time.time()
    print(
        f"[{end - start}s] After pruning, sub_graph contains "
        f"{Sample_State.img_meta['graph'].number_of_nodes()} nodes, {after_edge_num} edges "
        f"and {nx.number_connected_components(Sample_State.img_meta['graph'])} connected components, "
        f"the largest component has {num_nodes_max} nodes."
    )

    if n_cases is None or n_cases <= 0:
        n_cases = max(1, Sample_State.img_meta["graph"].number_of_nodes() // 8)
    else:
        n_cases = max(1, int(n_cases))
    sample_N_images(
        DEFAULT_N_IMGS,
        dst_scene_dir,
        N_cases=n_cases,
        max_N_start=DEFAULT_MAX_N_START,
    )


def run(
    scene_dir,
    recon_id,
    n_cases=None,
    db_path=None,
    output_dir=None,
    depth_dir=None,
    images_dir=None,
    sparse_dir=None,
):
    if not os.path.isdir(scene_dir):
        raise ValueError(f"scene_dir not found: {scene_dir}")
    if recon_id is None:
        recon_id = ""

    recon_id_str = str(recon_id).strip()
    if recon_id_str.lower() in {"all", "auto", "*"} or recon_id_str == "":
        recon_ids = _list_recon_ids(scene_dir)
    elif "," in recon_id_str:
        recon_ids = [r.strip() for r in recon_id_str.split(",") if r.strip()]
    else:
        recon_ids = [recon_id_str]

    if not recon_ids:
        print(f"No recon directories found under {scene_dir}, skip.")
        return

    for rid in recon_ids:
        sample_scene(
            scene_dir=scene_dir,
            recon_id=rid,
            n_cases=n_cases,
            depth_dir=depth_dir,
            db_path=db_path,
            output_dir=output_dir,
            images_dir=images_dir,
            sparse_dir=sparse_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Presampling for a single scene.")
    parser.add_argument("--scene_dir", type=str, default="")
    parser.add_argument("--recon_id", type=str, default="all")
    parser.add_argument("--n_cases", type=int, default=64)
    parser.add_argument("--db_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--depth_dir", type=str, default="")
    parser.add_argument("--images_dir", type=str, default="")
    parser.add_argument("--sparse_dir", type=str, default="")
    args = parser.parse_args()

    random.seed(42)

    run(
        scene_dir=args.scene_dir,
        recon_id=args.recon_id,
        n_cases=args.n_cases,
        db_path=_optional_path(args.db_path),
        output_dir=_optional_path(args.output_dir),
        depth_dir=_optional_path(args.depth_dir),
        images_dir=_optional_path(args.images_dir),
        sparse_dir=_optional_path(args.sparse_dir),
    )