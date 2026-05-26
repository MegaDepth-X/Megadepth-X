import os
from pathlib import Path
from multiprocessing import Pool, Lock, Manager, cpu_count
import random
import networkx as nx
import numpy as np
import heapq
from tqdm import tqdm
import time

from PIL import Image, ImageFile, ImageOps
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import piexif

from database import pair_id_to_image_ids, blob_to_array
from database import COLMAPDatabase

# Thresholds for filtering
FISHEYE_FOCAL_LENGTH_35MM_MAX = 10.0  # max 35mm focal length to consider fisheye or heavily distorted images
FISHEYE_FOCAL_LENGTH_MAX = 2.5  # max focal length to consider fisheye or heavily distorted images
PANORAMA_ASPECT_RATIO_THRESHOLD = 3.0  # aspect ratio above this considered panorama

def get_exif_data(image, fpath):
    try:
        # image = Image.open(fpath)
        exif_dict = piexif.load(image.info.get('exif', b''))
        return exif_dict
    except Exception as e:
        # print(f"Failed to read EXIF for {fpath}: {e}")
        return None

def get_tag_value(exif_dict, ifd, tag):
    # Safe retrieval of tag value from EXIF dict
    try:
        value = exif_dict[ifd][tag]
        if isinstance(value, bytes):
            return value.decode(errors='ignore')
        return value
    except KeyError:
        return None

def rational_to_float(rational):
    # Convert rational (num, den) to float
    if isinstance(rational, tuple) and len(rational) == 2 and rational[1] != 0:
        return rational[0] / rational[1]
    return None

def get_focal_length_35mm(exif_dict):
    # Tag 41989 in EXIF = FocalLengthIn35mmFilm
    tag = 41989
    val = get_tag_value(exif_dict, "Exif", tag)
    if val is None:
        return None
    if isinstance(val, int):
        return float(val)
    return None

def get_focal_length(exif_dict):
    # Tag 37386 in EXIF = FocalLength
    tag = 37386
    val = get_tag_value(exif_dict, "Exif", tag)
    if val is None:
        return None
    if isinstance(val, int):
        return float(val)
    return None

def get_lens_model(exif_dict):
    # Tag 42036 = LensModel (ASCII)
    tag = 42036
    val = get_tag_value(exif_dict, "Exif", tag)
    if val is None or not isinstance(val, str):
        return ""
    return val.lower()

def get_model(exif_dict):
    # Tag 272 = Model
    tag = 272
    val = get_tag_value(exif_dict, "0th", tag)
    if val is None or not isinstance(val, str):
        return ""
    return val.lower()

def get_make(exif_dict):
    # Tag 271 = Make
    tag = 271
    val = get_tag_value(exif_dict, "0th", tag)
    if val is None or not isinstance(val, str):
        return ""
    return val.lower()

def is_fisheye(exif_dict):
    abnormal_fullfocal = [0.0, 3.0, 4.0, 5.0] # may considered as smart phone cameras
    abnormal_focal = [1.1] # may considered as smart phone cameras
    lens_model = get_lens_model(exif_dict)
    if "fisheye" in lens_model:
        return True
    focal = get_focal_length(exif_dict)
    if focal is not None and focal <= FISHEYE_FOCAL_LENGTH_MAX and focal > 0:
        if focal not in abnormal_focal:
            return True
    focal_35mm = get_focal_length_35mm(exif_dict)
    if focal_35mm is not None and focal_35mm <= FISHEYE_FOCAL_LENGTH_35MM_MAX and focal_35mm > 0:
        if focal_35mm not in abnormal_fullfocal:
            return True
    return False

def is_panorama(image, fpath):
    w, h = image.size
    aspect_ratio = w / h if w >= h else h / w
    image_name = os.path.basename(fpath)
    image_name = image_name.lower()
    name_identifier = "pano" in image_name and "panoramio" not in image_name
    return aspect_ratio >= PANORAMA_ASPECT_RATIO_THRESHOLD or name_identifier

def filter_image(fpath):
    try:
        image = Image.open(fpath)
    except Exception as e:
        print(f"Failed to read Image for {fpath}: {e}")
        return True
    exif = get_exif_data(image, fpath)
    if exif is None:
        return False
    if is_fisheye(exif):
        # print(f"Filter out fisheye camera: {fpath}")
        return True
    if is_panorama(image, fpath):
        # print(f"Filter out panorama image: {fpath}")
        return True
    return False

def is_panorama(image, fpath):
    w, h = image.size
    aspect_ratio = w / h if w >= h else h / w
    image_name = os.path.basename(fpath)
    image_name = image_name.lower()
    name_identifier = "pano" in image_name and "panoramio" not in image_name
    return aspect_ratio >= PANORAMA_ASPECT_RATIO_THRESHOLD or name_identifier


def COLMAP_intrinsic_to_K(model, params):
    # suppose all images are undistorted images by COLMAP, which are pinhole cameras
    assert model in ["SIMPLE_PINHOLE", "PINHOLE"], f"Unsupported camera model: {model}"
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]])
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
    
    return K

def read_two_view_from_db_old(database_path):
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    pairs = []
    matches = []
    pairs_id = []

    for pair_id, rows, cols, data in db.execute("SELECT pair_id, rows, cols, data FROM matches"):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        name1 = name1.replace(" ", "_")
        name2 = name2.replace(" ", "_")
        if data is None:
            continue
        pairs_id.append(pair_id)
        pairs.append((name1, name2))
        match = blob_to_array(data, np.uint32, (rows, cols))
        matches.append(match)

    db.close()
    return pairs, matches, pairs_id

def read_two_view_from_db(database_path):
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    pairs = []
    matches = []
    H_list = []
    E_list = []
    F_list = []
    pairs_id = []

    for pair_id, rows, cols, data, config, F, E, H, qvec, tvec in db.execute("SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec FROM two_view_geometries"):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        name1 = name1.replace(" ", "_")
        name2 = name2.replace(" ", "_")
        if data is None:
            continue
        pairs_id.append(pair_id)
        pairs.append((name1, name2))
        match = blob_to_array(data, np.uint32, (rows, cols))
        matches.append(match)
        F = blob_to_array(F, np.float64).reshape(-1,3)
        E = blob_to_array(E, np.float64).reshape(-1,3)
        H = blob_to_array(H, np.float64).reshape(-1,3)
        qvec = blob_to_array(qvec, np.float64)
        tvec = blob_to_array(tvec, np.float64)
        H_list.append(H)
        F_list.append(F)
        E_list.append(E)
    db.close()
    return pairs, matches, E_list, F_list, H_list, pairs_id

def compute_common_pts(image_i, image_j, points3D):
    # Match 3D points observed by both images
    pid_i = set(image_i.point3D_ids)
    pid_j = set(image_j.point3D_ids)
    shared_pids = pid_i & pid_j
    return len(shared_pids) - 1

def beam_farthest_point_sampling(G, n_samples, node_comms, max_depth=None, n_trials=256, top_k=16):
    """
    Farthest-point sampling with branching on graph G.

    At each step, considers the frontier of ALL currently sampled nodes
    (not just the last one), and greedily picks the "farthest" frontier node
    to maximize spatial spread and community diversity.  This allows tree-like
    (branching) exploration instead of depth-first-search paths.

    Multiple random-start trials are run; the best *top_k* results (ranked by
    community diversity, then cumulative distance) are returned.

    Args:
        G: networkx graph (should be connected)
        n_samples: number of nodes to sample
        node_comms: dict mapping node -> community id
        max_depth: maximum number of expansion steps (hops from start).
                   If None, defaults to n_samples - 1 (no extra limit).
        n_trials: number of random starting points to try
        top_k: number of best results to return

    Returns:
        sampled_lists: list of sampled-node lists  (up to top_k, sorted best-first)
        comm_lists:    list of traversed-community sets (parallel to sampled_lists)
    """
    all_nodes = list(G.nodes)

    if len(all_nodes) == 0:
        return [[]], [set()]

    n_samples = min(n_samples, len(all_nodes))

    if len(all_nodes) <= n_samples:
        comms = {node_comms.get(n, 0) for n in all_nodes}
        return [all_nodes], [comms]

    if max_depth is None:
        max_depth = n_samples - 1
    # The number of expansion steps is the lesser of max_depth and n_samples-1
    n_steps = min(max_depth, n_samples - 1)

    results = []
    n_trials = min(n_trials, len(all_nodes) * 4)  # cap to avoid wasted work

    for _ in range(n_trials):
        start = random.choice(all_nodes)
        sampled = [start]
        sampled_set = {start}
        traversed_comms = {node_comms.get(start, 0)}
        comm_gain_total = 0
        comm_dis_gain_total = 0.0
        dist_total = 0.0

        for _ in range(n_steps):
            # Frontier: neighbours of ANY sampled node that are not yet sampled
            frontier = set()
            for node in sampled[-1:]:
                for neighbor in G.neighbors(node):
                    if neighbor not in sampled_set:
                        frontier.add(neighbor)

            if not frontier:
                break

            # Score every frontier node  ------------------------------------------------
            #   primary key  : community diversity bonus  (1 if new community, else 0)
            #   secondary key: distance-weighted community gain (distance if new community, else 0)
            #   tertiary key : min edge-distance to the sampled set
            #                  (farthest-point heuristic: we maximise this)
            scored = []
            for candidate in frontier:
                # min edge weight to any already-sampled neighbour
                min_dist = float('inf')
                for s_node in sampled:
                    if G.has_edge(candidate, s_node):
                        edge_dist = G[candidate][s_node].get("distance", 1.0)
                        if random.random() < 0.3:
                            edge_dist = 1e-4
                        min_dist = min(min_dist, edge_dist)
                if min_dist == float('inf'):
                    min_dist = 1.0  # fallback (should not happen for true frontier)

                cand_comm = node_comms.get(candidate, 0)
                comm_bonus = 1 if cand_comm not in traversed_comms else 0
                comm_dis_bonus = min_dist if comm_bonus else 0.0
                scored.append((comm_bonus, comm_dis_bonus, min_dist, candidate))

            scored.sort(reverse=True)  # best first

            # Pick with controlled randomness so different trials explore differently
            if random.random() < 0.7:
                chosen = scored[0][3]
            else:
                top_n = max(1, len(scored) // 3)
                chosen = random.choice(scored[:top_n])[3]

            sampled.append(chosen)
            sampled_set.add(chosen)

            # accumulate distance for ranking
            min_d = float('inf')
            for s_node in sampled[:-1]:
                if G.has_edge(chosen, s_node):
                    min_d = min(min_d, G[chosen][s_node].get("distance", 1.0))
            if min_d == float('inf'):
                min_d = 1.0
            dist_total += min_d

            chosen_comm = node_comms.get(chosen, 0)
            if chosen_comm not in traversed_comms:
                comm_gain_total += 1
                comm_dis_gain_total += min_d
                traversed_comms.add(chosen_comm)

        results.append((comm_gain_total, comm_dis_gain_total, dist_total, sampled, traversed_comms))

    # Rank: community diversity first, then distance-weighted community gain, then cumulative distance
    results.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    top_results = results[:top_k]

    if top_results:
        print(f">>>>>  In top {top_k}, best  spans {len(top_results[0][4])} communities "
              f"({len(top_results[0][3])} nodes)")
        print(f">>>>>  In top {top_k}, worst spans {len(top_results[-1][4])} communities "
              f"({len(top_results[-1][3])} nodes)")

    return [r[3] for r in top_results], [r[4] for r in top_results]


# keep the old name as an alias so existing imports still work
beam_random_long_paths_deduplicate = beam_farthest_point_sampling


# def steiner_tree_sampling(register_G, max_depth=10, N_nodes=20, idx=0):
def steiner_tree_sampling(register_G, comms_raw, max_depth=10, N_nodes=24, idx=0, N_starts=1):
    """
    Sample N_nodes from register_G using partitioned farthest-point sampling.

    Algorithm
    ---------
    1.  Randomly partition register_G into N_starts non-overlapping **connected**
        subgraphs by picking N_starts random seed nodes and growing BFS regions
        in round-robin until every reachable node is assigned.
    2.  Allocate nodes proportionally across partitions (minimum 2 per partition).
    3.  Run ``beam_farthest_point_sampling`` on each partition independently,
        producing one sampled "path" per partition.
    4.  Combine and return the union of sampled nodes.

    Parameters
    ----------
    register_G : nx.Graph
        The full registration graph (may be disconnected).
    max_depth : int
        Kept for backward-compatible call signatures; not used internally.
    N_nodes : int
        Total number of nodes to sample across all partitions.
    N_starts : int
        Number of partitions (= number of paths to sample from the graph).
    idx : int
        Controls the number of random trials (higher → more exploration).
    """
    if register_G.number_of_nodes() == 0:
        print(">>> The graph to be sampled is empty.")
        return []

    # ------------------------------------------------------------------
    # Step 0: community detection (used only for diversity scoring inside
    #         beam_farthest_point_sampling, NOT for partitioning)
    # ------------------------------------------------------------------
    # comms_raw = nx.community.louvain_communities(register_G)
    node_community = {}
    for i, comm in enumerate(comms_raw):
        for n in comm:
            node_community[n] = i
    n_cc = nx.number_connected_components(register_G)
    print(f">>> The graph has {len(register_G.nodes)} nodes, "
          f"{n_cc} connected components, "
          f"{len(comms_raw)} Louvain communities.")

    # Clamp N_starts so we can give at least 2 nodes to every partition
    N_starts = max(1, min(N_starts, N_nodes // 2, register_G.number_of_nodes() // 2))

    # ------------------------------------------------------------------
    # Step 1: randomly partition register_G into N_starts connected regions
    #   - Drop connected components with fewer than min_cc_size nodes.
    #   - Distribute seeds across remaining CCs proportionally to their
    #     size (larger CC gets more seeds).  Every kept CC gets at least 1.
    #   - Grow BFS frontiers in round-robin; each node is assigned to the
    #     first partition that reaches it.
    #   - Result: N_starts non-overlapping connected node sets that together
    #     cover all reachable nodes.
    # ------------------------------------------------------------------
    start = time.time()

    # Filter out small connected components (< min_cc_size nodes)
    min_cc_size = max(2, N_nodes // N_starts)   # need at least enough for one partition
    ccs = [cc for cc in nx.connected_components(register_G) if len(cc) >= min_cc_size]
    if not ccs:
        # Fallback: keep the largest connected component
        ccs = [max(nx.connected_components(register_G), key=len)]
    ccs.sort(key=len, reverse=True)
    ccs = ccs[:N_starts]  # keep at most N_starts CCs (if there are that many)

    # Allocate seeds proportionally to CC sizes
    total_cc_nodes = sum(len(cc) for cc in ccs)
    seeds = []
    seeds_per_cc = []
    remaining_seeds = N_starts
    for i, cc in enumerate(ccs):
        if i == len(ccs) - 1:
            n_seeds = remaining_seeds          # last CC gets whatever is left
        else:
            n_seeds = max(1, round(N_starts * len(cc) / total_cc_nodes))
            n_seeds = min(n_seeds, remaining_seeds - (len(ccs) - 1 - i))  # reserve 1 for each remaining CC
        n_seeds = max(1, min(n_seeds, remaining_seeds, len(cc)))
        seeds_per_cc.append(n_seeds)
        remaining_seeds -= n_seeds
        if remaining_seeds <= 0:
            break

    for cc, n_seeds in zip(ccs, seeds_per_cc):
        cc_list = list(cc)
        random.shuffle(cc_list)
        seeds.extend(cc_list[:n_seeds])

    all_ccs = list(nx.connected_components(register_G))
    n_dropped = len(all_ccs) - len(ccs)
    if n_dropped > 0:
        print(f">>> Dropped {n_dropped} connected components with < {min_cc_size} nodes.")
    print(f">>> Kept {len(ccs)} CCs ({total_cc_nodes} nodes). "
          f"Seed allocation: {list(zip([len(cc) for cc in ccs], seeds_per_cc))}")

    # BFS round-robin growth
    assigned = {}                          # node -> partition index
    partition_nodes = [[] for _ in range(len(seeds))]
    queues = []
    for pi, seed in enumerate(seeds):
        assigned[seed] = pi
        partition_nodes[pi].append(seed)
        queues.append([seed])              # BFS frontier

    while any(queues):
        for pi in range(len(queues)):
            if not queues[pi]:
                continue
            next_frontier = []
            for node in queues[pi]:
                for neighbor in register_G.neighbors(node):
                    if neighbor not in assigned:
                        assigned[neighbor] = pi
                        partition_nodes[pi].append(neighbor)
                        next_frontier.append(neighbor)
            queues[pi] = next_frontier

    # Build subgraph objects; drop empty partitions
    partitions = []
    for pnodes in partition_nodes:
        if len(pnodes) >= 2:
            partitions.append(register_G.subgraph(pnodes).copy())
        # Don't merge since time spent merging may not be worth it, 
        # and it can cause partitions to become unbalanced (one giant + many singletons)
        # elif len(pnodes) == 1:
        #     # Try to merge singleton into a neighbouring partition
        #     solo = pnodes[0]
        #     merged = False
        #     for pp in partitions:
        #         for ppn in pp.nodes:
        #             if register_G.has_edge(solo, ppn):
        #                 new_nodes = set(pp.nodes) | {solo}
        #                 idx_pp = partitions.index(pp)
        #                 partitions[idx_pp] = register_G.subgraph(new_nodes).copy()
        #                 merged = True
        #                 break
        #         if merged:
        #             break

    # Handle nodes from disconnected components that were never reached
    # (isolated nodes are dropped)

    # Fallback: if no valid partitions, use the largest connected component
    if not partitions:
        largest_cc = max(nx.connected_components(register_G), key=len)
        if len(largest_cc) >= 2:
            partitions.append(register_G.subgraph(largest_cc).copy())
    if not partitions:
        return list(register_G.nodes)[:N_nodes]

    n_partitions = len(partitions)
    print(f">>> Created {n_partitions} random partitions "
          f"(requested N_starts={N_starts}).")
    end = time.time()
    print(f">>> Partitioning took {end - start:.2f} seconds.")
    # ------------------------------------------------------------------
    # Step 2: allocate N_nodes proportionally (min 2 per partition)
    # ------------------------------------------------------------------
    # If we cannot give 2 nodes to every partition, keep the largest ones
    if 2 * n_partitions > N_nodes:
        partitions.sort(key=lambda p: len(p.nodes), reverse=True)
        max_partitions = N_nodes // 2
        partitions = partitions[:max_partitions]
        n_partitions = len(partitions)

    total_available = sum(len(p.nodes) for p in partitions)

    allocations = []
    for p in partitions:
        alloc = max(2, round(N_nodes * len(p.nodes) / total_available))
        alloc = min(alloc, len(p.nodes))
        allocations.append(alloc)

    # Adjust so that sum(allocations) == N_nodes
    current_sum = sum(allocations)
    for _ in range(N_nodes * 2):          # bounded iteration
        if current_sum == N_nodes:
            break
        if current_sum < N_nodes:
            caps = [(len(partitions[i].nodes) - allocations[i], i)
                    for i in range(n_partitions)
                    if allocations[i] < len(partitions[i].nodes)]
            if not caps:
                break
            caps.sort(reverse=True)
            allocations[caps[0][1]] += 1
            current_sum += 1
        else:
            shrink = [(allocations[i], i)
                      for i in range(n_partitions)
                      if allocations[i] > 2]
            if not shrink:
                break
            shrink.sort(reverse=True)
            allocations[shrink[0][1]] -= 1
            current_sum -= 1
    # ------------------------------------------------------------------
    # Step 3: sample from each partition using farthest-point sampling
    # ------------------------------------------------------------------
    all_sampled = []
    n_trials_cfg  = 512
    top_k_cfg     = 32

    for p_idx, (partition, alloc) in enumerate(zip(partitions, allocations)):
        partition_comms = {n: node_community.get(n, 0) for n in partition.nodes}

        # ---- Build a Steiner-tree subgraph for this partition ----
        # 1. Pick one representative node per community in this partition
        comm_to_nodes = {}
        for n in partition.nodes:
            c = partition_comms[n]
            if c not in comm_to_nodes:
                comm_to_nodes[c] = []
            comm_to_nodes[c].append(n)

        terminals = []
        for c, members in comm_to_nodes.items():
            terminals.append(random.choice(members))

        # 2. Build the Steiner tree connecting the terminals
        #    Use nx.approximation.steiner_tree which returns a subgraph
        #    of `partition` that spans all terminal nodes.
        if len(terminals) >= 2:
            try:
                steiner_sub = nx.approximation.steiner_tree(partition, terminals, weight="common_pts")
            except (nx.NetworkXError, nx.NodeNotFound):
                # Terminals may not all be in the same CC; fall back to
                # the largest CC that contains the most terminals
                steiner_sub = partition
        else:
            steiner_sub = partition

        # 3. If the Steiner tree has fewer nodes than the allocation,
        #    expand it by adding neighbours from the partition until we
        #    have enough candidates for beam search.
        steiner_nodes = set(steiner_sub.nodes)
        if len(steiner_nodes) < alloc:
            frontier = []
            for sn in steiner_nodes:
                for nb in partition.neighbors(sn):
                    if nb not in steiner_nodes:
                        frontier.append(nb)
            random.shuffle(frontier)
            for nb in frontier:
                steiner_nodes.add(nb)
                if len(steiner_nodes) >= max(alloc, min(2 * alloc, len(partition.nodes))):
                    break
            steiner_sub = partition.subgraph(steiner_nodes).copy()

        # Restrict community map to steiner subgraph nodes
        steiner_comms = {n: partition_comms[n] for n in steiner_sub.nodes}
        print(f">>>>> Partition {p_idx}: Steiner subgraph has {len(steiner_sub.nodes)} nodes ")

        sampled_lists, sampled_comms_lists = beam_farthest_point_sampling(
            steiner_sub, alloc, steiner_comms,
            max_depth=max_depth,
            n_trials=n_trials_cfg,
            top_k=top_k_cfg,
        )

        # Pick one random result from the top-k candidates
        choice_idx = random.randint(0, len(sampled_lists) - 1)
        sampled = sampled_lists[choice_idx]

        # Safety: guarantee sampled nodes form a connected subgraph
        if len(sampled) > 1:
            sub = partition.subgraph(sampled)
            if not nx.is_connected(sub):
                largest_cc = max(nx.connected_components(sub), key=len)
                sampled = list(largest_cc)

        all_sampled.extend(sampled)
        print(f">>>   Partition {p_idx}: sampled {len(sampled)}/{len(partition.nodes)} nodes "
              f"({len(sampled_comms_lists[choice_idx])} communities)")

    # ------------------------------------------------------------------
    # Step 4: if we fell short of N_nodes (due to partition constraints),
    #         greedily add neighbours from the full graph while keeping
    #         connectivity within each connected component's sample.
    # ------------------------------------------------------------------
    if len(all_sampled) < N_nodes:
        for _ in range(N_nodes - len(all_sampled)):
            frontier = set()
            for node in all_sampled:
                for neighbor in register_G.neighbors(node):
                    if neighbor not in all_sampled:
                        frontier.add(neighbor)
            if not frontier:
                break
            
            new_nodes = np.random.choice(list(frontier), size=min(N_nodes-len(all_sampled), len(frontier)), replace=False).tolist()
            all_sampled.extend(new_nodes)
            # sampled_set

            # scored = []
            # for c in frontier:
            #     ccomm = node_community.get(c, 0)
            #     new_comm = 1 if ccomm not in {node_community.get(s, 0) for s in all_sampled} else 0
            #     best_dist = max(
            #         (register_G[c][s].get("distance", 1.0) for s in all_sampled if register_G.has_edge(c, s)),
            #         default=1.0,
            #     )
            #     scored.append((new_comm, best_dist, c))
            # scored.sort(reverse=True)
            # chosen = scored[0][2]
            # all_sampled.append(chosen)
            # sampled_set.add(chosen)

    print(f">>> Total sampled: {len(all_sampled)} nodes from {n_partitions} partitions.")

    return all_sampled

def build_image_graph(pairs, matches=None):
    G = nx.Graph()
    for idx, pair in enumerate(pairs):
        img_a, img_b = pair
        G.add_edge(img_a, img_b)
        if matches is not None:
            G[img_a][img_b]['weight'] = matches[idx]
    return G


def export_graph_ply(register_G, sampled_nodes, save_path, pos=None, partition_labels=None):
    """
    Export register_G and sampled nodes to a .ply file for 3D visualization.

    Sampled nodes are rendered as small octahedra (6 vertices, 8 triangular
    faces each) so they appear visually larger.  Non-sampled nodes are single
    vertices (tiny points).  Edges are written as line segments.

    Color scheme
    ------------
    - Non-sampled nodes: gray (180, 180, 180)
    - Sampled nodes: colored by partition if *partition_labels* is provided,
      otherwise bright red (255, 50, 50).
    - Edges: blue (between sampled), light-blue (bridging), gray (background).

    Parameters
    ----------
    register_G : nx.Graph
        Full registration graph.
    sampled_nodes : list
        List of sampled node IDs (subset of register_G.nodes).
    save_path : str or Path
        Output .ply file path.
    pos : dict, optional
        Pre-computed 3D positions  {node: (x,y,z)}.  If None a spring
        layout is computed automatically.
    partition_labels : dict, optional
        Mapping from node -> partition index for sampled nodes.
        Used to color different partitions differently.
    """
    save_path = str(save_path)
    nodes = list(register_G.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    sampled_set = set(sampled_nodes)

    # 3D spring layout ---------------------------------------------------
    if pos is None:
        pos = nx.spring_layout(register_G, dim=3, seed=42,
                               k=1.0 / max(1, len(nodes) ** 0.3),
                               iterations=100)

    # Collect graph edges -------------------------------------------------
    edges = []
    for u, v in register_G.edges():
        if u in node_to_idx and v in node_to_idx:
            edges.append((node_to_idx[u], node_to_idx[v]))

    # Partition color palette (up to 20 distinct colors) ------------------
    palette = [
        (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
        (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
        (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
        (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
        (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
    ]

    # Assign per-node color -----------------------------------------------
    node_colors = {}
    for node in nodes:
        if node in sampled_set:
            if partition_labels is not None and node in partition_labels:
                pidx = partition_labels[node] % len(palette)
                node_colors[node] = palette[pidx]
            else:
                node_colors[node] = (255, 50, 50)
        else:
            node_colors[node] = (180, 180, 180)

    # Octahedron geometry for sampled nodes --------------------------------
    # 6 unit-offset directions; we scale by *radius* later.
    octa_offsets = np.array([
        [ 1,  0,  0],
        [-1,  0,  0],
        [ 0,  1,  0],
        [ 0, -1,  0],
        [ 0,  0,  1],
        [ 0,  0, -1],
    ], dtype=float)
    # 8 triangular faces (indices into the 6 offset vertices)
    octa_faces = [
        (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
        (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
    ]

    # Compute octahedron radius from the layout bounding box
    all_pos = np.array([pos[n] for n in nodes])
    bbox_diag = np.linalg.norm(all_pos.max(axis=0) - all_pos.min(axis=0))
    radius = bbox_diag * 0.001  # 2 % of bounding-box diagonal

    # --- Build vertex / face lists ----------------------------------------
    # Layout: first len(nodes) vertices are the node-center points (used for
    # edges), then 6 extra vertices per sampled node for the octahedra.
    vert_lines = []   # (x, y, z, r, g, b)
    face_lines = []   # list of (i0, i1, i2)

    # 1) Node-center vertices (one per graph node)
    for node in nodes:
        x, y, z = pos[node]
        r, g, b = node_colors[node]
        vert_lines.append((x, y, z, r, g, b))

    # 2) Octahedron vertices + faces for each sampled node
    for node in sampled_nodes:
        if node not in node_to_idx:
            continue
        cx, cy, cz = pos[node]
        r, g, b = node_colors[node]
        base_idx = len(vert_lines)  # first octa-vertex index for this node
        for dx, dy, dz in octa_offsets:
            vert_lines.append((cx + dx * radius,
                               cy + dy * radius,
                               cz + dz * radius, r, g, b))
        for f0, f1, f2 in octa_faces:
            face_lines.append((base_idx + f0, base_idx + f1, base_idx + f2))

    n_verts = len(vert_lines)
    n_faces = len(face_lines)
    n_edges = len(edges)

    # --- Write PLY --------------------------------------------------------
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {n_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write(f"element edge {n_edges}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Vertices
        for x, y, z, r, g, b in vert_lines:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

        # Faces (octahedra for sampled nodes)
        for f0, f1, f2 in face_lines:
            f.write(f"3 {f0} {f1} {f2}\n")

        # Edges (reference the first len(nodes) center-vertices)
        for i1, i2 in edges:
            n1, n2 = nodes[i1], nodes[i2]
            if n1 in sampled_set and n2 in sampled_set:
                f.write(f"{i1} {i2} 50 50 255\n")
            elif n1 in sampled_set or n2 in sampled_set:
                f.write(f"{i1} {i2} 150 200 255\n")
            else:
                f.write(f"{i1} {i2} 210 210 210\n")

    n_sampled = len([n for n in sampled_nodes if n in node_to_idx])
    print(f">>> Exported graph ({len(nodes)} nodes [{n_sampled} as octahedra], "
          f"{n_edges} edges, {n_faces} faces) to {save_path}")

def debug_steiner_tree_sampling(register_G, save_dir, max_depth=10,
                                N_nodes=24, idx=0, N_starts=1):
    """
    Run steiner_tree_sampling and export the result + full graph to both
    .ply and .obj for visual debugging.

    Files written
    -------------
    - ``<save_dir>/graph_sampled.ply``
    - ``<save_dir>/graph_sampled.obj``  (+ ``.mtl``)

    Returns the list of sampled nodes.
    """
    os.makedirs(save_dir, exist_ok=True)

    sampled = steiner_tree_sampling(register_G, max_depth=max_depth,
                                    N_nodes=N_nodes, idx=idx,
                                    N_starts=N_starts)

    # Build partition labels by re-running BFS with the same seeds logic
    # (approximate: assign each sampled node to the partition whose seed
    #  it is closest to in graph distance).
    partition_labels = {}
    if N_starts > 1 and len(sampled) > 0:
        # Simple heuristic: chunk the sampled list by the order they appear
        # (they are already grouped by partition from steiner_tree_sampling)
        chunk_size = max(1, len(sampled) // N_starts)
        for i, node in enumerate(sampled):
            partition_labels[node] = i // chunk_size

    ply_path = os.path.join(save_dir, "graph_sampled.ply")

    export_graph_ply(register_G, sampled, ply_path,
                     partition_labels=partition_labels)

    return sampled