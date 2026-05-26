import os
import random
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
from PIL import Image, ImageOps, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from tqdm import tqdm
from collections import deque

from.alignment import align_and_refine_torch

def greedy_maximum_leaf_spanning_tree(G, N_nodes, step=1, weight_decay_thres=0.1):
    """
    Greedy approximation of Maximum Leaf Spanning Tree.
    Nodes are colored:
        - 'red': already processed (added and expanded)
        - 'blue': added as leaf (but not expanded)
        - 'green': not yet added
    """
    # Initialize sets
    color = {node: 'green' for node in G.nodes}
    T = nx.Graph()
    T_skeleton = nx.Graph()

    # Step 1: Pick node with max degree
    v = max(G.degree, key=lambda x: x[1])[0]
    color[v] = 'red'
    T.add_node(v)
    T_skeleton.add_node(v)

    # Step 2: Add unmarked (green) neighbors of v as blue nodes (leaves)
    def probe(_v):
        if step == 0:
            for u in G.neighbors(_v):
                if color[u] == 'green':
                    color[u] = 'blue'
                    T.add_edge(_v, u)
                    T[_v][u]['weight'] = G[_v][u]['weight']
        else:
            for u in G.neighbors(_v):
                T.add_edge(_v, u)
                T[_v][u]['weight'] = G[_v][u]['weight']
            lengths = nx.single_source_shortest_path_length(G, _v, cutoff=(step+1))
            result_nodes = [node for node, dist in lengths.items() if dist < (step+1)]
            for u in result_nodes:
                if color[u] == 'green':
                    color[u] = 'yellow'
            result_nodes = [node for node, dist in lengths.items() if dist == (step+1)]
            for u in result_nodes:
                if color[u] == 'green':
                    color[u] = 'blue'
    probe(v)
    while True:
        # Step 3: Select blue node with most green neighbors
        candidates = []
        for node in G.nodes:
            if color[node] == 'blue':
                unmarked_neighbor = sum(1 for n in G.neighbors(node) if color[n] == 'green')
                if unmarked_neighbor > 0:
                    candidates.append((node, unmarked_neighbor))
        if not candidates:
            break

        # Choose blue node with most green neighbors
        v, max_unmarked = max(candidates, key=lambda x: x[1])
        color[v] = 'red'
        T_skeleton.add_node(v)
        flag = False
        for neighbor in G.neighbors(v):
            if color[neighbor] == 'red':
                T_skeleton.add_edge(v, neighbor)
                flag = True
        if step == 0:
            assert flag
        if len(T_skeleton.nodes) == N_nodes:
            break

        # Add unmarked neighbors of v to T and color them blue
        probe(v)

    return T, T_skeleton

# def chunk_by_tree(G, step=2, window_size=4, weight_decay_thres=0.1):
#     """
#     Chunk graph G into N_chunks using greedy maximum leaf spanning tree.
#     """
#     nodes = list(G.nodes)
#     T, T_skeleton = greedy_maximum_leaf_spanning_tree(G, len(nodes), step=step, weight_decay_thres=weight_decay_thres)
        
#     # inside each chunk, we will perform BFS to get step + 1 neighborhood
#     chunks = []
#     joint_nodes = list(T_skeleton.nodes())
#     for center in joint_nodes:
#         nodes_in_chunk = set()
#         queue = [(center, 0, 1)]
#         while queue:
#             current_node, depth, curr_decay = queue.pop(0)
#             if depth > window_size:
#                 continue
#             if curr_decay <= weight_decay_thres:
#                 continue
#             if current_node not in nodes_in_chunk:
#                 nodes_in_chunk.add(current_node)
#                 for neighbor in G.neighbors(current_node):
#                     weight = G[current_node][neighbor]['weight']
#                     queue.append((neighbor, depth + 1, curr_decay * weight))
#         chunks.append(list(nodes_in_chunk))
#     return chunks

def skeletal_clustering(
    G,
    tau=0.7,           # decay threshold for connectivity
    step=1,            # BFS expansion window size
    window_size=3,     # max hops from center per cluster
    overlap=True,      # enforce at least one overlapping node between clusters
    cover_all=True,    # ensure all nodes are covered
    min_weight=0.2,    # ignore weak edges
    max_clusters=None, # optional limit on number of clusters
    max_n_nodes=None   # optional limit on number of nodes per cluster
):
    # ---- 1. pick candidate centers (by degree centrality)
    degree_centrality = nx.degree_centrality(G)
    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: -x[1])
    centers = [n for n, _ in sorted_nodes]
    # nodes = list(G.nodes)
    # _, T_skeleton = greedy_maximum_leaf_spanning_tree(G, len(nodes), step=step)
    # centers = list(T_skeleton.nodes())
    if max_clusters:
        centers = centers[:max_clusters]

    clusters = []
    visited = set()

    # ---- 2. grow clusters iteratively
    for c in centers:
        if c in visited and not overlap:
            continue

        cluster = set([c])
        q = deque([(c, 1.0, 0)])  # node, accumulated weight

        while q:
            if not q:
                break
            node, acc, depth = q.popleft()
            for nbr, attr in G[node].items():
                w = attr.get('weight', 1.0)
                if w < min_weight:
                    continue
                new_acc = acc * w
                if new_acc < tau:
                    continue
                if depth >= window_size:
                    continue
                if nbr not in cluster:
                    cluster.add(nbr)
                    q.append((nbr, new_acc, depth + 1))
                    if max_n_nodes and len(cluster) >= max_n_nodes:
                        break

        clusters.append(cluster)
        visited.update(cluster)

    # ---- 3. ensure coverage
    if cover_all:
        remaining = set(G.nodes()) - set().union(*clusters)
        if remaining:
            # assign remaining nodes to nearest cluster (by strongest connection)
            for node in remaining:
                best_c = None
                best_w = -1
                for i, cluster in enumerate(clusters):
                    for nbr in G[node]:
                        if nbr in cluster:
                            w = G[node][nbr].get('weight', 1.0)
                            if w > best_w:
                                best_w = w
                                best_c = i
                if best_c is not None:
                    clusters[best_c].add(node)
                else:
                    # no nearby cluster? start a singleton cluster
                    clusters.append(set([node]))

    # ---- 4. enforce overlap
    if overlap:
        # ensure each cluster overlaps with another
        for i in range(len(clusters)):
            has_overlap = any(clusters[i] & clusters[j] for j in range(len(clusters)) if j != i)
            if not has_overlap:
                # merge with nearest cluster
                best_j, best_w = None, -1
                for j in range(len(clusters)):
                    if i == j: continue
                    common_edges = 0
                    for n1 in clusters[i]:
                        for n2 in clusters[j]:
                            if G.has_edge(n1, n2):
                                common_edges += G[n1][n2].get('weight', 1.0)
                    if common_edges > best_w:
                        best_w = common_edges
                        best_j = j
                if best_j is not None:
                    clusters[i].update(clusters[best_j])
    
    # ---- 5. optional: deduplicate overlapping identical clusters
    unique_clusters = []
    for c in clusters:
        if not any(c == uc for uc in unique_clusters):
            unique_clusters.append(list(c))
        # or any other clusters that includes current cluster
        if not any (c.issubset(set(uc)) for uc in clusters):
            unique_clusters.append(list(c))
    print(f"Skeletal clustering produced {len(unique_clusters)} clusters.")
    return unique_clusters

def greedy_highweight_connected_graph(W, top_k=1, verbose=True):
    """
    Build a sparse graph ensuring each node connects by its top-K edges
    and the graph is globally connected using an iterative method.

    Args:
        W (np.ndarray): symmetric weight matrix (NxN)
        top_k (int): number of strongest local edges per node to keep initially
        verbose (bool): print connectivity info

    Returns:
        G (nx.Graph): resulting connected graph
    """
    N = W.shape[0]
    np.fill_diagonal(W, 0)
    W = np.maximum(W, W.T)

    G = nx.Graph()
    G.add_nodes_from(range(N))

    # --- Step 1: Each node keeps top-k local edges
    for i in range(N):
        neighbors = np.argsort(W[i])[::-1][:top_k]
        for j in neighbors:
            G.add_edge(i, j, weight=W[i, j])

    if verbose:
        print(f"Initial graph: {nx.number_connected_components(G)} components")

    # --- Step 2: Iteratively connect components using strongest inter-cluster edges
    while not nx.is_connected(G):
        # Find connected components
        components = list(nx.connected_components(G))
        comp_map = {node: idx for idx, comp in enumerate(components) for node in comp}

        best_edge = None
        best_weight = -np.inf

        # Search for strongest edge connecting two different components
        for i in range(N):
            for j in range(i + 1, N):
                if comp_map[i] != comp_map[j] and W[i, j] > best_weight:
                    best_edge = (i, j)
                    best_weight = W[i, j]

        if best_edge is None:
            raise RuntimeError("No edge can connect remaining components!")
        
        G.add_edge(*best_edge, weight=best_weight)

        if verbose:
            print(f"Added edge {best_edge} (w={best_weight:.4f}), "
                  f"components={nx.number_connected_components(G)}")

    return G

def chunk_batch(images, global_features, graph_top_k=1, step=2, window_size=4, weight_decay_thres=0.1, max_clusters=None, max_n_nodes=None):
    # build graph by global_features
    normalized_features = global_features / (np.linalg.norm(global_features, axis=1, keepdims=True)+1.e-6)  # (N, D)
    similarity_matrix = np.dot(normalized_features, normalized_features.T)  # (N, N)
    similarity_matrix = (similarity_matrix + 1.0) / 2.0  # scale to [0, 1]
    np.fill_diagonal(similarity_matrix, -1)
    G = greedy_highweight_connected_graph(similarity_matrix, top_k=graph_top_k, verbose=False)
    # chunk_list = chunk_by_tree(G, step=step, window_size=window_size, weight_decay_thres=weight_decay_thres)
    chunk_list = skeletal_clustering(G, tau=weight_decay_thres, step=step, window_size=window_size, overlap=True, cover_all=True, min_weight=0.0, max_clusters=max_clusters, max_n_nodes=max_n_nodes)
    # assert all chunks are overlapped with at least one other chunk
    for idx, chunk in enumerate(chunk_list):
        has_overlap = any(len(set(chunk) & set(other_chunk)) > 0 for jdx, other_chunk in enumerate(chunk_list) if jdx != idx)
        assert has_overlap, f"Chunk {idx} has no overlap with other chunks!"
    chunked_images = []
    for idx_list in chunk_list:
        chunked_images.append(images[idx_list])
    return chunked_images, chunk_list, G

def merge_predictions_torch(chunked_preds, chunked_indices, N_total, device='cuda'):
    """
    Merge chunked predictions with overlapping indices using ICP alignment on GPU.
    
    Args:
        chunked_preds: List of dicts, each containing 'c2ws' (Nx4x4) and 'points' (NxHxWx3)
        chunked_indices: List of index arrays indicating which global frames each chunk corresponds to
        N_total: Total number of frames
        device: 'cuda' or 'cpu'
    
    Returns:
        merged_preds: Dict with 'c2ws' (N_total x 4x4) and 'points' (N_total x H x W x 3)
    """
    # Initialize output tensors on GPU
    H, W = chunked_preds[0]['points'].shape[1:3]
    merged_c2ws = torch.zeros((N_total, 4, 4), device=device)
    merged_point_maps = torch.zeros((N_total, H, W, 3), device=device)
    merged_confs = torch.zeros((N_total, H, W, 1), device=device)
    vote_counts = torch.zeros(N_total, device=device)
    
    # Reference chunk (first chunk is used as global reference)
    ref_chunk_idx = 0
    ref_indices = chunked_indices[ref_chunk_idx]
    merged_c2ws[ref_indices] = chunked_preds[ref_chunk_idx]['c2ws']
    merged_point_maps[ref_indices] = chunked_preds[ref_chunk_idx]['points']
    merged_confs[ref_indices] = chunked_preds[ref_chunk_idx]['conf']
    vote_counts[ref_indices] = 1
    
    # Process remaining chunks
    for chunk_idx in tqdm(range(1, len(chunked_preds)), desc="Merging chunks"):
        curr_indices = chunked_indices[chunk_idx]
        curr_c2ws = chunked_preds[chunk_idx]['c2ws']
        curr_point_maps = chunked_preds[chunk_idx]['points']
        curr_confs = chunked_preds[chunk_idx]['conf']
        
        # Find overlap with already merged data
        overlap_mask = vote_counts[curr_indices] > 0
        overlap_curr_idx = torch.where(overlap_mask)[0]
        
        if len(overlap_curr_idx) == 0:
            transform = torch.eye(4, device=device)
        else:
            overlap_global_idx = np.asarray(curr_indices)[overlap_curr_idx.cpu().numpy()].tolist()
            
            # Extract point clouds from overlapping regions
            source_pts_list = []
            target_pts_list = []
            source_cam_pts_list = []
            target_cam_pts_list = []
            
            for local_idx, global_idx in zip(overlap_curr_idx.cpu().numpy().tolist(), overlap_global_idx):
                curr_pts = curr_point_maps[local_idx].reshape(-1, 3)
                curr_c2ws_T = curr_c2ws[local_idx][:3, 3]
                valid_curr = torch.isfinite(curr_pts).all(dim=1)
                
                merged_pts = merged_point_maps[global_idx].reshape(-1, 3)
                merged_c2ws_T = merged_c2ws[global_idx][:3, 3]
                valid_merged = torch.isfinite(merged_pts).all(dim=1)
                
                valid = valid_curr & valid_merged
                if valid.sum() > 10:
                    source_pts_list.append(curr_pts[valid])
                    target_pts_list.append(merged_pts[valid])
                source_cam_pts_list.append(curr_c2ws_T)
                target_cam_pts_list.append(merged_c2ws_T)
            
            if len(source_pts_list) > 0:
                source_pts = torch.vstack(source_pts_list)
                target_pts = torch.vstack(target_pts_list)
                source_cam_pts = torch.vstack(source_cam_pts_list)
                target_cam_pts = torch.vstack(target_cam_pts_list)

                # Subsample if too many points
                # if len(source_pts) > 10000:
                #     indices = torch.randperm(len(source_pts), device=device)[:10000]
                #     source_pts = source_pts[indices]
                #     target_pts = target_pts[indices]

                scale, R, T = align_and_refine_torch(
                    source_pts, target_pts, source_cam_pts, target_cam_pts,
                    iterative_kwargs=dict(max_iters=10, mutual=True, w_cams=5.0),
                    refine_kwargs=dict(iters=1000, lr=5e-3, w_cams=5.0),
                    device=device)
                transform = torch.eye(4, device=device)
                transform[:3, :3] = R * scale
                transform[:3, 3] = T
            else:
                transform = torch.eye(4, device=device)
        
        # Apply transformation to current chunk
        for local_idx, global_idx in enumerate(curr_indices):
            aligned_c2w = transform @ curr_c2ws[local_idx]
            # make sure that c2w rotation is SO(3)
            U, _, Vt = torch.linalg.svd(aligned_c2w[:3, :3])
            if torch.det(U @ Vt) < 0:
                Vt[-1, :] *= -1
            aligned_c2w[:3, :3] = U @ Vt  # Ensure rotation matrix is orthogonal
            aligned_c2w[:3, 3] = aligned_c2w[:3, 3]  # Translation remains unchanged
            
            pts = curr_point_maps[local_idx].reshape(-1, 3)
            pts_homo = torch.cat([pts, torch.ones((len(pts), 1), device=device)], dim=1)
            aligned_pts = (transform @ pts_homo.T).T[:, :3]
            aligned_point_map = aligned_pts.reshape(H, W, 3)
            
            if vote_counts[global_idx] > 0:
                weight_existing = vote_counts[global_idx]
                weight_new = 1.0
                merged_c2ws[global_idx] = (merged_c2ws[global_idx] * weight_existing + aligned_c2w * weight_new) / (weight_existing + weight_new)
                merged_point_maps[global_idx] = (merged_point_maps[global_idx] * weight_existing + aligned_point_map * weight_new) / (weight_existing + weight_new)
                merged_confs[global_idx] = (merged_confs[global_idx] * weight_existing + curr_confs[local_idx] * weight_new) / (weight_existing + weight_new)
            else:
                merged_c2ws[global_idx] = aligned_c2w
                merged_point_maps[global_idx] = aligned_point_map
                merged_confs[global_idx] = curr_confs[local_idx]
            
            # make sure that c2w rotation is SO(3)
            U, _, Vt = torch.linalg.svd(merged_c2ws[global_idx][:3, :3])
            if torch.det(U @ Vt) < 0:
                Vt[-1, :] *= -1
            merged_c2ws[global_idx][:3, :3] = U @ Vt  # Ensure rotation matrix is orthogonal
            merged_c2ws[global_idx][:3, 3] = merged_c2ws[global_idx][:3, 3]  # Translation remains unchanged
            
            vote_counts[global_idx] += 1
    
    return {
        'c2ws': merged_c2ws.cpu(),
        'points': merged_point_maps.cpu(),
        'vote_counts': vote_counts.cpu(),
        'conf': merged_confs.cpu(),
    }

def visualize_graph_with_images(G, images, max_edges=100, title="Graph"):
    N = len(images)

    # Compute force-directed layout
    pos = nx.spring_layout(G, weight='weight', seed=42, k=None, iterations=100)

    # Draw edges
    plt.figure(figsize=(48, 48))
    nx.draw_networkx_edges(G, pos, alpha=0.7)

    # Overlay images instead of nodes
    ax = plt.gca()
    for n, (x, y) in pos.items():
        try:
            # img = Image.open(image_paths[n])
            img = Image.fromarray((images[n].permute((1,2,0)).cpu().numpy()*255).astype(np.uint8))
            img.thumbnail((150, 150), Image.Resampling.LANCZOS)
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img),
                (x, y),
                frameon=False,
                pad=0.1,
            )
            ax.add_artist(imagebox)
        except Exception as e:
            plt.text(x, y, f"{n}", fontsize=8, ha='center', va='center')

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    # plt.show()
    # plt.savefig(output_path)
    # Convert the matplotlib figure to a numpy array
    fig = plt.gcf()
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)[..., 1:]
    plt.close()
    return img_array