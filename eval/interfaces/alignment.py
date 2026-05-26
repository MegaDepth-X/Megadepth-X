import torch
from torch import nn
from torch.optim import Adam, LBFGS


# ==============================================================
# --- Utility functions
# ==============================================================

def apply_sim3(points, s, R, T):
    """Apply Sim3 transform: X' = s * (X @ R^T) + T."""
    return s * (points @ R.transpose(-1, -2)) + T


def umeyama_sim3_torch(src, tgt, with_scale=True, eps=1e-9):
    """
    Torch implementation of Umeyama Sim3 estimation.
    src,tgt: (N,3)
    Returns: scale (float), R (3x3), T (3,)
    """
    assert src.shape == tgt.shape
    mu_src = src.mean(dim=0, keepdim=True)
    mu_tgt = tgt.mean(dim=0, keepdim=True)
    src_c = src - mu_src
    tgt_c = tgt - mu_tgt

    cov = (tgt_c.T @ src_c) / src.shape[0]
    U, S, Vt = torch.linalg.svd(cov)
    R = U @ Vt
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        var_src = (src_c ** 2).sum() / src.shape[0]
        scale = S.sum() / (var_src + eps)
    else:
        scale = torch.tensor(1.0, device=src.device)
    T = mu_tgt.squeeze(0) - scale * (mu_src.squeeze(0) @ R.T)
    return scale, R, T


# ==============================================================
# --- Iterative alignment (torch)
# ==============================================================

def iterative_sim3_align_torch(src_pts,
                               tgt_pts,
                               src_cams=None,
                               tgt_cams=None,
                               max_iters=20,
                               nn_max_dist=None,
                               mutual=True,
                               w_points=1.0,
                               w_cams=2.0,
                               robust_loss='huber',  # Use 'huber' or None for loss
                               device='cuda'):
    """
    Torch-based iterative alignment with NN matching using torch.cdist.
    Returns: s,R,T
    """

    src_pts = src_pts.to(device)
    tgt_pts = tgt_pts.to(device).double()
    if src_cams is not None:
        src_cams = src_cams.to(device)
        tgt_cams = tgt_cams.to(device)

    s = torch.tensor(1.0, device=device)
    R = torch.eye(3, device=device)
    T = torch.zeros(3, device=device)

    for it in range(max_iters):
        src_transformed = apply_sim3(src_pts, s, R, T).double()

        # Compute nearest neighbors
        dists = torch.cdist(src_transformed, tgt_pts)
        tgt_idx = dists.argmin(dim=1)
        src2tgt_dist = dists.min(dim=1).values

        if mutual:
            dists_rev = torch.cdist(tgt_pts, src_transformed)
            src_idx_rev = dists_rev.argmin(dim=1)
            mutual_mask = torch.zeros_like(src2tgt_dist, dtype=torch.bool)
            for i in range(src_pts.shape[0]):
                ti = tgt_idx[i]
                if src_idx_rev[ti] == i:
                    mutual_mask[i] = True
            mask = mutual_mask
        else:
            mask = torch.ones_like(src2tgt_dist, dtype=torch.bool)

        if nn_max_dist is not None:
            mask &= (src2tgt_dist <= nn_max_dist)

        if mask.sum() < 3:
            print(f"[Iter {it}] insufficient matches ({mask.sum().item()})")
            break

        src_m = src_pts[mask]
        tgt_m = tgt_pts[tgt_idx[mask]]

        # Append camera correspondences
        if src_cams is not None:
            src_all = torch.cat([w_points * src_m, w_cams * src_cams], dim=0)
            tgt_all = torch.cat([w_points * tgt_m, w_cams * tgt_cams], dim=0)
        else:
            src_all, tgt_all = src_m, tgt_m

        if robust_loss == 'huber':
            delta = 0.1
            residuals = torch.norm(src_all - tgt_all, dim=-1)
            weights = torch.where(residuals < delta, 1.0, delta / residuals)
            src_all = src_all * weights.unsqueeze(1)
            tgt_all = tgt_all * weights.unsqueeze(1)

        s_new, R_new, T_new = umeyama_sim3_torch(src_all, tgt_all, with_scale=True)

        # Check convergence
        rot_delta = torch.acos(torch.clamp(((R_new @ R.T).trace() - 1) / 2, -1, 1))
        scale_delta = torch.abs(s_new - s)
        print(f"[Iter {it}] matches={mask.sum().item()}, rot_delta={rot_delta:.3e}, scale_delta={scale_delta:.3e}")

        s, R, T = s_new, R_new, T_new

        if rot_delta < 1e-6 and scale_delta < 1e-6:
            print(f"Converged at iteration {it}")
            break

    return s, R, T


# ==============================================================
# --- Robust refinement (torch)
# ==============================================================


class Sim3Refiner(nn.Module):
    def __init__(self, s0=1.0, R0=None, T0=None, device='cuda'):
        super().__init__()

        self.log_s = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(s0))), device=device))

        if R0 is None:
            self.rotvec = nn.Parameter(torch.zeros(3, device=device))
        else:
            # convert rotation matrix to rotation vector
            trace = R0.trace()
            cos_theta = torch.clamp((trace - 1) / 2, -1 + 1e-6, 1 - 1e-6)
            theta = torch.acos(cos_theta)
            if theta < 1e-8:
                rv = torch.zeros(3, device=device)
            else:
                rv = theta / (2 * torch.sin(theta)) * torch.stack([
                    R0[2, 1] - R0[1, 2],
                    R0[0, 2] - R0[2, 0],
                    R0[1, 0] - R0[0, 1],
                ])
            self.rotvec = nn.Parameter(rv)

        if T0 is None:
            self.T = nn.Parameter(torch.zeros(3, device=device))
        else:
            self.T = nn.Parameter(T0.clone().to(device))

    def forward(self, src):
        s = torch.exp(self.log_s)
        theta = torch.norm(self.rotvec) + 1e-8
        k = self.rotvec / theta

        K = torch.zeros((3, 3), device=src.device)
        K[0, 1], K[0, 2] = -k[2], k[1]
        K[1, 0], K[1, 2] = k[2], -k[0]
        K[2, 0], K[2, 1] = -k[1], k[0]

        R = torch.eye(3, device=src.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        T = self.T

        src_trans = s * (src @ R.T) + T
        return src_trans, s, R, T


def refine_sim3_torch(src_pts, tgt_pts, src_cams=None, tgt_cams=None,
                      s0=1.0, R0=None, T0=None,
                      w_points=1.0, w_cams=2.0,
                      iters=200, lr=1e-2, loss='huber', device='cuda'):
    """Robust gradient-based refinement on GPU."""
    src_pts = src_pts.to(device)
    tgt_pts = tgt_pts.to(device)
    if src_cams is not None:
        src_cams = src_cams.to(device)
        tgt_cams = tgt_cams.to(device)

    refiner = Sim3Refiner(s0, R0, T0, device=R0.device).to(device)
    optimizer = Adam(refiner.parameters(), lr=lr)

    for it in range(iters):
        optimizer.zero_grad()
        src_all = [w_points * src_pts]
        tgt_all = [w_points * tgt_pts]
        if src_cams is not None:
            src_all.append(w_cams * src_cams)
            tgt_all.append(w_cams * tgt_cams)
        src_cat = torch.cat(src_all)
        tgt_cat = torch.cat(tgt_all)
        pred, s, R, T = refiner(src_cat)
        res = pred - tgt_cat

        if loss == 'huber':
            delta = 0.01
            abs_res = torch.norm(res, dim=1)
            l = torch.where(abs_res < delta, 0.5 * abs_res ** 2, delta * (abs_res - 0.5 * delta))
            loss_val = l.mean()
        else:
            loss_val = (res ** 2).sum(dim=1).mean()
        loss_val.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"[Refine {it}] loss={loss_val.item():.4e}, scale={torch.exp(refiner.log_s).item():.4f}")

    with torch.no_grad():
        _, s_final, R_final, T_final = refiner(src_pts)
    return s_final, R_final, T_final


# ==============================================================
# --- Unified wrapper
# ==============================================================

def farthest_point_sampling_torch(points, num_samples):
    """
    Farthest Point Sampling (FPS) implementation in PyTorch.
    Args:
        points (torch.Tensor): Input point cloud of shape (N, 3).
        num_samples (int): Number of points to sample.
    Returns:
        torch.Tensor: Indices of the sampled points of shape (num_samples,).
    """
    N, _ = points.shape
    sampled_indices = torch.zeros(num_samples, dtype=torch.long, device=points.device)
    distances = torch.full((N,), float('inf'), device=points.device)

    # Randomly select the first point
    farthest_index = torch.randint(0, N, (1,), device=points.device).item()
    for i in range(num_samples):
        sampled_indices[i] = farthest_index
        centroid = points[farthest_index].unsqueeze(0)  # (1, 3)
        dist = torch.norm(points - centroid, dim=1)  # (N,)
        distances = torch.min(distances, dist)
        farthest_index = torch.argmax(distances).item()

    return sampled_indices

def align_and_refine_torch(src_pts, tgt_pts, src_cams=None, tgt_cams=None,
                           iterative_kwargs=None, refine_kwargs=None,
                           device='cuda'):
    iterative_kwargs = iterative_kwargs or {}
    refine_kwargs = refine_kwargs or {}

    # # first get all the points inside unit bbox
    # bbox_min_src = src_pts.min(dim=0)[0]
    # bbox_max_src = src_pts.max(dim=0)[0]
    # src_unit_bbox_mask = (src_pts >= bbox_min_src) & (src_pts <= bbox_max_src)
    # bbox_min_tgt = tgt_pts.min(dim=0)[0]
    # bbox_max_tgt = tgt_pts.max(dim=0)[0]
    # tgt_unit_bbox_mask = (tgt_pts >= bbox_min_tgt) & (tgt_pts <= bbox_max_tgt)
    # unified_bbox_mask = torch.logical_and(src_unit_bbox_mask, tgt_unit_bbox_mask)
    # src_pts = src_pts[unified_bbox_mask]
    # tgt_pts = tgt_pts[unified_bbox_mask]

    # path 1, use random points to get initial alignment
    if len(src_pts) > 10000:
        # indices = torch.randperm(len(src_pts), device=device)[:10000]
        # src_pts_path1 = src_pts[indices]
        # tgt_pts_path1 = tgt_pts[indices]
        print("Using FPS to subsample points for path 1")
        fps_indices = farthest_point_sampling_torch(src_pts, 10000)
        src_pts_path1 = src_pts[fps_indices]
        tgt_pts_path1 = tgt_pts[fps_indices]
    else:
        src_pts_path1 = src_pts
        tgt_pts_path1 = tgt_pts

    s_init, R_init, T_init = iterative_sim3_align_torch(
        src_pts_path1, tgt_pts_path1, src_cams=src_cams, tgt_cams=tgt_cams, device=device, **iterative_kwargs)
    print("Iterative path 1 done. Refining...")
    with torch.enable_grad():
        s_opt, R_opt, T_opt = refine_sim3_torch(
            src_pts_path1, tgt_pts_path1, src_cams, tgt_cams,
            s0=s_init.item(), R0=R_init, T0=T_init,
            device=device, **refine_kwargs)

    # # path 2, use best matches to get final alignment
    # src_pts_path1_final = src_pts @ R_opt.T * s_opt + T_opt
    # errors = torch.norm(src_pts_path1_final-tgt_pts, dim=-1)
    # error_rank_idx = torch.argsort(errors)
    # inlier_idx = error_rank_idx[:10000]
    # src_pts_path2 = src_pts[inlier_idx]
    # tgt_pts_path2 = tgt_pts[inlier_idx]
    # s_init, R_init, T_init = iterative_sim3_align_torch(
    #     src_pts_path2, tgt_pts_path2, src_cams=src_cams, tgt_cams=tgt_cams, device=device, **iterative_kwargs)
    # print("Iterative path 2 done. Refining...")
    # with torch.enable_grad():
    #     s_opt, R_opt, T_opt = refine_sim3_torch(
    #         src_pts_path2, tgt_pts_path2, src_cams, tgt_cams,
    #         s0=s_init.item(), R0=R_init, T0=T_init,
    #         device=device, **refine_kwargs)
    
    return s_opt, R_opt, T_opt