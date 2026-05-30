import numpy as np

try:
    import torch
except Exception:
    torch = None


def closed_form_inverse_se3(se3, R=None, T=None):
    """Compute the inverse for a batch of SE3 matrices."""
    is_numpy = isinstance(se3, np.ndarray)
    is_torch = torch is not None and torch.is_tensor(se3)

    if not (is_numpy or is_torch):
        raise TypeError("se3 must be a numpy array or torch tensor")

    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be (N,4,4) or (N,3,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        out = np.tile(np.eye(4, dtype=se3.dtype), (len(R), 1, 1))
        out[:, :3, :3] = R_t
        out[:, :3, 3:] = top_right
        return out

    R_t = R.transpose(1, 2)
    top_right = -torch.bmm(R_t, T)
    out = torch.eye(4, dtype=se3.dtype, device=se3.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
    out[:, :3, :3] = R_t
    out[:, :3, 3:] = top_right
    return out
