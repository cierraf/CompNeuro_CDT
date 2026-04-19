# -----------------------------
# training loss helper (configurable)
# -----------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(y_hat: torch.Tensor, y: torch.Tensor, feat_weight: torch.Tensor = None, use_sample_weight:bool=False, **kwargs):
    """
    y_hat: [B,n]
    y:     [B,n] with (r_norm, cos_theta, sin_theta)
    """
    B, D = y.shape

    # -------------------------
    # 1. base loss
    # -------------------------
    loss = F.mse_loss(y_hat, y)

    # -------------------------
    # 2. feature-wise weight
    # -------------------------
    if feat_weight is None:
        if D == 6:
            feat_weight = torch.tensor(
                [1.0, 1.0, 1.2, 1.5, 2.0, 1.5],
                device=y.device
            )
        else:
            feat_weight = torch.ones(D, device=y.device)

    loss = loss * feat_weight.view(1, D)

    if use_sample_weight:
        alpha = kwargs.get("alpha", 1.0)
        sample_weight = 1 + alpha * torch.norm(y, dim=-1, keepdim=True)  # [B, 1]
        loss = loss * sample_weight

    return loss.mean()

def mae_loss(y_hat: torch.Tensor, y: torch.Tensor, feat_weight: torch.Tensor = None, use_sample_weight:bool=False, **kwargs):
    """
    y_hat: [B,n]
    y:     [B,n] with (r_norm, cos_theta, sin_theta)
    """
    B, D = y.shape

    # -------------------------
    # 1. base loss
    # -------------------------
    loss = F.l1_loss(y_hat, y)

    # -------------------------
    # 2. feature-wise weight
    # -------------------------
    if feat_weight is None:
        if D == 6:
            feat_weight = torch.tensor(
                [2.0, 1.0, 1.0, 2.0, 2.5, 1.0],
                device=y.device
            )
        elif D == 2:
            feat_weight = torch.tensor(
                [1.0, 1.5],
                device=y.device
            )
        else:
            feat_weight = torch.ones(D, device=y.device)

    loss = loss * feat_weight.view(1, D)

    if use_sample_weight:
        alpha = kwargs.get("alpha", 1.0)
        sample_weight = 1 + alpha * torch.norm(y, dim=-1, keepdim=True)  # [B, 1]
        loss = loss * sample_weight

    return loss.mean()

def smooth_l1_loss(y_hat: torch.Tensor, y: torch.Tensor, beta: float = 0.5):
    """
    y_hat: [B,n]
    y:     [B,n]
    """
    return F.smooth_l1_loss(y_hat, y, beta=beta)

def weighted_smooth_l1_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    beta: float = 0.5,
    feat_weight: torch.Tensor = None,   # [D]
    use_sample_weight: bool = False,
    alpha: float = 1.0
):
    """
    Parameters
    ----------
    y_hat : [B, D]
    y     : [B, D]

    feat_weight : [D]  (dimension importance)
    use_sample_weight : whether to weight by distance from center
    alpha : strength of sample weighting
    """

    B, D = y.shape

    # -------------------------
    # 1. base loss
    # -------------------------
    loss = F.smooth_l1_loss(y_hat, y, beta=beta, reduction="none")  # [B, D]

    # -------------------------
    # 2. feature-wise weight
    # -------------------------
    if feat_weight is None:
        if D == 6:
            feat_weight = torch.tensor(
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                device=y.device
            )
        elif D == 2:
            feat_weight = torch.tensor(
                # [1.0, 1.5],
                [1.0, 1.0],
                device=y.device
            )
        else:
            feat_weight = torch.ones(D, device=y.device)

    loss = loss * feat_weight.view(1, D)

    if use_sample_weight:
        sample_weight = 1 + alpha * torch.norm(y, dim=-1, keepdim=True)  # [B, 1]
        loss = loss * sample_weight

    return loss.mean()


def grouped_weighted_smooth_l1_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    beta: float = 0.5,
    lambda_pos: float = 1.0,
    lambda_vel: float = 2.0,
    vel_axis_weight: torch.Tensor = torch.tensor([1.0, 1.0, 2.0]),   # [3], for vx vy vz
    # vel_axis_weight: torch.Tensor = None,   # [3], for vx vy vz
):
    """
    y_hat, y: [B, 6] = [x, y, z, vx, vy, vz]
    """

    assert y.shape[-1] == 6, "Expected D=6 for [x,y,z,vx,vy,vz]"

    loss_raw = F.smooth_l1_loss(y_hat, y, beta=beta, reduction="none")  # [B, 6]

    # pos group
    loss_pos = loss_raw[:, :3].mean()

    # vel group
    vel_loss = loss_raw[:, 3:]   # [B, 3]

    if vel_axis_weight is not None:
        vel_axis_weight = vel_axis_weight.to(y.device, y.dtype)
        vel_axis_weight = vel_axis_weight / vel_axis_weight.mean()  # normalize
        vel_loss = vel_loss * vel_axis_weight.view(1, 3)

    loss_vel = vel_loss.mean()

    loss = lambda_pos * loss_pos + lambda_vel * loss_vel
    return loss

def bce_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """
    logit: [B] or [B,1]
    y:     [B] or [B,1], values in {0,1}
    """
    y_hat = y_hat.squeeze(-1)
    y = y.float().squeeze(-1)
    return F.binary_cross_entropy_with_logits(y_hat, y)

def radial_angular_loss(y_hat, y):

    r_hat = y_hat[:,0]
    r = y[:,0]

    theta_hat = y_hat[:,1:]
    theta_hat = theta_hat / (theta_hat.norm(dim=-1, keepdim=True)+1e-6)

    theta = y[:,1:]

    loss_r = F.smooth_l1_loss(r_hat, r)
    loss_theta = F.smooth_l1_loss(theta_hat, theta)

    return loss_r + 0.5 * loss_theta

def polar_xy_loss(y_hat, y):
    r_hat = 3.0 * torch.sigmoid(y_hat[:, 0])
    ang_hat = y_hat[:, 1:]
    ang_hat = ang_hat / (ang_hat.norm(dim=-1, keepdim=True) + 1e-6)

    r = y[:, 0]
    ang = y[:, 1:]

    loss_r = F.smooth_l1_loss(r_hat, r)
    loss_ang = F.smooth_l1_loss(ang_hat, ang)

    x_hat = r_hat * ang_hat[:, 1]   # cos
    y_hat_xy = r_hat * ang_hat[:, 0]  # sin

    x = r * ang[:, 1]
    y_xy = r * ang[:, 0]

    loss_xy = torch.sqrt((x_hat - x) ** 2 + (y_hat_xy - y_xy) ** 2 + 1e-8).mean()

    return loss_r + 0.5 * loss_ang + 0.5 * loss_xy

def direction_loss(y_hat, y, lambda_r=1.0):
    """ Direction-aware loss: angle (via cos similarity) + r MSE """
    r_hat = y_hat[:,0]
    r     = y[:,0]

    dir_hat = y_hat[:,1:3]
    dir_gt  = y[:,1:3]

    dir_hat = F.normalize(dir_hat, dim=1)
    dir_gt  = F.normalize(dir_gt, dim=1)

    cos_sim = (dir_hat * dir_gt).sum(dim=1)

    angle_loss = (1 - cos_sim).mean()
    r_loss = F.mse_loss(r_hat, r)

    return lambda_r * r_loss + angle_loss


# Loss registry: map names -> callables
LOSS_FUNCS = {
    "mse": mse_loss,
    "direction": direction_loss,
    "mae": mae_loss,
    "smooth_l1": smooth_l1_loss,
    "weighted_smooth_l1": weighted_smooth_l1_loss,
    "radial_angular": radial_angular_loss,
    "polar_xy": polar_xy_loss,
    "bce": bce_loss,
    "grouped_weighted_smooth_l1": grouped_weighted_smooth_l1_loss,
}

def _resolve_loss(loss):
    """Resolve a loss spec into a callable."""
    # loss can be: str (key into LOSS_FUNCS), callable, or (str, kwargs_dict)
    if isinstance(loss, tuple):
        name, kwargs = loss
        if isinstance(name, str):
            base = LOSS_FUNCS.get(name)
            if base is None:
                raise KeyError(f"Unknown loss name: {name}")
            return lambda y_hat, y: base(y_hat, y, **(kwargs or {}))
        else:
            raise ValueError("Tuple loss must be (name:str, kwargs:dict)")
    if isinstance(loss, str):
        if loss not in LOSS_FUNCS:
            raise KeyError(f"Unknown loss name: {loss}")
        return LOSS_FUNCS[loss]
    if callable(loss):
        return loss
    raise ValueError("loss must be str, callable, or (str, kwargs)")