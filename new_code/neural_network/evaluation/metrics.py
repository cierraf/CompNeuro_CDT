import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import Dict, Callable

def _metrics_1d(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }

# -----------------------------
# Metrics for (r_norm, cos_theta, sin_theta)
# -----------------------------
@torch.no_grad()
def metrics_r_cos_sin(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable = None,
    angle_deg_thresh: float = 10.0,
    r_abs_thresh: float = 0.05,
) -> Dict[str, float]:

    y_hat = y_hat.float()
    y = y.float()

    # ---------- loss ----------
    if loss_fn is None:
        loss_val = F.mse_loss(y_hat, y).item()
    else:
        loss_val = loss_fn(y_hat, y).item()

    # ---------- r error ----------
    r_hat, r = y_hat[:, 0], y[:, 0]
    r_mae = (r_hat - r).abs().mean().item()

    # ---------- direction ----------
    if y_hat.shape[1] >= 3:
        c_hat, s_hat = y_hat[:, 1], y_hat[:, 2]
        c, s = y[:, 1], y[:, 2]

        norm = torch.sqrt(c_hat**2 + s_hat**2).clamp_min(1e-6)
        c_hat_n = c_hat / norm
        s_hat_n = s_hat / norm

        theta_hat = torch.atan2(s_hat_n, c_hat_n)
        theta = torch.atan2(s, c)

        dtheta = torch.atan2(
            torch.sin(theta_hat - theta),
            torch.cos(theta_hat - theta)
        )

        angle_mae_deg = dtheta.abs().mean().item() * (180.0 / math.pi)

        angle_deg = dtheta.abs() * (180.0 / math.pi)
        angle_acc = (angle_deg <= angle_deg_thresh).float().mean().item()

        joint_acc = (
            (angle_deg <= angle_deg_thresh)
            & ((r_hat - r).abs() <= r_abs_thresh)
        ).float().mean().item()
    else:
        angle_mae_deg = float('nan')
        angle_acc = float('nan')
        joint_acc = float('nan')

    return {
        "loss": loss_val,
        "r_mae": r_mae,
        "angle_mae_deg": angle_mae_deg,
        "angle_acc": angle_acc,
        "joint_acc": joint_acc,
    }

import numpy as np
import torch
import matplotlib.pyplot as plt


# =========================================================
# 1. label -> dim 的配置
# =========================================================
DEFAULT_LABEL_DIMS = {
    "eye_gaze": 2,
    "gaze": 2,
    "pupil": 1,
    "emo_neutral": 1,
    "jacky_ratio": 1,
    "front_face_ratio": 1,
}


def _metrics_1d(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # 避免全常数时 nan
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom < 1e-12:
        r2 = np.nan
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def _resolve_label_dims(y_order, label_dims=None):
    """
    根据 y_order 返回每个 label 的输出维度
    """
    final_dims = {}
    if label_dims is not None:
        final_dims.update(label_dims)

    for name in y_order:
        if name not in final_dims:
            final_dims[name] = DEFAULT_LABEL_DIMS.get(name, 1)

    return final_dims


def _build_slices_from_y_order(y_order, label_dims=None):
    """
    根据 y_order 生成切片信息:
    {
        "eye_gaze": slice(0, 2),
        "pupil": slice(2, 3),
        ...
    }
    """
    dims = _resolve_label_dims(y_order, label_dims=label_dims)

    out = {}
    start = 0
    for name in y_order:
        d = dims[name]
        out[name] = slice(start, start + d)
        start += d
    return out, dims


# =========================================================
# 2. 从 batch["y"] 中按 y_order 拼接成 tensor
# =========================================================
def _extract_y_tensor_from_batch(y, y_order, device="cpu", label_dims=None):
    """
    支持:
    1) y 本身就是 tensor -> 直接 to(device)
    2) y 是 dict -> 按 y_order 顺序拼接
    """
    if torch.is_tensor(y):
        return y.to(device)

    if not isinstance(y, dict):
        raise TypeError("batch['y'] must be a tensor or dict.")

    dims = _resolve_label_dims(y_order, label_dims=label_dims)
    pieces = []

    for name in y_order:
        if name not in y:
            raise KeyError(f"'{name}' not found in batch['y'].")

        v = y[name]
        if not torch.is_tensor(v):
            raise TypeError(f"batch['y']['{name}'] must be a tensor.")

        v = v.to(device)

        # 如果是 [B]，补成 [B, 1]
        if v.ndim == 1:
            v = v.unsqueeze(-1)

        expected_dim = dims[name]
        if v.shape[-1] != expected_dim:
            raise ValueError(
                f"batch['y']['{name}'] last dim = {v.shape[-1]}, "
                f"expected {expected_dim}"
            )

        pieces.append(v)

    return torch.cat(pieces, dim=-1)


# =========================================================
# 3. 预测
# =========================================================
def get_predictions(model, loader, y_order, device="cpu", label_dims=None):
    model.eval()

    y_true_all = []
    y_pred_all = []
    sub_all = []
    index_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            y = batch["y"]

            # move x to device
            if isinstance(x, dict):
                x = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in x.items()
                }
            else:
                x = x.to(device)

            # flexible y
            y_tensor = _extract_y_tensor_from_batch(
                y,
                y_order=y_order,
                device=device,
                label_dims=label_dims,
            )

            model_out = model(x)
            if isinstance(model_out, tuple):
                _, y_hat = model_out
            else:
                y_hat = model_out

            y_true_all.append(y_tensor.detach().cpu().numpy())
            y_pred_all.append(y_hat.detach().cpu().numpy())

            if "sub" in batch:
                sub_all.extend(batch["sub"])

            if "index" in batch:
                idx = batch["index"]
                if torch.is_tensor(idx):
                    index_all.extend(idx.detach().cpu().numpy().tolist())
                else:
                    index_all.extend(list(idx))

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "sub": sub_all,
        "index": index_all,
    }


# =========================================================
# 4. 按 y_order 拆输出
# =========================================================
def split_outputs_by_y_order(y, y_order, label_dims=None):
    slices, dims = _build_slices_from_y_order(y_order, label_dims=label_dims)

    out = {}
    for name in y_order:
        s = slices[name]
        arr = y[:, s]
        if dims[name] == 1:
            arr = arr[:, 0]
        out[name] = arr
    return out


# =========================================================
# 5. 画图：保持原来的风格，但变成通用版
# =========================================================
def _plot_1d_series_and_scatter(y_true, y_pred, name, max_plot_points=500):
    n = len(y_true)
    m = min(max_plot_points, n)
    idx = np.arange(m)

    # time series
    plt.figure(figsize=(8, 4))
    plt.plot(idx, y_true[:m], label=f"true {name}", alpha=0.8)
    plt.plot(idx, y_pred[:m], label=f"pred {name}", alpha=0.8)
    plt.title(f"{name}: true vs pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.3)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.title(f"{name}: pred vs true")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.grid(True)
    plt.show()


def _plot_2d_series_and_scatter(y_true, y_pred, name, max_plot_points=500):
    n_dim = y_true.shape[1]

    for d in range(n_dim):
        n = len(y_true)
        m = min(max_plot_points, n)
        idx = np.arange(m)

        # time series
        plt.figure(figsize=(8, 4))
        plt.plot(idx, y_true[:m, d], label=f"true {name}_{d}", alpha=0.8)
        plt.plot(idx, y_pred[:m, d], label=f"pred {name}_{d}", alpha=0.8)
        plt.title(f"{name}_{d}: true vs pred")
        plt.legend()
        plt.grid(True)
        plt.show()

        # scatter
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true[:, d], y_pred[:, d], alpha=0.3)
        mn = min(np.min(y_true[:, d]), np.min(y_pred[:, d]))
        mx = max(np.max(y_true[:, d]), np.max(y_pred[:, d]))
        plt.plot([mn, mx], [mn, mx], "--")
        plt.title(f"{name}_{d}: pred vs true")
        plt.xlabel("true")
        plt.ylabel("pred")
        plt.grid(True)
        plt.show()


# =========================================================
# 6. 主评估函数
# =========================================================
def evaluate_model(
    model,
    test_loader,
    y_order,
    device="cpu",
    max_plot_points=500,
    label_dims=None,
):
    """
    例子：
    y_order = ["emo_neutral", "jacky_ratio", "front_face_ratio"]
    或
    y_order = ["eye_gaze", "pupil"]
    """

    out = get_predictions(
        model,
        test_loader,
        y_order=y_order,
        device=device,
        label_dims=label_dims,
    )

    y_true = out["y_true"]
    y_pred = out["y_pred"]

    y_true_dict = split_outputs_by_y_order(y_true, y_order, label_dims=label_dims)
    y_pred_dict = split_outputs_by_y_order(y_pred, y_order, label_dims=label_dims)
    dims = _resolve_label_dims(y_order, label_dims=label_dims)

    metrics = {}

    print("=== Test metrics ===")
    for name in y_order:
        yt = y_true_dict[name]
        yp = y_pred_dict[name]
        d = dims[name]

        if d == 1:
            metrics[name] = _metrics_1d(yt, yp)

            print(f"\n{name}")
            for k, v in metrics[name].items():
                if np.isnan(v):
                    print(f"  {k}: nan")
                else:
                    print(f"  {k}: {v:.6f}")

        else:
            metrics[name] = {}
            for i in range(d):
                metrics[name][f"{name}_{i}"] = _metrics_1d(yt[:, i], yp[:, i])

            metrics[name][f"{name}_all"] = {
                "mae": float(np.mean(np.abs(yt - yp))),
                "mse": float(np.mean((yt - yp) ** 2)),
                "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
            }

            print(f"\n{name}")
            for sub_name, vals in metrics[name].items():
                print(f"  {sub_name}")
                for k, v in vals.items():
                    if isinstance(v, float) and np.isnan(v):
                        print(f"    {k}: nan")
                    else:
                        print(f"    {k}: {v:.6f}")

    # plot
    for name in y_order:
        yt = y_true_dict[name]
        yp = y_pred_dict[name]
        d = dims[name]

        if d == 1:
            _plot_1d_series_and_scatter(
                yt, yp, name=name, max_plot_points=max_plot_points
            )
        else:
            _plot_2d_series_and_scatter(
                yt, yp, name=name, max_plot_points=max_plot_points
            )

    return metrics, out