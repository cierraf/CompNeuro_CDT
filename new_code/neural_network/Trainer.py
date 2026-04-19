# -----------------------------
# End-to-end training runner
# Adapted for batch = {
#   "x": {modality_name: tensor},
#   "y": {label_name: tensor} or tensor,
#   "sub": ...,
#   "index": ...
# }
# -----------------------------
from typing import List, Dict, Sequence, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from new_code.neural_network.evaluation import metrics
from new_code.neural_network import LossFunction


def _move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    elif torch.is_tensor(obj):
        return obj.to(device)
    else:
        return obj


def _pack_y_dict(y_dict: dict, y_order: Optional[List[str]] = None):
    """
    Convert label dict to one tensor [B, D].

    Example:
        {"gaze": [B,2], "pupil": [B,1]} -> [B,3]
    """
    if y_order is None:
        # default: use insertion order
        y_order = list(y_dict.keys())

    y_list = []
    for k in y_order:
        if k not in y_dict:
            raise KeyError(f"y key '{k}' not found. Available keys: {list(y_dict.keys())}")
        yk = y_dict[k]
        if not torch.is_tensor(yk):
            raise TypeError(f"y['{k}'] must be a tensor, got {type(yk)}")

        if yk.ndim == 1:
            yk = yk.unsqueeze(-1)

        y_list.append(yk)

    return torch.cat(y_list, dim=-1)


def _parse_batch(batch, y_order: Optional[List[str]] = None):
    """
    Support:
    1) batch is dict with keys x/y/...
    2) batch is tuple/list like (x, y) or (x, y, meta)
    """
    if isinstance(batch, dict):
        if "x" not in batch or "y" not in batch:
            raise KeyError(f"Batch dict must contain 'x' and 'y'. Got keys: {list(batch.keys())}")

        x = batch["x"]
        y = batch["y"]
        meta = {k: v for k, v in batch.items() if k not in ["x", "y"]}
    elif isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
            meta = {}
        elif len(batch) >= 3:
            x, y, meta = batch[:3]
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    # y may be dict -> pack to tensor
    if isinstance(y, dict):
        y = _pack_y_dict(y, y_order=y_order)

    if not torch.is_tensor(y):
        raise TypeError(f"Parsed y must be a tensor, got {type(y)}")

    return x, y, meta


def _get_batch_size(x, y):
    if torch.is_tensor(y):
        return y.size(0)

    if isinstance(x, dict):
        first_key = next(iter(x.keys()))
        return x[first_key].size(0)

    if torch.is_tensor(x):
        return x.size(0)

    raise TypeError("Cannot infer batch size.")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    loss='mse',
    grad_clip: float = 1.0,
    y_order: Optional[List[str]] = None,
    task_type: str = "regression",   # "regression" or "binary"
):
    model.train()

    total_loss = 0.0
    total_mae = 0.0
    total_acc = 0.0
    n = 0

    loss_fn = LossFunction._resolve_loss(loss)

    for batch in loader:
        x, y, meta = _parse_batch(batch, y_order=y_order)

        x = _move_to_device(x, device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        _, y_hat = model(x)
        loss_val = loss_fn(y_hat, y)

        loss_val.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = _get_batch_size(x, y)
        total_loss += loss_val.item() * bs

        # ============================
        # metric (simple & correct)
        # ============================
        with torch.no_grad():
            if task_type == "regression":
                mae = torch.mean(torch.abs(y_hat - y))
                total_mae += mae.item() * bs

            elif task_type == "binary":
                probs = torch.sigmoid(y_hat)
                pred = (probs > 0.5).float()
                acc = (pred == y).float().mean()
                total_acc += acc.item() * bs

        n += bs

    out = {
        "loss": total_loss / max(n, 1),
    }

    if task_type == "regression":
        out["mae"] = total_mae / max(n, 1)

    if task_type == "binary":
        out["acc"] = total_acc / max(n, 1)

    return out


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    device,
    loss='mse',
    y_order: Optional[List[str]] = None,
):
    model.eval()
    total_loss, total_acc, total_joint, n = 0.0, 0.0, 0.0, 0

    loss_fn = LossFunction._resolve_loss(loss)

    for batch in loader:
        x, y, meta = _parse_batch(batch, y_order=y_order)

        x = _move_to_device(x, device)
        y = y.to(device)

        z_seq, y_hat = model(x)
        loss_val = loss_fn(y_hat, y)

        bs = _get_batch_size(x, y)
        total_loss += loss_val.item() * bs

        m = metrics.metrics_r_cos_sin(y_hat, y, loss_fn=loss_fn)
        total_acc += m["angle_acc"] * bs
        total_joint += m["joint_acc"] * bs
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "acc": total_acc / max(n, 1),
        "joint_acc": total_joint / max(n, 1),
    }


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 42
    save_path: str = "best_model.pt"
    early_stop_patience: int = 10
    min_delta: float = 1e-4

    # IMPORTANT:
    # specify label order for dict-y packing
    # Example: ["gaze", "pupil"] -> [B,2] + [B,1] = [B,3]
    y_order: Optional[List[str]] = None


def run_training_with_loaders(model: nn.Module, train_loader, val_loader, cfg, loss="mse", device=None):
    torch.manual_seed(cfg.seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_joint_acc": [],
        "val_joint_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss=loss,
            grad_clip=cfg.grad_clip,
            y_order=cfg.y_order,
        )

        va = eval_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            loss=loss,
            y_order=cfg.y_order,
        )

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr.get("acc", None))
        history["val_acc"].append(va.get("acc", None))
        history["train_joint_acc"].append(tr.get("joint_acc", None))
        history["val_joint_acc"].append(va.get("joint_acc", None))

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        if va["loss"] < best_val_loss - cfg.min_delta:
            best_val_loss = va["loss"]
            best_epoch = epoch
            early_stop_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                cfg.save_path,
            )
            improved = True
        else:
            early_stop_counter += 1
            improved = False

        scheduler.step(va["loss"])
        new_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"lr {current_lr:.2e} | "
            f"train loss {tr['loss']:.6f} | "
            f"val loss {va['loss']:.6f} | "
            f"best {best_val_loss:.6f} (ep {best_epoch:03d}) | "
            f"es {early_stop_counter}/{cfg.early_stop_patience}"
            + (" | improved" if improved else "")
            + (f" | lr-> {new_lr:.2e}" if new_lr != current_lr else "")
        )

        if early_stop_counter >= cfg.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    ckpt = torch.load(cfg.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded best model from epoch {ckpt.get('best_epoch', 'N/A')} "
        f"with val loss {ckpt.get('best_val_loss', 'N/A')}"
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, history["lr"], label="learning_rate")
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.title("LR curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return history, cfg.save_path