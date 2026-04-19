import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Any, Optional, List, Tuple


# =========================
# basic utils
# =========================
def _ensure_2d_array(x):
    """
    x: [T] or [T, D]
    return: [T, D]
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape={x.shape}")
    return x


def _reduce_window_sequence(
    x,
    mode="mean",   # "mean" | "last" | "flatten" | "time_mean" | "none"
):
    """
    Reduce one window sequence into feature vector / tensor.

    Parameters
    ----------
    x : array-like
        [T] or [T, D]
    mode : str
        - "mean" / "time_mean": mean over time -> [D]
        - "last": last time step -> [D]
        - "flatten": flatten -> [T*D]
        - "none": keep [T, D]

    Returns
    -------
    np.ndarray
    """
    x = _ensure_2d_array(x).astype(float)

    if mode in ["mean", "time_mean"]:
        return np.nanmean(x, axis=0)

    elif mode == "last":
        return x[-1]

    elif mode == "flatten":
        return x.reshape(-1)

    elif mode == "none":
        return x

    else:
        raise ValueError(f"Unknown reduce mode: {mode}")


def _to_tensor_safe(x, dtype=torch.float32):
    x = np.asarray(x)
    return torch.as_tensor(x, dtype=dtype)


# =========================
# label builders
# =========================
def get_label_dict_from_concat_data(
    concat_data,
    y_col,
    threshold=0.1,
    sub_col="sub_id",
    mode="binary",      # "binary" | "continuous" | "soft" | "category"
    temperature=0.05,
):
    """
    Return {sub: y_sub} from concat_data["window_df"].

    For numeric labels:
        - binary:   y > threshold -> {0,1}
        - soft:     sigmoid((y-threshold)/temperature)
        - continuous: raw y
    For categorical labels:
        - category: factorize within global dataframe
    """
    df = concat_data["window_df"].copy().reset_index(drop=True)

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found")
    if y_col not in df.columns:
        raise ValueError(f"`{y_col}` not found")

    out = {}
    sub_ids = df[sub_col].unique()

    if mode == "category":
        # global consistent coding
        codes, uniques = pd.factorize(df[y_col], sort=True)
        df[f"__{y_col}_coded__"] = codes

        for sub in sub_ids:
            idx = (df[sub_col] == sub).values
            y = df.loc[idx, f"__{y_col}_coded__"].to_numpy().astype(int)
            out[sub] = y

        return out, list(uniques)

    else:
        for sub in sub_ids:
            idx = (df[sub_col] == sub).values
            y = df.loc[idx, y_col].to_numpy()

            if mode == "binary":
                y = (y > threshold).astype(int)

            elif mode == "soft":
                y = 1 / (1 + np.exp(-(y - threshold) / temperature))

            elif mode == "continuous":
                y = y.astype(float)

            else:
                raise ValueError("mode must be binary / soft / continuous / category")

            out[sub] = y

        return out, None


def get_sequence_label_dict_from_concat_data(
    concat_data,
    seq_key,
    reduce_mode="mean",   # "mean" | "last" | "flatten" | "none"
    sub_col="sub_id",
):
    """
    Build label dict from sequences, e.g. gaze / pupil.

    Returns
    -------
    out : dict
        {sub: np.ndarray}, where each value is:
            [N, D] if reduced to vector
            [N, T, D] if reduce_mode="none"
    """
    df = concat_data["window_df"].copy().reset_index(drop=True)
    seqs = concat_data["sequences"]

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")
    if seq_key not in seqs:
        raise ValueError(f"`{seq_key}` not found in concat_data['sequences']")

    out = {}
    sub_ids = df[sub_col].unique()

    for sub in sub_ids:
        idx = np.where((df[sub_col] == sub).values)[0]
        feats = []
        for i in idx:
            x = seqs[seq_key][i]
            x_red = _reduce_window_sequence(x, mode=reduce_mode)
            feats.append(x_red)

        feats = np.asarray(feats)
        out[sub] = feats

    return out


# =========================
# label spec inference
# =========================
def build_y_dicts_from_concat_data(
    concat_data,
    y_labels: List[str],
    sub_col="sub_id",
    category_label_names=(),
    seq_label_names=("gaze", "pupil"),
    numeric_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    category_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sequence_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Build multiple label dicts:
        y_dicts[label_name][sub] = y_sub

    Supports:
    - category labels from df columns: emo_neutral, jacky_ratio, front_face_ratio
    - sequence labels from sequences: gaze, pupil
    - numeric labels from df columns via numeric_label_configs

    Returns
    -------
    y_dicts : dict
    label_meta : dict
    """
    if numeric_label_configs is None:
        numeric_label_configs = {}

    if category_label_configs is None:
        category_label_configs = {}

    if sequence_label_configs is None:
        sequence_label_configs = {}

    y_dicts = {}
    label_meta = {}

    for y_name in y_labels:
        # category labels from df
        if y_name in category_label_names:
            cfg = {
                "y_col": y_name,
                "mode": "category",
            }
            cfg.update(category_label_configs.get(y_name, {}))

            y_dict, classes = get_label_dict_from_concat_data(
                concat_data=concat_data,
                y_col=cfg["y_col"],
                sub_col=sub_col,
                mode=cfg.get("mode", "category"),
            )
            y_dicts[y_name] = y_dict
            label_meta[y_name] = {
                "type": "category",
                "classes": classes,
                "n_classes": None if classes is None else len(classes),
            }

        # sequence labels from sequences
        elif y_name in seq_label_names:
            cfg = {
                "seq_key": y_name,
                "reduce_mode": "mean",  # default
            }
            cfg.update(sequence_label_configs.get(y_name, {}))

            y_dict = get_sequence_label_dict_from_concat_data(
                concat_data=concat_data,
                seq_key=cfg["seq_key"],
                reduce_mode=cfg.get("reduce_mode", "mean"),
                sub_col=sub_col,
            )
            y_dicts[y_name] = y_dict

            sample_sub = next(iter(y_dict.keys()))
            sample_shape = np.asarray(y_dict[sample_sub]).shape[1:] if len(y_dict[sample_sub]) > 0 else None

            label_meta[y_name] = {
                "type": "sequence_regression",
                "reduce_mode": cfg.get("reduce_mode", "mean"),
                "shape_per_sample": sample_shape,
            }

        # numeric labels from df columns
        else:
            if y_name not in numeric_label_configs:
                raise ValueError(
                    f"y_label='{y_name}' not recognized. "
                    f"Provide it in numeric_label_configs / category_label_configs / sequence_label_configs."
                )

            cfg = {
                "y_col": y_name,
                "threshold": 0.1,
                "mode": "binary",
                "temperature": 0.05,
            }
            cfg.update(numeric_label_configs[y_name])

            y_dict, _ = get_label_dict_from_concat_data(
                concat_data=concat_data,
                y_col=cfg["y_col"],
                threshold=cfg.get("threshold", 0.1),
                sub_col=sub_col,
                mode=cfg.get("mode", "binary"),
                temperature=cfg.get("temperature", 0.05),
            )
            y_dicts[y_name] = y_dict
            label_meta[y_name] = {
                "type": cfg.get("mode", "binary"),
            }

    return y_dicts, label_meta


# =========================
# X builder
# =========================
def build_x_dict_from_concat_data(
    concat_data,
    modality_keys: List[str],
    modality_reduce_modes: Optional[Dict[str, str]] = None,
    sub_col="sub_id",
):
    """
    Build:
        x_dict[mod][sub] = np.ndarray of shape [N, ...]
    """
    df = concat_data["window_df"].copy().reset_index(drop=True)
    seqs = concat_data["sequences"]

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")

    if modality_reduce_modes is None:
        modality_reduce_modes = {}

    x_dict = {}
    sub_ids = df[sub_col].unique()

    for mod in modality_keys:
        if mod not in seqs:
            raise ValueError(f"Modality '{mod}' not found in concat_data['sequences']")

        red_mode = modality_reduce_modes.get(mod, "none")
        x_dict[mod] = {}

        for sub in sub_ids:
            idx = np.where((df[sub_col] == sub).values)[0]
            feats = []

            for i in idx:
                x = seqs[mod][i]
                x_red = _reduce_window_sequence(x, mode=red_mode)
                feats.append(x_red)

            feats = np.asarray(feats)
            x_dict[mod][sub] = feats

    return x_dict


# =========================
# flatten subject dict -> sample list
# =========================
def merge_subjectwise_x_y_to_samples(
    x_dict: Dict[str, Dict[Any, np.ndarray]],
    y_dicts: Dict[str, Dict[Any, np.ndarray]],
):
    """
    Merge subject-wise dicts into sample-wise lists.

    Returns
    -------
    samples : list[dict]
    """
    modality_keys = list(x_dict.keys())
    label_keys = list(y_dicts.keys())

    subjects = list(x_dict[modality_keys[0]].keys())
    samples = []

    for sub in subjects:
        n = len(x_dict[modality_keys[0]][sub])

        # check x lengths
        for mod in modality_keys:
            if len(x_dict[mod][sub]) != n:
                raise ValueError(f"Length mismatch in X for sub={sub}, mod={mod}")

        # check y lengths
        for y_name in label_keys:
            if len(y_dicts[y_name][sub]) != n:
                raise ValueError(f"Length mismatch in Y for sub={sub}, label={y_name}")

        for i in range(n):
            sample = {
                "x": {mod: x_dict[mod][sub][i] for mod in modality_keys},
                "y": {y_name: y_dicts[y_name][sub][i] for y_name in label_keys},
                "sub": sub,
                "index_within_sub": i,
            }
            samples.append(sample)

    return samples


# =========================
# dataset
# =========================
class MultiModalWindowDataset(Dataset):
    def __init__(self, samples, label_meta=None):
        self.samples = samples
        self.label_meta = label_meta if label_meta is not None else {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x_out = {}
        for mod, x in s["x"].items():
            x_out[mod] = _to_tensor_safe(x, dtype=torch.float32)

        y_out = {}
        for y_name, y in s["y"].items():
            meta_type = self.label_meta.get(y_name, {}).get("type", None)

            if meta_type == "category":
                y_out[y_name] = torch.as_tensor(y, dtype=torch.long)
            elif meta_type == "binary":
                y_out[y_name] = torch.as_tensor(y, dtype=torch.float32)
            else:
                # continuous / soft / sequence regression
                y_arr = np.asarray(y)
                if y_arr.ndim == 0:
                    y_out[y_name] = torch.as_tensor(y, dtype=torch.float32)
                else:
                    y_out[y_name] = torch.as_tensor(y_arr, dtype=torch.float32)

        return {
            "x": x_out,
            "y": y_out,
            "sub": s["sub"],
            "index": s["index_within_sub"],
        }


# =========================
# collate
# =========================
def multimodal_collate_fn(batch):
    """
    batch -> dict of stacked tensors
    """
    modality_keys = list(batch[0]["x"].keys())
    label_keys = list(batch[0]["y"].keys())

    x_batch = {}
    for mod in modality_keys:
        x_batch[mod] = torch.stack([b["x"][mod] for b in batch], dim=0)

    y_batch = {}
    for y_name in label_keys:
        vals = [b["y"][y_name] for b in batch]
        try:
            y_batch[y_name] = torch.stack(vals, dim=0)
        except Exception:
            # fallback if shape is not stackable
            y_batch[y_name] = vals

    sub_batch = [b["sub"] for b in batch]
    idx_batch = torch.as_tensor([b["index"] for b in batch], dtype=torch.long)

    return {
        "x": x_batch,
        "y": y_batch,
        "sub": sub_batch,
        "index": idx_batch,
    }


# =========================
# main interface
# =========================
def build_train_test_datasets_from_concat_data(
    concat_data,
    modality_keys: List[str],
    y_labels: List[str],
    sub_col="sub_id",
    modality_reduce_modes: Optional[Dict[str, str]] = None,
    numeric_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    category_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sequence_label_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    test_ratio=0.2,
    batch_size=128,
    shuffle_train=True,
    random_seed=42,
):
    """
    Main interface.

    Parameters
    ----------
    modality_keys : list[str]
        Input modalities used as X
    y_labels : list[str]
        Target names

    Special handling
    ----------------
    category labels:
        emo / jacky / front_face
    sequence labels:
        gaze / pupil

    modality_reduce_modes : dict
        e.g.
        {
            "lfp_macro": "none",
            "eye_gaze": "none",
            "pupil": "none",
            "firing_rate": "none",
        }

    sequence_label_configs : dict
        e.g.
        {
            "gaze": {"seq_key": "eye_gaze", "reduce_mode": "mean"},
            "pupil": {"seq_key": "pupil", "reduce_mode": "last"},
        }

    Returns
    -------
    out : dict
    """
    if modality_reduce_modes is None:
        modality_reduce_modes = {}

    x_dict = build_x_dict_from_concat_data(
        concat_data=concat_data,
        modality_keys=modality_keys,
        modality_reduce_modes=modality_reduce_modes,
        sub_col=sub_col,
    )

    y_dicts, label_meta = build_y_dicts_from_concat_data(
        concat_data=concat_data,
        y_labels=y_labels,
        sub_col=sub_col,
        numeric_label_configs=numeric_label_configs,
        category_label_configs=category_label_configs,
        sequence_label_configs=sequence_label_configs,
    )

    samples = merge_subjectwise_x_y_to_samples(
        x_dict=x_dict,
        y_dicts=y_dicts,
    )

    dataset = MultiModalWindowDataset(samples, label_meta=label_meta)

    n_total = len(dataset)
    n_test = int(round(n_total * test_ratio))
    n_test = max(1, n_test)
    n_train = n_total - n_test

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_test],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=multimodal_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate_fn,
    )

    return {
        "dataset": dataset,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "samples": samples,
        "label_meta": label_meta,
        "modality_keys": modality_keys,
        "y_labels": y_labels,
    }