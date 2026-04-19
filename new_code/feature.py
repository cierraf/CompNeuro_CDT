import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

import copy
from sklearn.decomposition import PCA

def make_lagged_matrix(X, lags):
    """
    X: [T, D]
    lags: list/array of ints
    return: [T, D * len(lags)]
    """
    X = np.asarray(X, dtype=float)
    T, D = X.shape
    mats = []

    for lag in lags:
        Xs = np.zeros((T, D), dtype=float)

        if lag < 0:
            Xs[:lag, :] = X[-lag:, :]
        elif lag > 0:
            Xs[lag:, :] = X[:-lag, :]
        else:
            Xs[:, :] = X

        mats.append(Xs)

    return np.concatenate(mats, axis=1)


# =========================
# 1. 一些基础工具函数
# =========================

def _as_2d(x):
    """
    Convert input to 2D array:
    - [T] -> [T, 1]
    - [T, D] -> [T, D]
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _safe_mean_std(x, axis=0, eps=1e-8):
    mu = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd


def _find_baseline_mask(t, baseline_start=-10.0, baseline_end=0.0):
    t = np.asarray(t, dtype=float)
    return (t >= baseline_start) & (t < baseline_end)


def _find_window_mask(t, start, end, right_inclusive=False):
    t = np.asarray(t, dtype=float)
    if right_inclusive:
        return (t >= start) & (t <= end)
    return (t >= start) & (t < end)


def _normalize_by_baseline(
    X: np.ndarray,
    baseline_X: np.ndarray,
    mode: Optional[str] = "zscore"
):
    """
    X: [T, D]
    baseline_X: [Tb, D]
    mode:
        - None / 'none': no normalization
        - 'zscore': (X - mu) / sd
        - 'center': X - mu
        - 'ratio': X / mu
        - 'percent_change': (X - mu) / |mu|
    """
    if mode is None or mode == "none":
        return X

    X = np.asarray(X, dtype=float)
    baseline_X = np.asarray(baseline_X, dtype=float)

    mu, sd = _safe_mean_std(baseline_X, axis=0)

    if mode == "zscore":
        return (X - mu) / sd
    elif mode == "center":
        return X - mu
    elif mode == "ratio":
        denom = np.where(np.abs(mu) < 1e-8, 1.0, mu)
        return X / denom
    elif mode == "percent_change":
        denom = np.where(np.abs(mu) < 1e-8, 1.0, np.abs(mu))
        return (X - mu) / denom
    else:
        raise ValueError(f"Unknown baseline normalization mode: {mode}")
    
def _normalize_array(X, method="zscore", eps=1e-8):
    """
    X: [T, D]
    return: [T, D]
    """
    if method == "zscore":
        mean = np.nanmean(X, axis=0, keepdims=True)
        std = np.nanstd(X, axis=0, keepdims=True)

        std = np.where(std < eps, 1.0, std)

        return (X - mean) / std

    elif method == "max":
        max_abs = np.nanmax(np.abs(X), axis=0, keepdims=True)
        max_abs = np.where(max_abs < eps, 1.0, max_abs)

        return X / max_abs

    elif method == "min_max":
        xmin = np.nanmin(X, axis=0, keepdims=True)
        xmax = np.nanmax(X, axis=0, keepdims=True)

        scale = xmax - xmin
        scale = np.where(scale < eps, 1.0, scale)

        return (X - xmin) / scale

    else:
        raise ValueError(f"Unknown norm_method: {method}")
    
def normalize_subject_modalities(
    sub_dict: Dict[str, Any],
    modality_keys: Optional[List[str]] = None,
    norm_method: str = "zscore",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对单个被试的多个模态做 normalization（非 baseline）。

    Parameters
    ----------
    sub_dict : dict
        单个 subject 的数据

    modality_keys : list[str] or None
        要处理的模态；None → 自动检测所有 ndarray-like

    norm_method : str
        "zscore" | "max" | "min_max"

    Returns
    -------
    new_sub : dict
        归一化后的新 dict（不改原始）
    """
    new_sub = dict(sub_dict)

    if modality_keys is None:
        modality_keys = [
            k for k, v in sub_dict.items()
            if isinstance(v, (list, np.ndarray))
        ]

    for key in modality_keys:
        if key not in sub_dict or sub_dict[key] is None:
            continue

        try:
            X_raw = sub_dict[key]
            X = _as_2d(X_raw)

            if X.size == 0 or np.isnan(X).all():
                if verbose:
                    print(f"[WARN] {key}: empty or all NaN → skip")
                continue

            Xn = _normalize_array(X, method=norm_method)

            if np.asarray(X_raw).ndim == 1:
                Xn = Xn[:, 0]

            new_sub[key] = Xn

        except Exception as e:
            if verbose:
                print(f"[WARN] {key}: normalization failed → keep original ({e})")
            new_sub[key] = sub_dict[key]

    return new_sub


def _infer_emotion_cols(df_face_note: pd.DataFrame) -> List[str]:
    emo_cols = [c for c in df_face_note.columns if c.startswith("emo_")]
    if len(emo_cols) == 0:
        raise ValueError("No emotion columns found. Expected columns like 'emo_neutral', 'emo_happy', ...")
    return emo_cols


def _check_required_face_cols(df_face_note: pd.DataFrame):
    required = ["movie_time"]
    for c in required:
        if c not in df_face_note.columns:
            raise ValueError(f"df_face_note missing required column: {c}")


# =========================
# 2. face label -> soft label window
# =========================

def build_face_soft_labels_for_windows(
    df_face_note: pd.DataFrame,
    window_starts: np.ndarray,
    window_size_sec: float,
    use_only_valid_face_frames: bool = True,
    valid_face_mode: str = "front_or_face",
    include_aux_labels: bool = True,
) -> pd.DataFrame:
    """
    根据窗口起点生成 soft labels。

    Parameters
    ----------
    df_face_note : DataFrame
        必须至少包含:
        - movie_time
        - emo_* columns
        可选:
        - n_faces
        - any_front_face
        - Jacky_present

    window_starts : array-like
        每个窗口的起点（秒）

    window_size_sec : float
        窗口长度（秒）

    use_only_valid_face_frames : bool
        是否只在“有效脸帧”上统计 emotion 占比

    valid_face_mode : str
        如何定义有效脸帧:
        - 'front_or_face': (n_faces > 0) OR (any_front_face == 1)
        - 'front_and_face': (n_faces > 0) AND (any_front_face == 1)
        - 'front_only': any_front_face == 1
        - 'face_only': n_faces > 0
        - 'all': 所有帧都算

    include_aux_labels : bool
        是否额外输出一些窗口统计:
        - valid_face_ratio
        - front_face_ratio
        - jacky_ratio
        - mean_n_faces
    """
    _check_required_face_cols(df_face_note)
    emo_cols = _infer_emotion_cols(df_face_note)

    df = df_face_note.copy().sort_values("movie_time").reset_index(drop=True)
    t_face = df["movie_time"].to_numpy(dtype=float)

    has_n_faces = "n_faces" in df.columns
    has_front = "any_front_face" in df.columns
    has_jacky = "Jacky_present" in df.columns

    if has_n_faces:
        n_faces = df["n_faces"].to_numpy(dtype=float)
    else:
        n_faces = np.zeros(len(df), dtype=float)

    if has_front:
        any_front = df["any_front_face"].astype(int).to_numpy()
    else:
        any_front = np.zeros(len(df), dtype=int)

    if has_jacky:
        jacky = df["Jacky_present"].astype(int).to_numpy()
    else:
        jacky = np.zeros(len(df), dtype=int)

    emo_mat = df[emo_cols].astype(float).to_numpy()  # [Nf, K]

    # 定义有效脸帧
    if valid_face_mode == "front_or_face":
        valid_face = ((n_faces > 0) | (any_front == 1))
    elif valid_face_mode == "front_and_face":
        valid_face = ((n_faces > 0) & (any_front == 1))
    elif valid_face_mode == "front_only":
        valid_face = (any_front == 1)
    elif valid_face_mode == "face_only":
        valid_face = (n_faces > 0)
    elif valid_face_mode == "all":
        valid_face = np.ones(len(df), dtype=bool)
    else:
        raise ValueError(f"Unknown valid_face_mode: {valid_face_mode}")

    rows = []

    for ws in window_starts:
        we = ws + window_size_sec
        m = _find_window_mask(t_face, ws, we, right_inclusive=False)

        row = {
            "window_start": ws,
            "window_end": we,
            "n_face_frames_total": int(m.sum()),
        }

        if m.sum() == 0:
            # 没覆盖到任何 face annotation frame
            for c in emo_cols:
                row[c] = np.nan
            row["p_non_neutral"] = np.nan
            row["hard_label"] = None

            if include_aux_labels:
                row["valid_face_ratio"] = np.nan
                row["front_face_ratio"] = np.nan
                row["jacky_ratio"] = np.nan
                row["mean_n_faces"] = np.nan

            rows.append(row)
            continue

        valid_m = m & valid_face if use_only_valid_face_frames else m
        n_valid = int(valid_m.sum())

        if n_valid == 0:
            # 窗口里有 frame，但没有有效脸
            for c in emo_cols:
                row[c] = 0.0
            row["p_non_neutral"] = 0.0
            row["hard_label"] = "no_valid_face"

            if include_aux_labels:
                row["valid_face_ratio"] = 0.0
                row["front_face_ratio"] = float(any_front[m].mean()) if m.sum() > 0 else np.nan
                row["jacky_ratio"] = float(jacky[m].mean()) if m.sum() > 0 else np.nan
                row["mean_n_faces"] = float(n_faces[m].mean()) if m.sum() > 0 else np.nan

            rows.append(row)
            continue

        emo_prob = emo_mat[valid_m].mean(axis=0)

        for i, c in enumerate(emo_cols):
            row[c] = float(emo_prob[i])

        if "emo_neutral" in emo_cols:
            row["p_non_neutral"] = float(1.0 - row["emo_neutral"])
        else:
            row["p_non_neutral"] = np.nan

        row["hard_label"] = emo_cols[int(np.nanargmax(emo_prob))]

        if include_aux_labels:
            row["valid_face_ratio"] = float(valid_m.sum() / m.sum()) if m.sum() > 0 else np.nan
            row["front_face_ratio"] = float(any_front[m].mean()) if m.sum() > 0 else np.nan
            row["jacky_ratio"] = float(jacky[m].mean()) if m.sum() > 0 else np.nan
            row["mean_n_faces"] = float(n_faces[m].mean()) if m.sum() > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# =========================
# 3. 对单个模态做 baseline normalization
# =========================

def baseline_normalize_subject_modalities(
    sub_dict: Dict[str, Any],
    modality_keys: Optional[List[str]] = None,
    baseline_time_key: str = "time_grid",
    baseline_start: float = -10.0,
    baseline_end: float = 0.0,
    mode: Optional[str] = "zscore",
) -> Dict[str, Any]:
    """
    对单个被试的各模态按 baseline 做归一化。
    返回一个新的 sub_dict，不改原始数据。
    """
    new_sub = dict(sub_dict)

    if modality_keys is None:
        modality_keys = ["lfp_macro", "lfp_micro", "lfp_bandpower"]

    if baseline_time_key not in sub_dict:
        raise ValueError(f"sub_dict missing baseline_time_key: {baseline_time_key}")

    t = np.asarray(sub_dict[baseline_time_key], dtype=float)
    baseline_mask = _find_baseline_mask(t, baseline_start=baseline_start, baseline_end=baseline_end)

    if baseline_mask.sum() == 0:
        raise ValueError(
            f"No baseline samples found in [{baseline_start}, {baseline_end}) "
            f"using time key '{baseline_time_key}'."
        )

    for key in modality_keys:
        if key not in sub_dict or sub_dict[key] is None:
            continue

        X = _as_2d(sub_dict[key]).astype(float)
        if len(X) != len(t):
            print(f"Warning: modality '{key}' length {len(X)} does not match time length {len(t)}. Skipping baseline normalization for this modality.")
            continue

        Xb = X[baseline_mask]
        Xn = _normalize_by_baseline(X, Xb, mode=mode)
        print(f"normalized modality '{key}' with baseline mode '{mode}' using {baseline_mask.sum()} baseline samples.")

        if np.asarray(sub_dict[key]).ndim == 1:
            Xn = Xn[:, 0]

        new_sub[key] = Xn

    return new_sub


# =========================
# 4. 从单个被试生成窗口 neural sequence
# =========================

def extract_windowed_neural_sequences_for_subject(
    sub_dict: Dict[str, Any],
    window_size_sec: float = 0.5,
    stride_sec: float = 0.1,
    start_time: Optional[float] = 0.0,
    end_time: Optional[float] = None,
    remove_baseline_windows: bool = True,
    modality_keys: Optional[List[str]] = None,
    require_all_modalities_nonempty: bool = False,
) -> Dict[str, Any]:
    """
    对单个被试，从 time_grid 上切窗口，直接截取每个窗口的原始序列，不做 summary。

    Returns
    -------
    {
        "window_df": DataFrame,
        "sequences": {
            modality_key: List[np.ndarray],   # 每个元素 shape [Tw, D]
        }
    }
    """
    if modality_keys is None:
        modality_keys = ["firing_rate", "lfp_macro", "lfp_micro", "lfp_bandpower", "eye_gaze", "pupil"]

    t_grid = np.asarray(sub_dict["time_grid"], dtype=float)

    if end_time is None:
        end_time = float(np.nanmax(t_grid))

    if remove_baseline_windows and start_time is not None:
        start_time = max(start_time, 0.0)

    if start_time is None:
        start_time = float(np.nanmin(t_grid))

    # 窗口起点
    # 确保窗口完整落在 [start_time, end_time]
    last_start = end_time - window_size_sec
    if last_start < start_time:
        raise ValueError("No valid windows: end_time - window_size_sec < start_time")

    window_starts = np.arange(start_time, last_start + 1e-12, stride_sec, dtype=float)

    sequences = {k: [] for k in modality_keys}
    rows = []

    for ws in window_starts:
        we = ws + window_size_sec
        m = _find_window_mask(t_grid, ws, we, right_inclusive=False)

        row = {
            "window_start": float(ws),
            "window_end": float(we),
            "n_timepoints": int(m.sum()),
        }

        keep = True

        for key in modality_keys:
            if key not in sub_dict or sub_dict[key] is None:
                seq = None
            else:
                X = _as_2d(sub_dict[key])
                if len(X) != len(t_grid):
                    seq = None
                else:
                    seq = X[m].copy()

            sequences[key].append(seq)

            if require_all_modalities_nonempty:
                if seq is None or len(seq) == 0:
                    keep = False

        row["valid_window"] = bool(keep and m.sum() > 0)
        rows.append(row)

    window_df = pd.DataFrame(rows)

    return {
        "window_df": window_df,
        "sequences": sequences,
    }


# =========================
# 5. 单个被试：整合 soft label + neural sequence
# =========================

def build_subject_window_dataset(
    sub_id: Any,
    sub_dict: Dict[str, Any],
    df_face_note: pd.DataFrame,
    window_size_sec: float = 0.5,
    stride_sec: float = 0.1,
    start_time: Optional[float] = 0.0,
    end_time: Optional[float] = None,
    remove_baseline_windows: bool = True,
    do_baseline_norm: bool = True,
    norm_modality_dict: Optional[Dict[str, bool]] = None,
    baseline_mode: Optional[str] = "zscore",
    baseline_start: float = -10.0,
    baseline_end: float = 0.0,
    modality_keys: Optional[List[str]] = None,
    use_only_valid_face_frames: bool = True,
    valid_face_mode: str = "front_or_face",
    require_all_modalities_nonempty: bool = False,
) -> Dict[str, Any]:
    """
    单个被试完整处理：
    1) baseline normalization
    2) 切 neural window sequence
    3) 生 face soft label
    4) merge 成一个窗口级 dataset
    """
    if modality_keys is None:
        modality_keys = ["firing_rate", "lfp_macro", "lfp_micro", "lfp_bandpower", "eye_gaze", "pupil"]

    baseline_modality_keys = ["lfp_macro", "lfp_micro"]

    # 1) baseline normalize
    sub_proc = sub_dict
    if do_baseline_norm:
        sub_proc = baseline_normalize_subject_modalities(
            sub_proc,
            modality_keys=baseline_modality_keys,
            baseline_time_key="time_grid",
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            mode=baseline_mode,
        )

    for key, [do_norm, norm_method] in (norm_modality_dict or {}).items():
        if do_norm:
            sub_proc = normalize_subject_modalities(
                sub_proc,
                modality_keys=[key],
                norm_method=norm_method,
            )

    # 2) neural sequences
    neural_pack = extract_windowed_neural_sequences_for_subject(
        sub_proc,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
        start_time=start_time,
        end_time=end_time,
        remove_baseline_windows=remove_baseline_windows,
        modality_keys=modality_keys,
        require_all_modalities_nonempty=require_all_modalities_nonempty,
    )

    window_df = neural_pack["window_df"].copy()
    window_starts = window_df["window_start"].to_numpy(dtype=float)

    # 3) face soft labels
    label_df = build_face_soft_labels_for_windows(
        df_face_note=df_face_note,
        window_starts=window_starts,
        window_size_sec=window_size_sec,
        use_only_valid_face_frames=use_only_valid_face_frames,
        valid_face_mode=valid_face_mode,
        include_aux_labels=True,
    )

    # 4) merge
    merged_df = window_df.merge(
        label_df,
        on=["window_start", "window_end"],
        how="left",
        validate="one_to_one",
    )
    merged_df.insert(0, "sub", sub_id)

    return {
        "window_df": merged_df,
        "sequences": neural_pack["sequences"],
        "config": {
            "window_size_sec": window_size_sec,
            "stride_sec": stride_sec,
            "start_time": start_time,
            "end_time": end_time,
            "remove_baseline_windows": remove_baseline_windows,
            "do_baseline_norm": do_baseline_norm,
            "baseline_mode": baseline_mode,
            "baseline_start": baseline_start,
            "baseline_end": baseline_end,
            "modality_keys": modality_keys,
            "use_only_valid_face_frames": use_only_valid_face_frames,
            "valid_face_mode": valid_face_mode,
            "require_all_modalities_nonempty": require_all_modalities_nonempty,
        }
    }


# =========================
# 6. 对所有被试 out[sub] 批量处理
# =========================

def build_all_subject_window_datasets(
    out: Dict[Any, Dict[str, Any]],
    df_face_note: pd.DataFrame,
    window_size_sec: float = 0.5,
    stride_sec: float = 0.1,
    start_time: Optional[float] = 0.0,
    end_time: Optional[float] = None,
    remove_baseline_windows: bool = True,
    do_baseline_norm: bool = True,
    norm_modality_dict: Optional[Dict[str, bool]] = None,
    baseline_mode: Optional[str] = "zscore",
    baseline_start: float = -10.0,
    baseline_end: float = 0.0,
    modality_keys: Optional[List[str]] = None,
    use_only_valid_face_frames: bool = True,
    valid_face_mode: str = "front_or_face",
    require_all_modalities_nonempty: bool = False,
    verbose: bool = True,
) -> Dict[Any, Dict[str, Any]]:
    """
    对 out 里的所有 sub 做处理。
    返回:
        all_window_data[sub] = {
            "window_df": ...,
            "sequences": ...,
            "config": ...
        }
    """
    all_window_data = {}

    for sub_id, sub_dict in out.items():
        try:
            if verbose:
                print(f"[build] subject {sub_id}")

            res = build_subject_window_dataset(
                sub_id=sub_id,
                sub_dict=sub_dict,
                df_face_note=df_face_note,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                start_time=start_time,
                end_time=end_time,
                remove_baseline_windows=remove_baseline_windows,
                do_baseline_norm=do_baseline_norm,
                norm_modality_dict=norm_modality_dict,
                baseline_mode=baseline_mode,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                modality_keys=modality_keys,
                use_only_valid_face_frames=use_only_valid_face_frames,
                valid_face_mode=valid_face_mode,
                require_all_modalities_nonempty=require_all_modalities_nonempty,
            )
            all_window_data[sub_id] = res

            if verbose:
                print(
                    f"  -> n_windows = {len(res['window_df'])}, "
                    f"valid_windows = {int(res['window_df']['valid_window'].sum())}"
                )

        except Exception as e:
            print(f"[failed] subject {sub_id}: {repr(e)}")

    return all_window_data


# =========================
# 7. 一个小工具：把某个模态的窗口序列pad成3D
# =========================

def pad_window_sequences(
    seq_list: List[Optional[np.ndarray]],
    pad_value: float = np.nan,
) -> (np.ndarray, np.ndarray):
    """
    把 list of [Ti, D] pad 成 [N, Tmax, D]
    返回:
        X_pad: [N, Tmax, D]
        mask : [N, Tmax]   (True=有效)
    """
    valid_seqs = [s for s in seq_list if s is not None and len(s) > 0]
    if len(valid_seqs) == 0:
        raise ValueError("No valid sequences to pad.")

    D = valid_seqs[0].shape[1]
    Tmax = max(s.shape[0] for s in valid_seqs)
    N = len(seq_list)

    X_pad = np.full((N, Tmax, D), pad_value, dtype=float)
    mask = np.zeros((N, Tmax), dtype=bool)

    for i, s in enumerate(seq_list):
        if s is None or len(s) == 0:
            continue
        Ti = s.shape[0]
        X_pad[i, :Ti, :] = s
        mask[i, :Ti] = True

    return X_pad, mask

# =========================
# filter and concat
# =========================


def filter_window_dataset(
    subject_window_data: Dict[str, Any],
    valid_window_only: bool = True,
    min_valid_face_ratio: Optional[float] = None,
    min_face_frames: Optional[int] = None,
    min_non_neutral: Optional[float] = None,
    allowed_hard_labels: Optional[List[str]] = None,
    require_modalities_nonempty: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    对单个被试的 window dataset 做筛选。

    Parameters
    ----------
    subject_window_data : dict
        build_subject_window_dataset(...) 的输出，即:
        {
            "window_df": ...,
            "sequences": {...},
            "config": ...
        }

    valid_window_only : bool
        是否只保留 window_df["valid_window"] == True 的窗口

    min_valid_face_ratio : float or None
        只保留 valid_face_ratio >= 该阈值 的窗口

    min_face_frames : int or None
        只保留 n_face_frames_total >= 该阈值 的窗口

    min_non_neutral : float or None
        只保留 p_non_neutral >= 该阈值 的窗口

    allowed_hard_labels : list[str] or None
        只保留 hard_label 在这个集合中的窗口

    require_modalities_nonempty : list[str] or None
        要求这些模态对应的 sequence 不能是 None 且长度 > 0

    verbose : bool
        是否打印筛选前后数量

    Returns
    -------
    filtered_data : dict
        结构与输入一致，但已经过滤过
    """
    df = subject_window_data["window_df"].copy().reset_index(drop=True)
    seqs = subject_window_data["sequences"]
    config = subject_window_data.get("config", {})

    n0 = len(df)
    keep = np.ones(n0, dtype=bool)

    # 1) valid_window
    if valid_window_only and "valid_window" in df.columns:
        keep &= df["valid_window"].fillna(False).to_numpy(dtype=bool)

    # 2) valid_face_ratio
    if min_valid_face_ratio is not None:
        if "valid_face_ratio" not in df.columns:
            raise ValueError("window_df missing column 'valid_face_ratio'")
        keep &= (df["valid_face_ratio"].fillna(-np.inf).to_numpy(dtype=float) >= min_valid_face_ratio)

    # 3) n_face_frames_total
    if min_face_frames is not None:
        if "n_face_frames_total" not in df.columns:
            raise ValueError("window_df missing column 'n_face_frames_total'")
        keep &= (df["n_face_frames_total"].fillna(-1).to_numpy(dtype=float) >= min_face_frames)

    # 4) p_non_neutral
    if min_non_neutral is not None:
        if "p_non_neutral" not in df.columns:
            raise ValueError("window_df missing column 'p_non_neutral'")
        keep &= (df["p_non_neutral"].fillna(-np.inf).to_numpy(dtype=float) >= min_non_neutral)

    # 5) hard_label
    if allowed_hard_labels is not None:
        if "hard_label" not in df.columns:
            raise ValueError("window_df missing column 'hard_label'")
        keep &= df["hard_label"].isin(allowed_hard_labels).to_numpy(dtype=bool)

    # 6) 模态非空要求
    if require_modalities_nonempty is not None:
        for mod in require_modalities_nonempty:
            if mod not in seqs:
                raise ValueError(f"sequences missing modality: {mod}")

            mod_keep = []
            for s in seqs[mod]:
                ok = (s is not None) and (len(s) > 0)
                mod_keep.append(ok)
            keep &= np.asarray(mod_keep, dtype=bool)

    idx = np.where(keep)[0]

    filtered_df = df.iloc[idx].reset_index(drop=True)
    filtered_seqs = {}

    for mod, seq_list in seqs.items():
        filtered_seqs[mod] = [seq_list[i] for i in idx]

    if verbose:
        print(f"[filter_window_dataset] kept {len(idx)} / {n0} windows")

    return {
        "window_df": filtered_df,
        "sequences": filtered_seqs,
        "config": config,
        "kept_original_indices": idx,
    }


def concat_subject_window_data(
    all_window_data: Dict[Any, Dict[str, Any]],
    subjects: Optional[List[Any]] = None,
    modality_keys: Optional[List[str]] = None,
    add_subject_column: bool = True,
    reset_index: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    把多个被试的 window dataset 拼起来。

    Parameters
    ----------
    all_window_data : dict
        build_all_subject_window_datasets(...) 或过滤后的结果:
        {
            sub1: {"window_df": ..., "sequences": ..., "config": ...},
            sub2: {"window_df": ..., "sequences": ..., "config": ...},
            ...
        }

    subjects : list or None
        指定要拼接的被试；None 表示全部

    modality_keys : list[str] or None
        指定要拼接哪些模态；None 表示取所有被试共有/出现过的模态并集

    add_subject_column : bool
        若 window_df 中没有 "sub" 列，则自动加上

    reset_index : bool
        是否重置拼接后 df 的索引

    verbose : bool
        是否打印拼接信息

    Returns
    -------
    concat_data : dict
        {
            "window_df": DataFrame,
            "sequences": {
                mod1: list[np.ndarray],
                mod2: list[np.ndarray],
                ...
            },
            "subject_slices": {
                sub1: slice(start, end),
                sub2: slice(start, end),
                ...
            },
            "subjects": [...],
        }
    """
    if subjects is None:
        subjects = list(all_window_data.keys())

    # 过滤只保留存在的 subjects
    subjects = [s for s in subjects if s in all_window_data]
    if len(subjects) == 0:
        raise ValueError("No valid subjects found in all_window_data.")

    # 推断模态
    if modality_keys is None:
        modality_set = set()
        for sub in subjects:
            modality_set.update(all_window_data[sub]["sequences"].keys())
        modality_keys = sorted(modality_set)

    df_list = []
    concat_seqs = {mod: [] for mod in modality_keys}
    subject_slices = {}

    cursor = 0

    for sub in subjects:
        sub_data = all_window_data[sub]
        df = sub_data["window_df"].copy()
        seqs = sub_data["sequences"]

        if add_subject_column and "sub" not in df.columns:
            df.insert(0, "sub", sub)

        n_sub = len(df)

        # 检查长度一致
        for mod in modality_keys:
            if mod not in seqs:
                raise ValueError(f"Subject {sub} missing modality '{mod}' in sequences")
            if len(seqs[mod]) != n_sub:
                raise ValueError(
                    f"Length mismatch in subject {sub}, modality {mod}: "
                    f"len(seqs)={len(seqs[mod])}, len(window_df)={n_sub}"
                )

        df_list.append(df)

        for mod in modality_keys:
            concat_seqs[mod].extend(seqs[mod])

        subject_slices[sub] = slice(cursor, cursor + n_sub)
        cursor += n_sub

        if verbose:
            print(f"[concat] sub={sub}, n_windows={n_sub}")

    concat_df = pd.concat(df_list, axis=0)
    if reset_index:
        concat_df = concat_df.reset_index(drop=True)

    if verbose:
        print(f"[concat_subject_window_data] total windows = {len(concat_df)}")

    return {
        "window_df": concat_df,
        "sequences": concat_seqs,
        "subject_slices": subject_slices,
        "subjects": subjects,
        "modality_keys": modality_keys,
    }


def apply_subjectwise_pca_to_modalities(
    out,
    modality_pca_dict,
    copy_data=True,
    fillna_value=0.0,
    store_pca_model=False,
    verbose=True,
):
    """
    对 out 中每个 subject 的指定 modality 做 subject-wise PCA，
    返回与 out 相同的数据结构。

    Parameters
    ----------
    out : dict
        结构类似:
        {
            sub: {
                "lfp_macro": np.ndarray[T, D],
                "lfp_micro": np.ndarray[T, D],
                "eye_gaze": np.ndarray[T, D],
                ...
                "meta": {...}
            },
            ...
        }

    modality_pca_dict : dict
        类似:
        {
            "lfp_micro": 10,
            "lfp_macro": 10,
            "eye_gaze": 5,
            "pupil": 0,
        }

        规则:
        - 如果 modality 不在 dict 里: 不做 PCA，原样返回
        - 如果值是 0: 不做 PCA，原样返回
        - 如果值 > 0: 做 PCA 到 min(目标维数, 原始特征维数)

    copy_data : bool, default=True
        True: 深拷贝后返回，不修改原 out
        False: 原地修改并返回

    fillna_value : float, default=0.0
        PCA 前若数据中有 NaN，用该值填充

    store_pca_model : bool, default=False
        是否把每个 sub / modality 的 PCA model 存下来。
        注意这会让返回对象更大。

    verbose : bool, default=True
        是否打印处理信息

    Returns
    -------
    out_pca : dict
        与 out 相同结构，但指定 modality 被 PCA 后替换。
    """

    if not isinstance(out, dict):
        raise TypeError("`out` must be a dict like {sub: data_dict}")

    if modality_pca_dict is None:
        modality_pca_dict = {}

    out_pca = copy.deepcopy(out) if copy_data else out

    for sub, sub_dict in out_pca.items():
        if not isinstance(sub_dict, dict):
            if verbose:
                print(f"[skip] sub={sub}: value is not a dict")
            continue

        # 确保 meta 存在
        if "meta" not in sub_dict or sub_dict["meta"] is None:
            sub_dict["meta"] = {}

        if "modality_pca" not in sub_dict["meta"] or sub_dict["meta"]["modality_pca"] is None:
            sub_dict["meta"]["modality_pca"] = {}

        if store_pca_model and "modality_pca_models" not in sub_dict["meta"]:
            sub_dict["meta"]["modality_pca_models"] = {}

        for modality, target_dim in modality_pca_dict.items():
            # modality 不存在：跳过
            if modality not in sub_dict:
                if verbose:
                    print(f"[skip] sub={sub}, modality={modality}: not found")
                continue

            x = sub_dict[modality]

            # None: 跳过
            if x is None:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": "input is None",
                }
                if verbose:
                    print(f"[skip] sub={sub}, modality={modality}: input is None")
                continue

            x = np.asarray(x)

            # target_dim == 0: 不做 PCA
            if target_dim == 0:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": "target_dim == 0",
                    "orig_shape": tuple(x.shape),
                    "new_shape": tuple(x.shape),
                }
                if verbose:
                    print(f"[keep] sub={sub}, modality={modality}: target_dim=0, keep original")
                continue

            # 只处理二维 [T, D]
            if x.ndim != 2:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": f"expected 2D [T, D], got ndim={x.ndim}",
                    "orig_shape": tuple(x.shape),
                    "new_shape": tuple(x.shape),
                }
                if verbose:
                    print(f"[skip] sub={sub}, modality={modality}: expected 2D, got shape={x.shape}")
                continue

            T, D = x.shape

            # 空数组 / 特征维异常
            if T == 0 or D == 0:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": "empty input",
                    "orig_shape": tuple(x.shape),
                    "new_shape": tuple(x.shape),
                }
                if verbose:
                    print(f"[skip] sub={sub}, modality={modality}: empty shape={x.shape}")
                continue

            n_comp = min(int(target_dim), D, T)

            # 有效维度太小，不值得 PCA
            if n_comp <= 0:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": f"invalid n_components={n_comp}",
                    "orig_shape": tuple(x.shape),
                    "new_shape": tuple(x.shape),
                }
                if verbose:
                    print(f"[skip] sub={sub}, modality={modality}: invalid n_comp={n_comp}")
                continue

            # 如果目标维度 >= 原维度，直接保留原值
            if n_comp >= D:
                sub_dict["meta"]["modality_pca"][modality] = {
                    "applied": False,
                    "reason": "target_dim >= original feature dim",
                    "orig_shape": tuple(x.shape),
                    "new_shape": tuple(x.shape),
                    "orig_dim": int(D),
                    "target_dim": int(target_dim),
                    "used_dim": int(n_comp),
                }
                if verbose:
                    print(f"[keep] sub={sub}, modality={modality}: target_dim >= D ({target_dim} >= {D})")
                continue

            # 处理 NaN
            x_in = x.astype(np.float64, copy=True)
            if np.isnan(x_in).any():
                x_in = np.nan_to_num(x_in, nan=fillna_value)

            # PCA
            pca = PCA(n_components=n_comp)
            x_pca = pca.fit_transform(x_in)   # [T, n_comp]

            sub_dict[modality] = x_pca

            sub_dict["meta"]["modality_pca"][modality] = {
                "applied": True,
                "orig_shape": tuple(x.shape),
                "new_shape": tuple(x_pca.shape),
                "orig_dim": int(D),
                "target_dim": int(target_dim),
                "used_dim": int(n_comp),
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }

            if store_pca_model:
                sub_dict["meta"]["modality_pca_models"][modality] = pca

            if verbose:
                evr_sum = float(np.sum(pca.explained_variance_ratio_))
                print(
                    f"[pca] sub={sub}, modality={modality}: "
                    f"{x.shape} -> {x_pca.shape}, explained_var_sum={evr_sum:.4f}"
                )

    return out_pca