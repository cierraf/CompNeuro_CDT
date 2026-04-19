import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.signal import welch

def _extract_stats_features(X):
    """
    X: [N, T, K]
    return: [N, K * n_stats]
    """

    N, T, K = X.shape
    feats = []

    # ===== basic stats =====
    feats.append(np.nanmean(X, axis=1))   # [N, K]
    feats.append(np.nanstd(X, axis=1))
    feats.append(np.nanmin(X, axis=1))
    feats.append(np.nanmax(X, axis=1))

    # ===== slope (per channel) =====
    if T > 1:
        t = np.arange(T, dtype=float)
        t = (t - t.mean()) / (t.std() + 1e-8)

        slopes = np.zeros((N, K))

        for i in range(N):
            for d in range(K):
                xd = X[i, :, d]
                mask = np.isfinite(xd)

                if mask.sum() < 2:
                    slopes[i, d] = np.nan
                else:
                    slopes[i, d] = np.polyfit(t[mask], xd[mask], 1)[0]

        feats.append(slopes)  # [N, K]

    else:
        feats.append(np.full((N, K), np.nan))

    # ===== concat =====
    return np.concatenate(feats, axis=1)  # [N, K * 5]

def _safe_trapezoid(y, x):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def _compute_bandpower_1d(
    sig,
    fs=1000,
    bands=None,
    relative=False,
    welch_nperseg=None,
):
    """
    sig: [T]
    return: [n_bands]
    """
    sig = np.asarray(sig, dtype=float)

    if sig.ndim != 1:
        raise ValueError("`sig` must be 1D")

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 80),
        }

    if sig.size == 0 or np.isnan(sig).all():
        return np.full(len(bands), np.nan, dtype=float)

    if np.isnan(sig).any():
        m = np.nanmean(sig)
        sig = np.where(np.isnan(sig), m, sig)

    if welch_nperseg is None:
        nperseg = min(256, len(sig))
    else:
        nperseg = min(welch_nperseg, len(sig))

    if nperseg < 2:
        return np.full(len(bands), np.nan, dtype=float)

    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)

    if len(freqs) == 0 or len(psd) == 0:
        return np.full(len(bands), np.nan, dtype=float)

    total_power = _safe_trapezoid(psd, freqs)
    feats = []

    for _, (f_low, f_high) in bands.items():
        idx = (freqs >= f_low) & (freqs <= f_high)

        if not np.any(idx):
            bp = np.nan
        else:
            bp = _safe_trapezoid(psd[idx], freqs[idx])

        if relative:
            if np.isnan(total_power) or total_power <= 0:
                bp = np.nan
            else:
                bp = bp / total_power

        feats.append(bp)

    return np.asarray(feats, dtype=float)


def _extract_bandpower_features(
    Z,
    fs=1000,
    bands=None,
    relative=False,
    welch_nperseg=None,
):
    """
    Z: [N, T, K]
    return: [N, K * n_bands]
    """
    Z = np.asarray(Z, dtype=float)

    if Z.ndim != 3:
        raise ValueError("`Z` must be 3D [N, T, K]")

    N, T, K = Z.shape

    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 80),
        }

    X = np.full((N, K * len(bands)), np.nan, dtype=float)

    for i in range(N):
        feat_i = []
        for k in range(K):
            sig = Z[i, :, k]  # [T]
            bp = _compute_bandpower_1d(
                sig,
                fs=fs,
                bands=bands,
                relative=relative,
                welch_nperseg=welch_nperseg,
            )
            feat_i.append(bp)

        X[i] = np.concatenate(feat_i, axis=0)

    return X

# ====================================
# label spliter for time wind
# ====================================
def split_neutral_vs_nonneutral_windows(
    df_window,
    neutral_col="emo_neutral",
    nonneutral_cols=None,
    eps=1e-8
):
    if nonneutral_cols is None:
        nonneutral_cols = ["emo_afraid", "emo_angry", "emo_happy", "emo_surprised"]

    nonneutral_sum = df_window[nonneutral_cols].sum(axis=1)

    neutral_mask = (df_window[neutral_col] > eps) & (nonneutral_sum <= eps)
    nonneutral_mask = (nonneutral_sum > eps)

    return neutral_mask.to_numpy(), nonneutral_mask.to_numpy()

def split_jacky_present_windows(
    df_window,
    jacky_present_col = "jacky_ratio",
    eps:float = 0.5,
):
    assert jacky_present_col in df_window.columns
    print(df_window[jacky_present_col].min(), df_window[jacky_present_col].max())
    jacky_present_mask = df_window[jacky_present_col] >= eps
    jacky_nonpresent_mask = df_window[jacky_present_col] < eps

    return jacky_nonpresent_mask.to_numpy(), jacky_present_mask.to_numpy()

def split_front_face_present_windows(
    df_window,
    front_face_present_col = "front_face_ratio",
    eps:float = 0.5,
):
    assert front_face_present_col in df_window.columns
    front_face_present_mask = df_window[front_face_present_col] >= eps
    front_face_nonpresent_mask = df_window[front_face_present_col] < eps

    return front_face_nonpresent_mask.to_numpy(), front_face_present_mask.to_numpy()

# ====================================
# modality stacker
# ====================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _stack_equal_length_sequences(seq_list):
    keep_idx_local = []
    X_list = []

    ref_shape = None
    for i, x in enumerate(seq_list):
        if x is None:
            continue

        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x[:, None]

        if x.size == 0 or np.isnan(x).all():
            continue

        if ref_shape is None:
            ref_shape = x.shape

        if x.shape != ref_shape:
            raise ValueError(
                f"Found unequal window shape: first shape={ref_shape}, current shape={x.shape}. "
                "If windows are not equal length, use padding/truncation first."
            )

        X_list.append(x)
        keep_idx_local.append(i)

    if len(X_list) == 0:
        raise ValueError("No valid sequences found.")

    X = np.stack(X_list, axis=0)   # [N, T, D]
    keep_idx_local = np.asarray(keep_idx_local, dtype=int)
    return X, keep_idx_local


def _run_grouped_trajectory_pca_on_one_df_seq(
    df,
    seq_list,
    modality="eye_gaze",
    spliter="emo",
    n_components=3,
    eps=1e-8,
    verbose=True,
    sub_id=None,
):
    """
    对单个 sub 的 df + seq_list 跑 grouped trajectory PCA。
    返回结构尽量保持和你原来一致，只额外加 sub_id。
    """
    valid_spliters = {"emo", "jacky", "front_face"}
    if spliter not in valid_spliters:
        raise ValueError(f"`spliter` must be one of {valid_spliters}, got {spliter}")

    # ----------------------------
    # 1) define split rule
    # ----------------------------
    if spliter == "emo":
        group1_mask, group2_mask = split_neutral_vs_nonneutral_windows(
            df,
            neutral_col="emo_neutral",
            nonneutral_cols=["emo_afraid", "emo_angry", "emo_happy", "emo_surprised"],
            eps=eps,
        )
        group1_name = "neutral"
        group2_name = "nonneutral"

    elif spliter == "jacky":
        group1_mask, group2_mask = split_jacky_present_windows(
            df,
            eps=eps,
        )
        group1_name = "no_jacky"
        group2_name = "jacky_present"

    elif spliter == "front_face":
        group1_mask, group2_mask = split_front_face_present_windows(
            df,
            eps=eps,
        )
        group1_name = "no_front_face"
        group2_name = "front_face_present"

    # ----------------------------
    # 2) select windows
    # ----------------------------
    group1_idx = np.where(group1_mask)[0]
    group2_idx = np.where(group2_mask)[0]

    group1_seq = [seq_list[i] for i in group1_idx]
    group2_seq = [seq_list[i] for i in group2_idx]

    X_group1, keep_group1_local = _stack_equal_length_sequences(group1_seq)
    X_group2, keep_group2_local = _stack_equal_length_sequences(group2_seq)

    keep_group1_idx = group1_idx[keep_group1_local]
    keep_group2_idx = group2_idx[keep_group2_local]

    # ----------------------------
    # 3) fit one shared PCA basis
    # ----------------------------
    X_all = np.concatenate([X_group1, X_group2], axis=0)   # [N_all, T, D]
    N_all, T, D = X_all.shape

    n_components_use = min(n_components, D, N_all * T)
    if n_components_use < 1:
        raise ValueError("No valid dimension for PCA.")

    X_all_2d = X_all.reshape(N_all * T, D)

    pca = PCA(n_components=n_components_use)
    Z_all_2d = pca.fit_transform(X_all_2d)
    Z_all = Z_all_2d.reshape(N_all, T, n_components_use)

    n_group1 = X_group1.shape[0]
    Z_group1 = Z_all[:n_group1]
    Z_group2 = Z_all[n_group1:]

    df_g1 = df.iloc[keep_group1_idx].reset_index(drop=True).copy()
    df_g2 = df.iloc[keep_group2_idx].reset_index(drop=True).copy()

    # 显式加上 sub_id 列，方便后面 concat / 分组
    if sub_id is not None:
        df_g1["sub_id"] = sub_id
        df_g2["sub_id"] = sub_id

    if verbose:
        prefix = f"[{modality}]"
        if sub_id is not None:
            prefix += f" [sub={sub_id}]"
        print(f"{prefix} split by: {spliter}")
        print(f"{prefix} {group1_name} windows: {len(keep_group1_idx)}")
        print(f"{prefix} {group2_name} windows: {len(keep_group2_idx)}")
        print(f"{prefix} shared explained variance ratio: {pca.explained_variance_ratio_}")

    return {
        "modality": modality,
        "spliter": spliter,
        "group1_name": group1_name,
        "group2_name": group2_name,

        "sub_id": sub_id,

        "pca": pca,
        "explained_variance_ratio": pca.explained_variance_ratio_,

        group1_name: {
            "window_idx": keep_group1_idx,
            "X": X_group1,   # [N, T, D]
            "Z": Z_group1,   # [N, T, K]
            "df": df_g1,
            "sub_id": sub_id,
        },

        group2_name: {
            "window_idx": keep_group2_idx,
            "X": X_group2,   # [N, T, D]
            "Z": Z_group2,   # [N, T, K]
            "df": df_g2,
            "sub_id": sub_id,
        },
    }


def run_grouped_trajectory_pca_on_concat_data(
    concat_data,
    modality="eye_gaze",
    spliter="emo",
    n_components=3,
    eps=1e-8,
    verbose=True,
    split_by_sub=False,
    sub_col="sub",
    global_key="all",
):
    """
    统一返回结构：
        {
            sub_id_or_global_key: result_dict
        }

    其中每个 result_dict 的结构保持为：
        {
            "modality": ...,
            "spliter": ...,
            "group1_name": ...,
            "group2_name": ...,
            "sub_id": ...,
            "pca": ...,
            "explained_variance_ratio": ...,
            group1_name: {...},
            group2_name: {...},
        }

    参数
    ----
    split_by_sub : bool
        True  -> 每个被试单独做 PCA，返回 {sub_id: result}
        False -> 整体一起做 PCA，但仍返回 {global_key: result}
    global_key : str
        当 split_by_sub=False 时，外层 dict 的 key
    """
    df = concat_data["window_df"].copy().reset_index(drop=True)
    seq_list = concat_data["sequences"][modality]

    if len(df) != len(seq_list):
        raise ValueError(
            f"len(window_df)={len(df)} does not match "
            f"len(sequences[{modality}])={len(seq_list)}"
        )

    results_by_sub = {}

    # ============================
    # 不按 sub 分：仍然返回 dict
    # ============================
    if not split_by_sub:
        res = _run_grouped_trajectory_pca_on_one_df_seq(
            df=df,
            seq_list=seq_list,
            modality=modality,
            spliter=spliter,
            n_components=n_components,
            eps=eps,
            verbose=verbose,
            sub_id=global_key,
        )
        results_by_sub[global_key] = res
        return results_by_sub

    # ============================
    # 按 sub 分：返回 {sub_id: result}
    # ============================
    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")

    sub_ids = sorted(df[sub_col].dropna().unique())

    for sub_id in sub_ids:
        idx = np.where(df[sub_col].values == sub_id)[0]
        if len(idx) == 0:
            continue

        df_sub = df.iloc[idx].reset_index(drop=True).copy()
        seq_sub = [seq_list[i] for i in idx]

        try:
            res_sub = _run_grouped_trajectory_pca_on_one_df_seq(
                df=df_sub,
                seq_list=seq_sub,
                modality=modality,
                spliter=spliter,
                n_components=n_components,
                eps=eps,
                verbose=verbose,
                sub_id=sub_id,
            )
            results_by_sub[sub_id] = res_sub

        except ValueError as e:
            if verbose:
                print(f"[{modality}] skip sub={sub_id}: {e}")

    return results_by_sub


# ====================================
# plotter
# ====================================
def plot_pc_time_series(
    result,
    pcs=(1, 2, 3),
    with_sem=True,
    figsize=(8, 4),
):
    g1 = result["group1_name"]
    g2 = result["group2_name"]

    Z1 = result[g1]["Z"]   # [N, T, K]
    Z2 = result[g2]["Z"]

    T = Z1.shape[1]
    t = np.arange(T)

    fig, axes = plt.subplots(
        len(pcs), 1,
        figsize=(figsize[0], figsize[1] * len(pcs)),
        squeeze=False
    )

    for row, pc in enumerate(pcs):
        ax = axes[row, 0]
        k = pc - 1

        if k >= Z1.shape[2] or k >= Z2.shape[2]:
            ax.set_visible(False)
            continue

        m_z1 = Z1[:, :, k].mean(axis=0)
        m_z2 = Z2[:, :, k].mean(axis=0)

        ax.plot(t, m_z1, label=g1)
        ax.plot(t, m_z2, label=g2)

        if with_sem:
            se_1 = Z1[:, :, k].std(axis=0) / np.sqrt(max(Z1.shape[0], 1))
            se_2 = Z2[:, :, k].std(axis=0) / np.sqrt(max(Z2.shape[0], 1))

            ax.fill_between(t, m_z1 - se_1, m_z1 + se_1, alpha=0.2)
            ax.fill_between(t, m_z2 - se_2, m_z2 + se_2, alpha=0.2)

        title = f"{result['modality']} | PC{pc}"
        if result.get("sub_id") is not None:
            title += f" | sub={result['sub_id']}"

        ax.axhline(0, linestyle="--", alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("Time within window")
        ax.set_ylabel("Score")
        ax.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")

    plt.tight_layout()
    plt.show()


def plot_grouped_pca(result, pcx=1, pcy=2, alpha=0.7, equal_axis=True, title_suffix=""):
    g1 = result["group1_name"]
    g2 = result["group2_name"]

    Z1 = result[g1]["Z"]   # [N, T, K]
    Z2 = result[g2]["Z"]

    ix = pcx - 1
    iy = pcy - 1

    if ix >= Z1.shape[2] or iy >= Z1.shape[2]:
        raise ValueError(f"Requested PC out of range. Available K={Z1.shape[2]}")

    Z1_mean = Z1.mean(axis=0)   # [T, K]
    Z2_mean = Z2.mean(axis=0)

    plt.figure(figsize=(5, 5))
    plt.plot(Z1_mean[:, ix], Z1_mean[:, iy], label=g1, alpha=alpha)
    plt.plot(Z2_mean[:, ix], Z2_mean[:, iy], label=g2, alpha=alpha)

    plt.xlabel(f"PC{pcx}")
    plt.ylabel(f"PC{pcy}")
    plt.legend(bbox_to_anchor=(1.05, 0.0), loc="lower left")

    title = f"{result['modality']} PCA: {g1} vs {g2}"
    if result.get("sub_id") is not None:
        title += f" | sub={result['sub_id']}"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)

    if equal_axis:
        plt.gca().set_aspect("equal", "box")

    plt.grid(alpha=0.3)
    plt.show()

def plot_grouped_pca_across_sub(
    results_by_sub,
    pcx=1,
    pcy=2,
    alpha=0.4,
    equal_axis=True,
    title_suffix="",
):
    first_res = next(iter(results_by_sub.values()))

    g1 = first_res["group1_name"]
    g2 = first_res["group2_name"]

    colors = {
        g1: "tab:blue",
        g2: "tab:orange",
    }

    ix = pcx - 1
    iy = pcy - 1

    plt.figure(figsize=(6, 6))

    Z1_all = []
    Z2_all = []

    # markers = ["o", "s", "^", "v", "D", "P", "X", "*"]

    for i, (sub_id, res) in enumerate(results_by_sub.items()):
        Z1 = res[g1]["Z"]
        Z2 = res[g2]["Z"]

        if ix >= Z1.shape[2] or iy >= Z1.shape[2]:
            continue

        Z1_mean = Z1.mean(axis=0)
        Z2_mean = Z2.mean(axis=0)

        # marker = markers[i % len(markers)]

        # per-sub trajectory
        plt.plot(
            Z1_mean[:, ix], Z1_mean[:, iy],
            color=colors[g1],
            alpha=alpha,
            # marker=marker,
            markevery=[0],
        )
        plt.plot(
            Z2_mean[:, ix], Z2_mean[:, iy],
            color=colors[g2],
            alpha=alpha,
            # marker=marker,
            markevery=[0],
        )

        Z1_all.append(Z1_mean)
        Z2_all.append(Z2_mean)

    # ===== group mean trajectory =====
    Z1_all = np.stack(Z1_all, axis=0)  # [S, T, K]
    Z2_all = np.stack(Z2_all, axis=0)

    Z1_mean = Z1_all.mean(axis=0)
    Z2_mean = Z2_all.mean(axis=0)

    plt.plot(
        Z1_mean[:, ix], Z1_mean[:, iy],
        color=colors[g1],
        linewidth=2,
        label=f"{g1} (mean)",
        alpha=0.5,
    )
    plt.plot(
        Z2_mean[:, ix], Z2_mean[:, iy],
        color=colors[g2],
        linewidth=2,
        label=f"{g2} (mean)",
        alpha=0.5,
    )

    plt.xlabel(f"PC{pcx}")
    plt.ylabel(f"PC{pcy}")

    title = f"{first_res['modality']} PCA: {g1} vs {g2}"
    if title_suffix:
        title += f" {title_suffix}"
    plt.title(title)

    if equal_axis:
        plt.gca().set_aspect("equal", "box")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_pc_time_series_across_sub(
    results_by_sub,
    pcs=(1, 2, 3),
    with_sem=True,
    figsize=(8, 4),
    alpha=0.3,
):
    sub_ids = list(results_by_sub.keys())
    first_res = next(iter(results_by_sub.values()))

    g1 = first_res["group1_name"]
    g2 = first_res["group2_name"]

    colors = {
        g1: "tab:blue",
        g2: "tab:orange",
    }

    # assume same T
    any_res = first_res
    T = any_res[g1]["Z"].shape[1]
    D = any_res[g1]["Z"].shape[2]
    t = np.arange(T)
    pcs = [pc for pc in pcs if pc - 1 < D]

    fig, axes = plt.subplots(
        len(pcs), 1,
        figsize=(figsize[0], figsize[1] * len(pcs)),
        squeeze=False
    )

    for row, pc in enumerate(pcs):
        ax = axes[row, 0]
        k = pc - 1

        Z1_all = []
        Z2_all = []

        # ===== per-sub overlay =====
        for sub_id, res in results_by_sub.items():
            Z1 = res[g1]["Z"]
            Z2 = res[g2]["Z"]

            if k >= Z1.shape[2]:
                continue

            m1 = Z1[:, :, k].mean(axis=0)
            m2 = Z2[:, :, k].mean(axis=0)

            ax.plot(t, m1, color=colors[g1], alpha=alpha)
            ax.plot(t, m2, color=colors[g2], alpha=alpha)

            Z1_all.append(Z1[:, :, k])
            Z2_all.append(Z2[:, :, k])

        # ===== group-level mean =====
        Z1_all = np.concatenate(Z1_all, axis=0)   # [N_total, T]
        Z2_all = np.concatenate(Z2_all, axis=0)

        mean1 = Z1_all.mean(axis=0)
        mean2 = Z2_all.mean(axis=0)

        ax.plot(t, mean1, color=colors[g1], linewidth=2.5, label=g1)
        ax.plot(t, mean2, color=colors[g2], linewidth=2.5, label=g2)

        if with_sem:
            se1 = Z1_all.std(axis=0) / np.sqrt(max(Z1_all.shape[0], 1))
            se2 = Z2_all.std(axis=0) / np.sqrt(max(Z2_all.shape[0], 1))

            ax.fill_between(t, mean1 - se1, mean1 + se1, color=colors[g1], alpha=0.2)
            ax.fill_between(t, mean2 - se2, mean2 + se2, color=colors[g2], alpha=0.2)

        ax.axhline(0, linestyle="--", alpha=0.5)

        ax.set_title(f"{first_res['modality']} | PC{pc}")
        ax.set_xlabel("Time within window")
        ax.set_ylabel("Score")
        ax.legend()

    plt.tight_layout()
    plt.show()

# ====================================
# helpers for per-sub plotting
# ====================================
def plot_pc_time_series_by_sub(
    results_by_sub,
    pcs=(1, 2, 3),
    with_sem=True,
    figsize=(8, 4),
):
    for sub_id, result in results_by_sub.items():
        plot_pc_time_series(
            result,
            pcs=pcs,
            with_sem=with_sem,
            figsize=figsize,
        )


def plot_grouped_pca_by_sub(
    results_by_sub,
    pcx=1,
    pcy=2,
    alpha=0.7,
    equal_axis=True,
    title_suffix="",
):
    for sub_id, result in results_by_sub.items():
        plot_grouped_pca(
            result,
            pcx=pcx,
            pcy=pcy,
            alpha=alpha,
            equal_axis=equal_axis,
            title_suffix=title_suffix,
        )

# ====================================
# decoding
# ====================================
def run_decoding_on_pca_res(
    res_dict,
    mode="time_mean",   # "time_mean" | "flatten" | "channel_mean"
    cv=5,
    fs=1000,
    bands=None,
    relative_bandpower=False,
    welch_nperseg=None,
    verbose=True,
):
    """
    对统一结构的 PCA 结果做 binary decoding。
    输入必须是：
        {
            sub_id: result_dict,
            ...
        }

    跨 sub 聚合所有 window 后做预测。
    """

    if not isinstance(res_dict, dict) or len(res_dict) == 0:
        raise ValueError("`res_dict` must be a non-empty dict like {sub_id: result_dict}")

    first_key = next(iter(res_dict.keys()))
    first_res = res_dict[first_key]

    if "group1_name" not in first_res or "group2_name" not in first_res:
        raise ValueError("Each result must contain 'group1_name' and 'group2_name'")

    g1 = first_res["group1_name"]
    g2 = first_res["group2_name"]

    X_all = []
    y_all = []
    sub_all = []

    for sub_id, res in res_dict.items():
        if g1 not in res or g2 not in res:
            raise ValueError(f"sub={sub_id} does not contain groups {g1}, {g2}")

        Z1 = res[g1]["Z"]   # [N1, T, K]
        Z2 = res[g2]["Z"]   # [N2, T, K]

        if Z1.ndim != 3 or Z2.ndim != 3:
            raise ValueError(f"sub={sub_id}: Z must be 3D [N, T, K]")

        N1 = Z1.shape[0]
        N2 = Z2.shape[0]

        if mode == "time_mean":
            X1 = Z1.mean(axis=1)         # [N, K]
            X2 = Z2.mean(axis=1)

        elif mode == "flatten":
            X1 = Z1.reshape(N1, -1)      # [N, T*K]
            X2 = Z2.reshape(N2, -1)

        elif mode == "channel_mean":
            X1 = Z1.mean(axis=-1)        # [N, T]
            X2 = Z2.mean(axis=-1)

        elif mode == "stats":
            X1 = _extract_stats_features(Z1)   # [N1, K*5]
            X2 = _extract_stats_features(Z2)

        elif mode == "bandpower":
            X1 = _extract_bandpower_features(
                Z1,
                fs=fs,
                bands=bands,
                relative=relative_bandpower,
                welch_nperseg=welch_nperseg,
            )
            X2 = _extract_bandpower_features(
                Z2,
                fs=fs,
                bands=bands,
                relative=relative_bandpower,
                welch_nperseg=welch_nperseg,
            )

        else:
            raise ValueError("mode must be one of: 'time_mean', 'flatten', 'channel_mean'")

        X_sub = np.concatenate([X1, X2], axis=0)
        y_sub = np.concatenate([
            np.zeros(N1, dtype=int),
            np.ones(N2, dtype=int),
        ])
        sub_vec = np.array([sub_id] * (N1 + N2), dtype=object)

        X_all.append(X_sub)
        y_all.append(y_sub)
        sub_all.append(sub_vec)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    sub_labels = np.concatenate(sub_all, axis=0)

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    )

    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="linear", probability=True)
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def evaluate_model(model):
        accs = []
        aucs = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_te)[:, 1]
            else:
                y_score = model.decision_function(X_te)

            auc = roc_auc_score(y_te, y_score)

            accs.append(acc)
            aucs.append(auc)

        return np.asarray(accs), np.asarray(aucs)

    logreg_acc, logreg_auc = evaluate_model(logreg)
    svm_acc, svm_auc = evaluate_model(svm)

    if verbose:
        print("=== Decoding results ===")
        print(f"Groups: {g1} vs {g2}")
        print(f"Mode: {mode}")
        print(f"Total samples: {len(y)}")
        print(f"Total subjects: {len(np.unique(sub_labels))}")

        print("\n[Logistic Regression]")
        print(f"Acc: {logreg_acc.mean():.3f} ± {logreg_acc.std():.3f}")
        print(f"AUC: {logreg_auc.mean():.3f} ± {logreg_auc.std():.3f}")

        print("\n[SVM]")
        print(f"Acc: {svm_acc.mean():.3f} ± {svm_acc.std():.3f}")
        print(f"AUC: {svm_auc.mean():.3f} ± {svm_auc.std():.3f}")

    return {
        "logreg": {
            "acc": logreg_acc,
            "auc": logreg_auc,
        },
        "svm": {
            "acc": svm_acc,
            "auc": svm_auc,
        },
        "X": X,
        "y": y,
        "sub_labels": sub_labels,
        "groups": (g1, g2),
        "n_subjects": len(np.unique(sub_labels)),
    }