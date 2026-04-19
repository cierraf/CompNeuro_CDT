import numpy as np
from itertools import combinations
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.signal import detrend

# ====================================
# window feature extraction
# ====================================
def _window_to_feature(x, mode="channel_mean"):
    """
    x: [T, D] or [T]
    return: [F]
    """
    if x is None:
        return None

    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x = x[:, None]

    if x.size == 0 or np.isnan(x).all():
        return None

    if mode == "channel_mean":
        # [T, D] -> [T]
        return np.nanmean(x, axis=-1)

    elif mode == "flatten":
        # [T, D] -> [T*D]
        return x.reshape(-1)

    elif mode == "time_mean":
        # [T, D] -> [D]
        return np.nanmean(x, axis=0)

    elif mode == "stats":
        feat_list = [
            np.nanmean(x, axis=0),
            np.nanstd(x, axis=0),
            np.nanmin(x, axis=0),
            np.nanmax(x, axis=0),
        ]

        T = x.shape[0]
        if T > 1:
            t = np.arange(T, dtype=float)
            t = (t - t.mean()) / (t.std() + 1e-8)

            slopes = []
            for d in range(x.shape[1]):
                xd = x[:, d]
                mask = np.isfinite(xd)
                if mask.sum() < 2:
                    slopes.append(np.nan)
                else:
                    beta = np.polyfit(t[mask], xd[mask], 1)[0]
                    slopes.append(beta)
            feat_list.append(np.asarray(slopes, dtype=float))
        else:
            feat_list.append(np.full(x.shape[1], np.nan))

        return np.concatenate(feat_list, axis=0)

    else:
        raise ValueError("mode must be 'channel_mean', 'flatten', 'time_mean', or 'stats'")

def build_modality_feature_dict(concat_data, modality_keys=None, mode="channel_mean", verbose=True):
    """
    Returns
    -------
    feat_dict : dict
        {
            mod1: [N, F1],
            mod2: [N, F2],
            ...
        }
    valid_mask_dict : dict
        {
            mod1: [N] bool,
            ...
        }
    """
    sequences = concat_data["sequences"]

    if modality_keys is None:
        modality_keys = list(sequences.keys())

    feat_dict = {}
    valid_mask_dict = {}

    for mod in modality_keys:
        seq_list = sequences[mod]
        feats = []
        valid = []

        for x in seq_list:
            f = _window_to_feature(x, mode=mode)
            if f is None or np.isnan(f).all():
                feats.append(None)
                valid.append(False)
            else:
                feats.append(np.asarray(f, dtype=float))
                valid.append(True)

        ref_dim = None
        for f in feats:
            if f is not None:
                ref_dim = len(f)
                break

        if ref_dim is None:
            if verbose:
                print(f"[skip] {mod}: no valid windows")
            continue

        X = []
        for f in feats:
            if f is None:
                X.append(np.full(ref_dim, np.nan))
            else:
                if len(f) != ref_dim:
                    raise ValueError(f"{mod} feature dim mismatch")
                X.append(f)

        X = np.asarray(X, dtype=float)
        feat_dict[mod] = X
        valid_mask_dict[mod] = np.asarray(valid, dtype=bool)

        if verbose:
            print(f"[{mod}] X.shape={X.shape}, valid={valid_mask_dict[mod].sum()}/{len(valid)}")

    return feat_dict, valid_mask_dict

# ====================================
# synergy calculator
# ====================================

def _cv_auc(X, y, cv=5, random_state=42):
    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    aucs = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return np.mean(aucs), np.std(aucs)

def compute_pairwise_synergy_matrix(
    concat_data,
    y,
    modality_keys=None,
    feature_mode="stats",
    cv=5,
    verbose=True,
):
    """
    y: [N] label for each window
    """
    y = np.asarray(y)
    feat_dict, valid_mask_dict = build_modality_feature_dict(
        concat_data,
        modality_keys=modality_keys,
        mode=feature_mode,
        verbose=verbose,
    )

    mods = list(feat_dict.keys())
    n_mod = len(mods)

    single_auc = {}
    single_std = {}

    # 单模态
    for mod in mods:
        X = feat_dict[mod]
        mask = valid_mask_dict[mod] & np.isfinite(y)

        X_use = X[mask]
        y_use = y[mask]

        auc_mean, auc_std = _cv_auc(X_use, y_use, cv=cv)
        single_auc[mod] = auc_mean
        single_std[mod] = auc_std

        if verbose:
            print(f"[single] {mod}: AUC={auc_mean:.4f} ± {auc_std:.4f}")

    # 双模态 synergy
    synergy_mat = np.full((n_mod, n_mod), np.nan)
    joint_auc_mat = np.full((n_mod, n_mod), np.nan)

    for i, mod_i in enumerate(mods):
        synergy_mat[i, i] = 0.0
        joint_auc_mat[i, i] = single_auc[mod_i]

    for mod_i, mod_j in combinations(mods, 2):
        i = mods.index(mod_i)
        j = mods.index(mod_j)

        Xi = feat_dict[mod_i]
        Xj = feat_dict[mod_j]
        mask = valid_mask_dict[mod_i] & valid_mask_dict[mod_j] & np.isfinite(y)

        X_joint = np.concatenate([Xi[mask], Xj[mask]], axis=1)
        y_use = y[mask]

        joint_auc, joint_std = _cv_auc(X_joint, y_use, cv=cv)

        synergy = joint_auc - max(single_auc[mod_i], single_auc[mod_j])

        synergy_mat[i, j] = synergy
        synergy_mat[j, i] = synergy
        joint_auc_mat[i, j] = joint_auc
        joint_auc_mat[j, i] = joint_auc

        if verbose:
            print(
                f"[pair] {mod_i} + {mod_j}: "
                f"joint AUC={joint_auc:.4f} ± {joint_std:.4f}, "
                f"synergy={synergy:.4f}"
            )

    return {
        "modalities": mods,
        "single_auc": single_auc,
        "single_std": single_std,
        "joint_auc_mat": joint_auc_mat,
        "synergy_mat": synergy_mat,
        "feature_mode": feature_mode,
    }

# ====================================
# plotter
# ====================================
def plot_synergy_matrix(res, title="Pairwise modality synergy"):
    mods = res["modalities"]
    M = res["synergy_mat"]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, aspect="auto")
    plt.colorbar(im, label="Synergy (joint AUC - best single AUC)")
    plt.xticks(range(len(mods)), mods, rotation=45, ha="right")
    plt.yticks(range(len(mods)), mods)
    plt.title(title)

    for i in range(len(mods)):
        for j in range(len(mods)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()

def plot_matrix(M, labels, title="", cbar_label="value", fmt=".3f", vmin=None, vmax=None):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=cbar_label)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, format(M[i, j], fmt), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def _get_pair_key(cca_res, mod1, mod2):
    if (mod1, mod2) in cca_res:
        return (mod1, mod2), False
    elif (mod2, mod1) in cca_res:
        return (mod2, mod1), True
    else:
        raise ValueError(f"No CCA result for pair: {mod1}, {mod2}")


def _extract_time_cca_window_result(pair_res, window_idx=None):
    """
    pair_res: cca_res[(m1, m2)] for time-CCA

    Returns
    -------
    wr : dict
        one entry in pair_res["window_results"]
    """
    if "window_results" not in pair_res:
        raise ValueError("This pair result is not a time-CCA result")

    window_results = pair_res["window_results"]
    if len(window_results) == 0:
        raise ValueError("No valid window_results")

    if window_idx is None:
        return window_results[0]

    for wr in window_results:
        if wr["window_idx"] == window_idx:
            return wr

    raise ValueError(f"window_idx={window_idx} not found in window_results")


def plot_cca_pair_scatter(
    cca_res,
    mod1,
    mod2,
    y=None,
    comp=0,
    title=None,
    window_idx=None,
):
    """
    Supports both:
    1) window-summary CCA
    2) time-CCA per window

    Parameters
    ----------
    cca_res : dict
    mod1, mod2 : str
    y : array-like or None
        For summary CCA:
            y should align with all windows; y[mask] will be used
        For time-CCA:
            if provided, it should align with time points of the selected window
            after valid_t_mask filtering, or already be length T_valid
    comp : int
    window_idx : int or None
        used only for time-CCA
    """
    key, reversed_order = _get_pair_key(cca_res, mod1, mod2)
    res = cca_res[key]

    if title is None:
        if res.get("mode", None) == "time_cca_per_window":
            title = f"time-CCA latent scatter: {mod1} vs {mod2} (comp {comp+1})"
            if window_idx is not None:
                title += f", window={window_idx}"
        else:
            title = f"CCA shared latent: {mod1} vs {mod2} (comp {comp+1})"

    # -------------------------------------------------
    # Case 1: window-summary CCA
    # -------------------------------------------------
    if "Xi_c" in res and "Xj_c" in res and res.get("mode", "") != "time_cca_per_window":
        Xi_c = res["Xi_c"]
        Xj_c = res["Xj_c"]
        mask = res["mask"]

        if reversed_order:
            Xi_c, Xj_c = Xj_c, Xi_c

        if comp >= Xi_c.shape[1]:
            raise ValueError(f"comp={comp} out of range, only {Xi_c.shape[1]} components")

        plt.figure(figsize=(5, 5))

        if y is None:
            plt.scatter(Xi_c[:, comp], Xj_c[:, comp], alpha=0.7)
        else:
            y = np.asarray(y)
            y_use = y[mask]
            classes = np.unique(y_use)

            for c in classes:
                idx = (y_use == c)
                plt.scatter(Xi_c[idx, comp], Xj_c[idx, comp], alpha=0.7, label=f"class {c}")
            plt.legend()

        plt.xlabel(f"{mod1} canonical variate {comp+1}")
        plt.ylabel(f"{mod2} canonical variate {comp+1}")
        plt.title(title)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.show()
        return

    # -------------------------------------------------
    # Case 2: time-CCA
    # -------------------------------------------------
    if "window_results" in res:
        wr = _extract_time_cca_window_result(res, window_idx=window_idx)

        Xi_c = wr["Xi_c"]   # [T_valid, K]
        Xj_c = wr["Xj_c"]   # [T_valid, K]

        if reversed_order:
            Xi_c, Xj_c = Xj_c, Xi_c

        if comp >= Xi_c.shape[1]:
            raise ValueError(f"comp={comp} out of range, only {Xi_c.shape[1]} components")

        plt.figure(figsize=(5, 5))

        if y is None:
            plt.plot(Xi_c[:, comp], Xj_c[:, comp], alpha=0.7)
        else:
            y = np.asarray(y)

            # 两种兼容方式：
            # 1) y 已经是 T_valid 长度
            # 2) y 是原始 T 长度，需要用 valid_t_mask 过滤
            if len(y) == Xi_c.shape[0]:
                y_use = y
            elif len(y) == len(wr["valid_t_mask"]):
                y_use = y[wr["valid_t_mask"]]
            else:
                raise ValueError(
                    f"For time-CCA, y length must be either T_valid={Xi_c.shape[0]} "
                    f"or raw T={len(wr['valid_t_mask'])}, got {len(y)}"
                )

            classes = np.unique(y_use)
            for c in classes:
                idx = (y_use == c)
                plt.plot(Xi_c[idx, comp], Xj_c[idx, comp], alpha=0.7, label=f"class {c}")
            plt.legend()

        plt.xlabel(f"{mod1} canonical variate {comp+1}")
        plt.ylabel(f"{mod2} canonical variate {comp+1}")
        plt.title(title)
        plt.axhline(0, linestyle="--", linewidth=1)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.show()
        return

    raise ValueError(f"Unrecognized result format for pair {key}")


def plot_time_cca_mean_timeseries_heatmap(
    cca_res,
    mod1,
    mod2,
    title=None,
    use_abs=False,
):
    """
    For time-CCA only.
    Plot mean latent time series heatmaps for the two modalities.

    Requires Xi_c_all / Xj_c_all to be available, i.e. all valid windows
    must share the same [T, K] shape.

    Heatmap shape:
        mean_Xi: [T, K]
        mean_Xj: [T, K]
    """
    key, reversed_order = _get_pair_key(cca_res, mod1, mod2)
    res = cca_res[key]

    if "Xi_c_all" not in res or "Xj_c_all" not in res:
        raise ValueError("This function only supports time-CCA results")

    Xi_c_all = res["Xi_c_all"]
    Xj_c_all = res["Xj_c_all"]

    if Xi_c_all is None or Xj_c_all is None:
        raise ValueError("Xi_c_all / Xj_c_all is None; valid windows likely do not share the same shape")

    if reversed_order:
        Xi_c_all, Xj_c_all = Xj_c_all, Xi_c_all

    mean_Xi = np.nanmean(Xi_c_all, axis=0)   # [T, K]
    mean_Xj = np.nanmean(Xj_c_all, axis=0)   # [T, K]

    if use_abs:
        mean_Xi = np.abs(mean_Xi)
        mean_Xj = np.abs(mean_Xj)

    if title is None:
        title = f"Mean time-CCA latent heatmap: {mod1} vs {mod2}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    im0 = axes[0].imshow(mean_Xi, aspect="auto", origin="lower")
    axes[0].set_title(f"{mod1} mean latent")
    axes[0].set_xlabel("CCA component")
    axes[0].set_ylabel("time step")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mean_Xj, aspect="auto", origin="lower")
    axes[1].set_title(f"{mod2} mean latent")
    axes[1].set_xlabel("CCA component")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_time_cca_component_timeseries(
    cca_res,
    mod1,
    mod2,
    comps=None,
    title=None,
):
    """
    For time-CCA only.
    Plot average latent time series across windows for selected components.

    Requires Xi_c_all / Xj_c_all to be available.
    """
    key, reversed_order = _get_pair_key(cca_res, mod1, mod2)
    res = cca_res[key]

    Xi_c_all = res.get("Xi_c_all", None)
    Xj_c_all = res.get("Xj_c_all", None)

    if Xi_c_all is None or Xj_c_all is None:
        raise ValueError("Xi_c_all / Xj_c_all is None; cannot compute mean component time series")

    if reversed_order:
        Xi_c_all, Xj_c_all = Xj_c_all, Xi_c_all

    mean_Xi = np.nanmean(Xi_c_all, axis=0)   # [T, K]
    mean_Xj = np.nanmean(Xj_c_all, axis=0)   # [T, K]

    K = mean_Xi.shape[1]
    if comps is None:
        comps = list(range(min(3, K)))

    if title is None:
        title = f"Mean latent component time series: {mod1} vs {mod2}"

    fig, axes = plt.subplots(len(comps), 1, figsize=(10, 2.8 * len(comps)), sharex=True)
    if len(comps) == 1:
        axes = [axes]

    t = np.arange(mean_Xi.shape[0])

    for ax, comp in zip(axes, comps):
        if comp >= K:
            raise ValueError(f"Requested comp={comp}, but only K={K}")

        ax.plot(t, mean_Xi[:, comp], label=f"{mod1} CC{comp+1}")
        ax.plot(t, mean_Xj[:, comp], label=f"{mod2} CC{comp+1}")
        ax.set_ylabel(f"CC{comp+1}")
        ax.legend()

    axes[-1].set_xlabel("time step")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_time_cca_component_trajectory(
    cca_res,
    mod1,
    mod2,
    dims=(0, 1),
    title=None,
    mark_start_end=True,
):
    """
    For time-CCA only.
    Plot the average latent trajectory in component space.

    Requires Xi_c_all / Xj_c_all to be available.

    dims=(0,1) means CC1 vs CC2 trajectory.
    """
    key, reversed_order = _get_pair_key(cca_res, mod1, mod2)
    res = cca_res[key]

    Xi_c_all = res.get("Xi_c_all", None)
    Xj_c_all = res.get("Xj_c_all", None)

    if Xi_c_all is None or Xj_c_all is None:
        raise ValueError("Xi_c_all / Xj_c_all is None; cannot compute mean trajectory")

    if reversed_order:
        Xi_c_all, Xj_c_all = Xj_c_all, Xi_c_all

    mean_Xi = np.nanmean(Xi_c_all, axis=0)   # [T, K]
    mean_Xj = np.nanmean(Xj_c_all, axis=0)   # [T, K]

    d1, d2 = dims
    K = mean_Xi.shape[1]
    if max(d1, d2) >= K:
        raise ValueError(f"Requested dims={dims}, but only K={K}")

    if title is None:
        title = f"Mean latent trajectory: {mod1} vs {mod2}"

    plt.figure(figsize=(6, 6))
    plt.plot(mean_Xi[:, d1], mean_Xi[:, d2], marker="o", alpha=0.8, label=f"{mod1}")
    plt.plot(mean_Xj[:, d1], mean_Xj[:, d2], marker="o", alpha=0.8, label=f"{mod2}")

    if mark_start_end:
        plt.scatter(mean_Xi[0, d1], mean_Xi[0, d2], marker="s", s=70)
        plt.scatter(mean_Xi[-1, d1], mean_Xi[-1, d2], marker="x", s=70)
        plt.scatter(mean_Xj[0, d1], mean_Xj[0, d2], marker="s", s=70)
        plt.scatter(mean_Xj[-1, d1], mean_Xj[-1, d2], marker="x", s=70)

    plt.xlabel(f"CC{d1+1}")
    plt.ylabel(f"CC{d2+1}")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ====================================
# cca
# ====================================

def _safe_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan

    if np.std(a[mask]) < 1e-10 or np.std(b[mask]) < 1e-10:
        return np.nan

    return np.corrcoef(a[mask], b[mask])[0, 1]

def _prepare_two_modalities_for_cca(
    Xi,
    Xj,
    mask,
    pca_dim=None,
):
    """
    Window-summary level CCA preparation.

    Xi: [N, Fi]
    Xj: [N, Fj]
    mask: [N]
    """
    Xi_use = Xi[mask]
    Xj_use = Xj[mask]

    imp_i = SimpleImputer(strategy="mean")
    imp_j = SimpleImputer(strategy="mean")

    Xi_use = imp_i.fit_transform(Xi_use)
    Xj_use = imp_j.fit_transform(Xj_use)

    sc_i = StandardScaler()
    sc_j = StandardScaler()

    Xi_use = sc_i.fit_transform(Xi_use)
    Xj_use = sc_j.fit_transform(Xj_use)

    if pca_dim is not None:
        di = min(pca_dim, Xi_use.shape[1], Xi_use.shape[0] - 1)
        dj = min(pca_dim, Xj_use.shape[1], Xj_use.shape[0] - 1)

        if di >= 1:
            Xi_use = PCA(n_components=di).fit_transform(Xi_use)
        if dj >= 1:
            Xj_use = PCA(n_components=dj).fit_transform(Xj_use)

    return Xi_use, Xj_use


def _prepare_two_time_series_for_cca(
    Xi,
    Xj,
    pca_dim=None,
    b_detrend=True,
):
    """
    Time-CCA preparation for one window.

    Xi: [T, Di] or [T]
    Xj: [T, Dj] or [T]

    Returns
    -------
    Xi_use : [T_valid, di]
    Xj_use : [T_valid, dj]
    valid_t_mask : [T]
    """
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)

    if Xi.ndim == 1:
        Xi = Xi[:, None]
    if Xj.ndim == 1:
        Xj = Xj[:, None]

    if b_detrend:
        Xi = detrend(Xi, axis=0, type="linear")
        Xj = detrend(Xj, axis=0, type="linear")

    if Xi.shape[0] != Xj.shape[0]:
        raise ValueError(f"time length mismatch: {Xi.shape[0]} vs {Xj.shape[0]}")

    if Xi.size == 0 or Xj.size == 0:
        raise ValueError("empty input")

    valid_i = ~np.isnan(Xi).all(axis=1)
    valid_j = ~np.isnan(Xj).all(axis=1)
    valid_t_mask = valid_i & valid_j

    Xi_use = Xi[valid_t_mask]
    Xj_use = Xj[valid_t_mask]

    if Xi_use.shape[0] < 3:
        raise ValueError("too few valid time points")

    imp_i = SimpleImputer(strategy="mean")
    imp_j = SimpleImputer(strategy="mean")

    Xi_use = imp_i.fit_transform(Xi_use)
    Xj_use = imp_j.fit_transform(Xj_use)

    sc_i = StandardScaler()
    sc_j = StandardScaler()

    Xi_use = sc_i.fit_transform(Xi_use)
    Xj_use = sc_j.fit_transform(Xj_use)

    if pca_dim is not None:
        di = min(pca_dim, Xi_use.shape[1], Xi_use.shape[0] - 1)
        dj = min(pca_dim, Xj_use.shape[1], Xj_use.shape[0] - 1)

        if di >= 1:
            Xi_use = PCA(n_components=di).fit_transform(Xi_use)
        if dj >= 1:
            Xj_use = PCA(n_components=dj).fit_transform(Xj_use)

    return Xi_use, Xj_use, valid_t_mask


def _remove_low_variance_cols(X, eps=1e-8):
    var = np.nanvar(X, axis=0)
    keep = var > eps
    return X[:, keep], keep

def compute_time_cca_for_one_window(
    Xi,
    Xj,
    pca_dim=10,
    n_components=3,
    max_iter=1000,
    b_detrend=True,
    detrend_type="linear",
    var_eps=1e-8,
):
    Xi = np.asarray(Xi, dtype=float)
    Xj = np.asarray(Xj, dtype=float)

    if Xi.ndim == 1:
        Xi = Xi[:, None]
    if Xj.ndim == 1:
        Xj = Xj[:, None]

    if Xi.shape[0] != Xj.shape[0]:
        raise ValueError("Xi and Xj must have the same number of time points")

    if b_detrend:
        Xi = detrend(Xi, axis=0, type=detrend_type)
        Xj = detrend(Xj, axis=0, type=detrend_type)

    Xi_use, Xj_use, valid_t_mask = _prepare_two_time_series_for_cca(
        Xi, Xj, pca_dim=pca_dim
    )

    Xi_use, keep_i = _remove_low_variance_cols(Xi_use, eps=var_eps)
    Xj_use, keep_j = _remove_low_variance_cols(Xj_use, eps=var_eps)

    if Xi_use.shape[1] == 0 or Xj_use.shape[1] == 0:
        raise ValueError("no usable features after low-variance filtering")

    rank_i = np.linalg.matrix_rank(Xi_use)
    rank_j = np.linalg.matrix_rank(Xj_use)

    k = min(
        n_components,
        Xi_use.shape[1],
        Xj_use.shape[1],
        Xi_use.shape[0] - 1,
        rank_i,
        rank_j,
    )

    if k < 1:
        raise ValueError("k < 1 for CCA")

    cca = CCA(n_components=k, max_iter=max_iter)
    Xi_c, Xj_c = cca.fit_transform(Xi_use, Xj_use)

    # 再做一次保险，防止某个 canonical variate 塌缩
    good = []
    corrs = []
    for i in range(k):
        sx = np.nanstd(Xi_c[:, i])
        sy = np.nanstd(Xj_c[:, i])
        if sx < var_eps or sy < var_eps:
            continue
        good.append(i)
        corrs.append(_safe_corr(Xi_c[:, i], Xj_c[:, i]))

    if len(good) == 0:
        raise ValueError("all canonical components collapsed")

    Xi_c = Xi_c[:, good]
    Xj_c = Xj_c[:, good]

    return {
        "corrs": corrs,
        "Xi_c": Xi_c,
        "Xj_c": Xj_c,
        "cca_model": cca,
        "valid_t_mask": valid_t_mask,
    }


def compute_pairwise_cca(
    concat_data,
    modality_keys=None,
    n_components=3,
    pca_dim=10,
    bUseWindowSummary=True,
    window_feature_mode="flatten",
    min_windows=20,
    min_T_valid=8,
    max_iter=2000,
    verbose=True,
):
    """
    Parameters
    ----------
    concat_data : dict
        must contain concat_data["sequences"]

    modality_keys : list[str] or None

    bUseWindowSummary : bool
        True:
            window -> summary feature -> CCA across windows
        False:
            do time-CCA separately for each window

    window_feature_mode : str
        used only when bUseWindowSummary=True

    Returns
    -------
    cca_res : dict

    Case 1: bUseWindowSummary=True
        cca_res[(m1, m2)] = {
            "mode": "window_summary",
            "corrs": [K],
            "n_windows": n,
            "Xi_c": [n, K],
            "Xj_c": [n, K],
            "cca_model": fitted CCA,
            "mask": [N] bool,
        }

    Case 2: bUseWindowSummary=False
        cca_res[(m1, m2)] = {
            "mode": "time_cca_per_window",
            "n_windows": n_valid_windows,
            "mask": [N] bool,  # pair-valid window mask
            "window_results": [
                {
                    "window_idx": w,
                    "corrs": [K],
                    "Xi_c": [T_valid, K],
                    "Xj_c": [T_valid, K],
                    "cca_model": fitted CCA,
                    "valid_t_mask": [T] bool,
                },
                ...
            ],
            "mean_corrs": [K] or None,
            "Xi_c_all": [N, T, K] if all valid windows share same shape else None,
            "Xj_c_all": [N, T, K] if all valid windows share same shape else None,
        }
    """
    sequences = concat_data["sequences"]

    if modality_keys is None:
        modality_keys = list(sequences.keys())

    cca_res = {}

    # =========================================================
    # A) window-summary CCA
    # =========================================================
    if bUseWindowSummary:
        feat_dict, valid_mask_dict = build_modality_feature_dict(
            concat_data,
            modality_keys=modality_keys,
            mode=window_feature_mode,
            verbose=verbose,
        )

        for i in range(len(modality_keys)):
            for j in range(i + 1, len(modality_keys)):
                m1 = modality_keys[i]
                m2 = modality_keys[j]

                if m1 not in feat_dict or m2 not in feat_dict:
                    continue

                mask = valid_mask_dict[m1] & valid_mask_dict[m2]
                n = int(mask.sum())

                if n < max(min_windows, n_components + 2):
                    if verbose:
                        print(f"[skip-summary] {m1} vs {m2}: too few shared windows ({n})")
                    continue

                Xi, Xj = _prepare_two_modalities_for_cca(
                    feat_dict[m1], feat_dict[m2], mask, pca_dim=pca_dim
                )

                k = min(n_components, Xi.shape[1], Xj.shape[1], Xi.shape[0] - 1)
                if k < 1:
                    if verbose:
                        print(f"[skip-summary] {m1} vs {m2}: k < 1")
                    continue

                cca = CCA(n_components=k, max_iter=max_iter)
                Xi_c, Xj_c = cca.fit_transform(Xi, Xj)

                corrs = [_safe_corr(Xi_c[:, d], Xj_c[:, d]) for d in range(k)]

                cca_res[(m1, m2)] = {
                    "mode": "window_summary",
                    "corrs": corrs,
                    "n_windows": n,
                    "Xi_c": Xi_c,      # [n_windows, K]
                    "Xj_c": Xj_c,      # [n_windows, K]
                    "cca_model": cca,
                    "mask": mask,
                }

                if verbose:
                    print(f"[CCA-summary] {m1} vs {m2}: " + ", ".join([f"{c:.3f}" for c in corrs]))

        return cca_res

    # =========================================================
    # B) time-CCA per window
    # =========================================================
    valid_mask_dict = {}
    for mod in modality_keys:
        seq_list = sequences[mod]
        valid = []
        for x in seq_list:
            if x is None:
                valid.append(False)
                continue

            x = np.asarray(x, dtype=float)
            if x.ndim == 1:
                x = x[:, None]

            ok = (x.size > 0) and (not np.isnan(x).all())
            valid.append(ok)

        valid_mask_dict[mod] = np.asarray(valid, dtype=bool)

        if verbose:
            print(f"[{mod}] valid windows = {valid_mask_dict[mod].sum()}/{len(valid_mask_dict[mod])}")

    for i in range(len(modality_keys)):
        for j in range(i + 1, len(modality_keys)):
            m1 = modality_keys[i]
            m2 = modality_keys[j]

            seq1 = sequences[m1]
            seq2 = sequences[m2]

            mask = valid_mask_dict[m1] & valid_mask_dict[m2]
            valid_window_idx = np.where(mask)[0]

            if len(valid_window_idx) == 0:
                if verbose:
                    print(f"[skip-time] {m1} vs {m2}: no shared valid windows")
                continue

            window_results = []

            for w in valid_window_idx:
                Xi = seq1[w]
                Xj = seq2[w]

                try:
                    res_w = compute_time_cca_for_one_window(
                        Xi,
                        Xj,
                        pca_dim=pca_dim,
                        n_components=n_components,
                        max_iter=max_iter,
                    )

                    T_valid = res_w["Xi_c"].shape[0]
                    if T_valid < min_T_valid:
                        if verbose:
                            print(f"[skip-time-window] {m1} vs {m2}, window={w}: too few valid time points ({T_valid})")
                        continue

                    window_results.append({
                        "window_idx": int(w),
                        "corrs": res_w["corrs"],
                        "Xi_c": res_w["Xi_c"],                 # [T_valid, K]
                        "Xj_c": res_w["Xj_c"],                 # [T_valid, K]
                        "cca_model": res_w["cca_model"],
                        "valid_t_mask": res_w["valid_t_mask"],
                    })

                    if verbose:
                        print(
                            f"[CCA-time] {m1} vs {m2}, window={w}: "
                            + ", ".join([f"{c:.3f}" for c in res_w["corrs"]])
                        )

                except Exception as e:
                    if verbose:
                        print(f"[skip-time-window] {m1} vs {m2}, window={w}: {e}")

            n_valid = len(window_results)
            if n_valid == 0:
                if verbose:
                    print(f"[skip-time] {m1} vs {m2}: no usable windows after time-CCA")
                continue

            # mean canonical corr across windows
            max_k = max(len(r["corrs"]) for r in window_results)
            corr_mat = np.full((n_valid, max_k), np.nan, dtype=float)
            for ii, r in enumerate(window_results):
                corr_mat[ii, :len(r["corrs"])] = r["corrs"]
            mean_corrs = np.nanmean(corr_mat, axis=0).tolist()

            xi_shapes = [r["Xi_c"].shape for r in window_results]
            xj_shapes = [r["Xj_c"].shape for r in window_results]

            if len(set(xi_shapes)) == 1 and len(set(xj_shapes)) == 1:
                Xi_c_all = np.stack([r["Xi_c"] for r in window_results], axis=0)
                Xj_c_all = np.stack([r["Xj_c"] for r in window_results], axis=0)
            else:
                Xi_c_all = None
                Xj_c_all = None

            cca_res[(m1, m2)] = {
                "mode": "time_cca_per_window",
                "n_windows": n_valid,
                "mask": mask,                    # [N_windows_total]
                "window_results": window_results,
                "mean_corrs": mean_corrs,
                "Xi_c_all": Xi_c_all,           # [N, T, K] if possible else None
                "Xj_c_all": Xj_c_all,           # [N, T, K] if possible else None
            }

            if verbose:
                print(
                    f"[CCA-time-summary] {m1} vs {m2}: "
                    f"usable_windows={n_valid}, "
                    f"mean_corrs="
                    + ", ".join([f"{c:.3f}" for c in mean_corrs])
                )

    return cca_res

def cca_results_to_matrix(cca_res, modality_keys=None, mode="first"):
    """
    Convert pairwise CCA results to a symmetric matrix.

    Parameters
    ----------
    cca_res : dict
        Output of compute_pairwise_cca(...)

    modality_keys : list[str] or None
        Order of modalities in the matrix

    mode : str
        For window-summary CCA:
            "first" -> first canonical corr
            "mean"  -> mean of canonical corrs

        For time-CCA:
            "first" -> first entry of mean_corrs
            "mean"  -> mean of mean_corrs across canonical dims

    Returns
    -------
    mods : list[str]
    M : [n_mod, n_mod]
    """
    if modality_keys is None:
        mods = sorted(list(set(
            [k[0] for k in cca_res.keys()] + [k[1] for k in cca_res.keys()]
        )))
    else:
        mods = list(modality_keys)

    M = np.full((len(mods), len(mods)), np.nan, dtype=float)

    for i, m in enumerate(mods):
        M[i, i] = 1.0

    for (m1, m2), v in cca_res.items():
        i = mods.index(m1)
        j = mods.index(m2)

        # -------------------------------------------------
        # Case 1: window-summary CCA
        # -------------------------------------------------
        if "corrs" in v:
            corrs = np.asarray(v["corrs"], dtype=float)

        # -------------------------------------------------
        # Case 2: time-CCA per window
        # use mean_corrs if available
        # -------------------------------------------------
        elif "mean_corrs" in v:
            corrs = np.asarray(v["mean_corrs"], dtype=float)

        # -------------------------------------------------
        # Fallback: compute from window_results
        # -------------------------------------------------
        elif "window_results" in v:
            max_k = max(len(r["corrs"]) for r in v["window_results"])
            corr_mat = np.full((len(v["window_results"]), max_k), np.nan, dtype=float)

            for ii, r in enumerate(v["window_results"]):
                rr = np.asarray(r["corrs"], dtype=float)
                corr_mat[ii, :len(rr)] = rr

            corrs = np.nanmean(corr_mat, axis=0)

        else:
            raise ValueError(f"Unrecognized cca result format for pair {(m1, m2)}")

        if corrs.size == 0 or np.all(np.isnan(corrs)):
            val = np.nan
        else:
            if mode == "first":
                val = corrs[0]
            elif mode == "mean":
                val = np.nanmean(corrs)
            else:
                raise ValueError("mode must be 'first' or 'mean'")

        M[i, j] = val
        M[j, i] = val

    return mods, M