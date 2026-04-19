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
from scipy.signal import spectrogram

# ====================================
# basic utils
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


def _remove_low_variance_cols(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    var = np.nanvar(X, axis=0)
    keep = var > eps
    return X[:, keep], keep


def _cv_auc(X, y, cv=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    model = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    )

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    aucs = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")
    return np.mean(aucs), np.std(aucs)


def _get_pair_key(cca_res, mod1, mod2):
    if (mod1, mod2) in cca_res:
        return (mod1, mod2), False
    elif (mod2, mod1) in cca_res:
        return (mod2, mod1), True
    else:
        raise ValueError(f"No CCA result for pair: {mod1}, {mod2}")


# ====================================
# window feature extraction
# ====================================

import numpy as np
from scipy.signal import welch


def _compute_bandpower_features(
    x,
    fs,
    bands=None,
    relative=False,
    welch_nperseg=None,
):
    """
    x: [T, D]
    return: [D * n_bands]
    """
    if bands is None:
        bands = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 80),
        }

    T, D = x.shape
    feat_all = []

    for d in range(D):
        xd = x[:, d]
        mask = np.isfinite(xd)

        if mask.sum() < 2:
            feat_all.append(np.full(len(bands), np.nan))
            continue

        xd_valid = xd[mask]

        nperseg = welch_nperseg
        if nperseg is None:
            nperseg = min(256, len(xd_valid))

        if nperseg < 2:
            feat_all.append(np.full(len(bands), np.nan))
            continue

        freqs, psd = welch(
            xd_valid,
            fs=fs,
            nperseg=nperseg,
            scaling="density",
        )

        total_power = np.trapezoid(psd, freqs) + 1e-12

        bp_list = []
        for _, (f_low, f_high) in bands.items():
            band_mask = (freqs >= f_low) & (freqs < f_high)

            if not np.any(band_mask):
                bp = np.nan
            else:
                bp = np.trapezoid(psd[band_mask], freqs[band_mask])

            if relative and np.isfinite(bp):
                bp = bp / total_power

            bp_list.append(bp)

        feat_all.append(np.asarray(bp_list, dtype=float))

    return np.concatenate(feat_all, axis=0)


def _window_to_feature(
    x,
    mode="channel_mean",
    fs=None,
    bands=None,
    relative_bandpower=False,
    welch_nperseg=None,
):
    """
    x: [T, D] or [T]
    return: [F]

    mode:
        - "channel_mean": [T, D] -> [T]
        - "flatten":      [T, D] -> [T*D]
        - "time_mean":    [T, D] -> [D]
        - "stats":        [T, D] -> [5*D]
        - "bandpower":    [T, D] -> [n_bands * D]
        - "stats_bandpower": stats + bandpower
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

    elif mode == "bandpower":
        if fs is None:
            raise ValueError("`fs` must be provided when mode='bandpower'")
        return _compute_bandpower_features(
            x,
            fs=fs,
            bands=bands,
            relative=relative_bandpower,
            welch_nperseg=welch_nperseg,
        )

    elif mode == "stats_bandpower":
        if fs is None:
            raise ValueError("`fs` must be provided when mode='stats_bandpower'")

        # stats
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

        # bandpower
        bp_feat = _compute_bandpower_features(
            x,
            fs=fs,
            bands=bands,
            relative=relative_bandpower,
            welch_nperseg=welch_nperseg,
        )
        feat_list.append(bp_feat)

        return np.concatenate(feat_list, axis=0)

    else:
        raise ValueError(
            "mode must be one of "
            "['channel_mean', 'flatten', 'time_mean', 'stats', 'bandpower', 'stats_bandpower']"
        )


def build_modality_feature_dict(concat_data, modality_keys=None, mode="channel_mean", verbose=True, fs=250):
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
        if mod not in sequences:
            if verbose:
                print(f"[skip] {mod}: not in concat_data['sequences']")
            continue

        seq_list = sequences[mod]
        feats = []
        valid = []

        for x in seq_list:
            f = _window_to_feature(
                x,
                mode=mode,
                fs=fs,
                bands={
                    "delta": (1, 4),
                    "theta": (4, 8),
                    "alpha": (8, 12),
                    "beta": (12, 30),
                    "gamma": (30, 80),
                    "high_gamma": (80, 150),
                },
                relative_bandpower=True,
            )
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
# subject split utils
# ====================================

def split_concat_data_by_sub(concat_data, sub_col="sub_id", modality_keys=None, verbose=True):
    """
    Split one concat_data into {sub: concat_data_sub} using window_df[sub_col].

    Parameters
    ----------
    concat_data : dict
        {
            "window_df": DataFrame,
            "sequences": {mod: list of window arrays}
        }

    Returns
    -------
    by_sub : dict
        {
            sub1: {
                "window_df": df_sub,
                "sequences": {
                    mod1: [...],
                    mod2: [...],
                }
            },
            ...
        }
    """
    if "window_df" not in concat_data:
        raise ValueError("concat_data must contain 'window_df'")
    if "sequences" not in concat_data:
        raise ValueError("concat_data must contain 'sequences'")

    df = concat_data["window_df"].copy().reset_index(drop=True)
    sequences = concat_data["sequences"]

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")

    if modality_keys is None:
        modality_keys = list(sequences.keys())

    n = len(df)
    for mod in modality_keys:
        if mod not in sequences:
            continue
        if len(sequences[mod]) != n:
            raise ValueError(
                f"len(window_df)={n} does not match len(sequences[{mod}])={len(sequences[mod])}"
            )

    sub_ids = sorted(df[sub_col].dropna().unique())
    by_sub = {}

    for sub in sub_ids:
        idx = np.where(df[sub_col].values == sub)[0]
        df_sub = df.iloc[idx].reset_index(drop=True)

        seq_sub = {}
        for mod in modality_keys:
            if mod not in sequences:
                continue
            seq_sub[mod] = [sequences[mod][i] for i in idx]

        by_sub[sub] = {
            "window_df": df_sub,
            "sequences": seq_sub,
        }

        if verbose:
            print(f"[split] sub={sub}: n_windows={len(df_sub)}")

    return by_sub


def get_label_dict_from_concat_data(
    concat_data,
    y_col,
    threshold=0.1,
    sub_col="sub_id",
    mode="binary",      # "binary" | "continuous" | "soft"
    temperature=0.05,
):
    df = concat_data["window_df"].copy().reset_index(drop=True)

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found")
    if y_col not in df.columns:
        raise ValueError(f"`{y_col}` not found")

    out = {}
    sub_ids = df[sub_col].unique()

    for sub in sub_ids:
        idx = (df[sub_col] == sub).values
        y = df.loc[idx, y_col].to_numpy()

        if mode == "binary":
            y = (y > threshold).astype(int)

        elif mode == "soft":
            y = 1 / (1 + np.exp(-(y - threshold) / temperature))

        elif mode == "continuous":
            pass

        else:
            raise ValueError("mode must be binary / soft / continuous")

        out[sub] = y

    return out


# ====================================
# subject-wise feature -> PCA latent
# ====================================

def build_subjectwise_modality_latents(
    concat_data,
    modality_keys=None,
    feature_mode="stats",
    pca_dim=10,
    sub_col="sub_id",
    verbose=True,
    fs=250,
):
    """
    Split concat_data by subject, then do feature extraction + PCA within each subject.

    Returns
    -------
    res : dict
        {
            sub: {
                "feat_dict": {mod: [N_sub, F_mod]},
                "valid_mask_dict": {mod: [N_sub] bool},
                "latent_dict": {mod: [N_sub, K_mod]},
                "pca_dict": {mod: {...}},
                "window_df": df_sub,
            }
        }
    """
    by_sub = split_concat_data_by_sub(
        concat_data,
        sub_col=sub_col,
        modality_keys=modality_keys,
        verbose=verbose,
    )

    res = {}

    for sub, concat_sub in by_sub.items():
        feat_dict, valid_mask_dict = build_modality_feature_dict(
            concat_sub,
            modality_keys=modality_keys,
            mode=feature_mode,
            verbose=verbose,
            fs=fs,
        )

        latent_dict = {}
        pca_dict = {}

        for mod, X in feat_dict.items():
            mask = valid_mask_dict[mod]
            X_use = X[mask]

            if X_use.shape[0] < 2:
                if verbose:
                    print(f"[skip PCA] sub={sub}, mod={mod}: too few valid windows")
                latent_dict[mod] = np.full((X.shape[0], 1), np.nan)
                pca_dict[mod] = None
                continue

            imp = SimpleImputer(strategy="mean")
            sc = StandardScaler()

            Xz = imp.fit_transform(X_use)
            Xz = sc.fit_transform(Xz)

            k = min(pca_dim, Xz.shape[0] - 1, Xz.shape[1])

            if k < 1:
                latent_dict[mod] = np.full((X.shape[0], 1), np.nan)
                pca_dict[mod] = None
                continue

            pca = PCA(n_components=k)
            Z_use = pca.fit_transform(Xz)

            Z_full = np.full((X.shape[0], k), np.nan)
            Z_full[mask] = Z_use

            latent_dict[mod] = Z_full
            pca_dict[mod] = {
                "imputer": imp,
                "scaler": sc,
                "pca": pca,
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }

            if verbose:
                print(
                    f"[subject PCA] sub={sub}, mod={mod}, "
                    f"X_use.shape={X_use.shape}, Z.shape={Z_full.shape}"
                )

        res[sub] = {
            "feat_dict": feat_dict,
            "valid_mask_dict": valid_mask_dict,
            "latent_dict": latent_dict,
            "pca_dict": pca_dict,
            "window_df": concat_sub.get("window_df", None),
        }

    return res


def stack_subjectwise_latents(
    subject_latent_res,
    modality_keys=None,
    y_dict=None,
    verbose=True,
):
    """
    Stack subject-wise PCA latents across subjects.
    Each modality is cropped to a common latent dim = min subject latent dims.

    Returns
    -------
    stacked : dict
        {
            "latent_dict": {mod: [N_all, K_common]},
            "valid_mask_dict": {mod: [N_all] bool},
            "y": [N_all] or None,
            "sub": [N_all],
            "common_k": {mod: int},
        }
    """
    if modality_keys is None:
        modality_keys = sorted(list(set(
            m for sub in subject_latent_res
            for m in subject_latent_res[sub]["latent_dict"].keys()
        )))

    common_k = {}
    for mod in modality_keys:
        ks = []
        for sub in subject_latent_res:
            if mod in subject_latent_res[sub]["latent_dict"]:
                Z = subject_latent_res[sub]["latent_dict"][mod]
                if Z.ndim == 2 and Z.shape[1] >= 1:
                    ks.append(Z.shape[1])
        if len(ks) > 0:
            common_k[mod] = min(ks)

    latent_dict_out = {}
    valid_mask_out = {}
    y_blocks = []
    sub_blocks = []

    for mod in modality_keys:
        if mod not in common_k:
            continue

        X_blocks = []
        valid_blocks = []

        for sub, sub_res in subject_latent_res.items():
            if mod not in sub_res["latent_dict"]:
                continue

            Z = sub_res["latent_dict"][mod][:, :common_k[mod]]
            valid = np.isfinite(Z).any(axis=1)

            X_blocks.append(Z)
            valid_blocks.append(valid)

        latent_dict_out[mod] = np.concatenate(X_blocks, axis=0)
        valid_mask_out[mod] = np.concatenate(valid_blocks, axis=0)

        if verbose:
            print(f"[stack latent] mod={mod}, shape={latent_dict_out[mod].shape}, K_common={common_k[mod]}")

    for sub, sub_res in subject_latent_res.items():
        n_sub = None
        for _, Z in sub_res["latent_dict"].items():
            n_sub = Z.shape[0]
            break
        if n_sub is None:
            continue

        sub_blocks.append(np.asarray([sub] * n_sub))

        if y_dict is not None and sub in y_dict:
            y_sub = np.asarray(y_dict[sub])
            if len(y_sub) != n_sub:
                raise ValueError(f"sub={sub}: len(y_sub)={len(y_sub)} != n_windows={n_sub}")
            y_blocks.append(y_sub)
        else:
            y_blocks.append(np.full(n_sub, np.nan))

    return {
        "latent_dict": latent_dict_out,
        "valid_mask_dict": valid_mask_out,
        "y": np.concatenate(y_blocks, axis=0) if len(y_blocks) > 0 else None,
        "sub": np.concatenate(sub_blocks, axis=0) if len(sub_blocks) > 0 else None,
        "common_k": common_k,
    }


# ====================================
# synergy / decoding
# ====================================

def compute_pairwise_synergy_matrix_from_latents(
    stacked_latent_data,
    cv=5,
    verbose=True,
):
    """
    stacked_latent_data: output of stack_subjectwise_latents(...)
    """
    latent_dict = stacked_latent_data["latent_dict"]
    valid_mask_dict = stacked_latent_data["valid_mask_dict"]
    y = np.asarray(stacked_latent_data["y"], dtype=float)

    mods = list(latent_dict.keys())
    n_mod = len(mods)

    single_auc = {}
    single_std = {}

    for mod in mods:
        X = latent_dict[mod]
        mask = valid_mask_dict[mod] & np.isfinite(y)

        X_use = X[mask]
        y_use = y[mask]

        if len(np.unique(y_use)) < 2:
            if verbose:
                print(f"[skip single latent] {mod}: y has <2 classes")
            single_auc[mod] = np.nan
            single_std[mod] = np.nan
            continue

        auc_mean, auc_std = _cv_auc(X_use, y_use, cv=cv)
        single_auc[mod] = auc_mean
        single_std[mod] = auc_std

        if verbose:
            print(f"[single latent] {mod}: AUC={auc_mean:.4f} ± {auc_std:.4f}")

    synergy_mat = np.full((n_mod, n_mod), np.nan)
    joint_auc_mat = np.full((n_mod, n_mod), np.nan)

    for i, mod_i in enumerate(mods):
        synergy_mat[i, i] = 0.0
        joint_auc_mat[i, i] = single_auc[mod_i]

    for mod_i, mod_j in combinations(mods, 2):
        i = mods.index(mod_i)
        j = mods.index(mod_j)

        Xi = latent_dict[mod_i]
        Xj = latent_dict[mod_j]
        mask = valid_mask_dict[mod_i] & valid_mask_dict[mod_j] & np.isfinite(y)

        X_joint = np.concatenate([Xi[mask], Xj[mask]], axis=1)
        y_use = y[mask]

        if len(np.unique(y_use)) < 2:
            if verbose:
                print(f"[skip pair latent] {mod_i}+{mod_j}: y has <2 classes")
            continue

        joint_auc, joint_std = _cv_auc(X_joint, y_use, cv=cv)
        synergy = joint_auc - max(single_auc[mod_i], single_auc[mod_j])

        synergy_mat[i, j] = synergy
        synergy_mat[j, i] = synergy
        joint_auc_mat[i, j] = joint_auc
        joint_auc_mat[j, i] = joint_auc

        if verbose:
            print(
                f"[pair latent] {mod_i}+{mod_j}: "
                f"AUC={joint_auc:.4f} ± {joint_std:.4f}, synergy={synergy:.4f}"
            )

    return {
        "modalities": mods,
        "single_auc": single_auc,
        "single_std": single_std,
        "joint_auc_mat": joint_auc_mat,
        "synergy_mat": synergy_mat,
    }


# ====================================
# plotting matrices
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


# ====================================
# subject-wise CCA
# ====================================

def compute_pairwise_cca_subjectwise(
    subject_latent_res,
    modality_keys=None,
    n_components=3,
    min_windows=20,
    max_iter=2000,
    verbose=True,
):
    """
    Do CCA within each subject on subject-specific PCA latents, then aggregate across subjects.

    Returns
    -------
    cca_res[(m1, m2)] = {
        "mode": "subjectwise_window_summary",
        "per_subject": [
            {
                "sub": sub,
                "corrs": [K],
                "n_windows": n,
                "Xi_c": [n, K],
                "Xj_c": [n, K],
                "mask": [N_sub] bool,
                "cca_model": fitted CCA,
            },
            ...
        ],
        "mean_corrs": [K],
        "Xi_c_all": [N_all, K_common],
        "Xj_c_all": [N_all, K_common],
        "sub_all": [N_all],
    }
    """
    out = {}

    if modality_keys is None:
        modality_keys = sorted(list(set(
            m for sub in subject_latent_res
            for m in subject_latent_res[sub]["latent_dict"].keys()
        )))

    for m1, m2 in combinations(modality_keys, 2):
        per_subject = []
        Xi_c_all = []
        Xj_c_all = []
        sub_all = []

        for sub, sub_res in subject_latent_res.items():
            if m1 not in sub_res["latent_dict"] or m2 not in sub_res["latent_dict"]:
                continue

            Xi = sub_res["latent_dict"][m1]
            Xj = sub_res["latent_dict"][m2]

            mask = np.isfinite(Xi).any(axis=1) & np.isfinite(Xj).any(axis=1)
            n = int(mask.sum())

            if n < max(min_windows, n_components + 2):
                if verbose:
                    print(f"[skip sub CCA] sub={sub}, {m1} vs {m2}: n={n}")
                continue

            Xi_use = Xi[mask]
            Xj_use = Xj[mask]

            Xi_use, keep_i = _remove_low_variance_cols(Xi_use)
            Xj_use, keep_j = _remove_low_variance_cols(Xj_use)

            if Xi_use.shape[1] == 0 or Xj_use.shape[1] == 0:
                if verbose:
                    print(f"[skip sub CCA] sub={sub}, {m1} vs {m2}: no usable cols")
                continue

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
                if verbose:
                    print(f"[skip sub CCA] sub={sub}, {m1} vs {m2}: k < 1")
                continue

            cca = CCA(n_components=k, max_iter=max_iter)
            Xi_c, Xj_c = cca.fit_transform(Xi_use, Xj_use)

            good = []
            corrs = []
            for d in range(k):
                sx = np.nanstd(Xi_c[:, d])
                sy = np.nanstd(Xj_c[:, d])
                if sx < 1e-8 or sy < 1e-8:
                    continue
                good.append(d)
                corrs.append(_safe_corr(Xi_c[:, d], Xj_c[:, d]))

            if len(good) == 0:
                if verbose:
                    print(f"[skip sub CCA] sub={sub}, {m1} vs {m2}: all components collapsed")
                continue

            Xi_c = Xi_c[:, good]
            Xj_c = Xj_c[:, good]

            per_subject.append({
                "sub": sub,
                "corrs": corrs,
                "n_windows": n,
                "Xi_c": Xi_c,
                "Xj_c": Xj_c,
                "mask": mask,
                "cca_model": cca,
            })

            Xi_c_all.append(Xi_c)
            Xj_c_all.append(Xj_c)
            sub_all.extend([sub] * Xi_c.shape[0])

            if verbose:
                print(
                    f"[subject CCA] sub={sub}, {m1} vs {m2}: "
                    + ", ".join(f"{c:.3f}" for c in corrs)
                )

        if len(per_subject) == 0:
            continue

        max_k = max(len(x["corrs"]) for x in per_subject)
        corr_mat = np.full((len(per_subject), max_k), np.nan)
        for i, r in enumerate(per_subject):
            corr_mat[i, :len(r["corrs"])] = r["corrs"]

        common_k = min(r["Xi_c"].shape[1] for r in per_subject)
        Xi_c_all_crop = np.concatenate([r["Xi_c"][:, :common_k] for r in per_subject], axis=0)
        Xj_c_all_crop = np.concatenate([r["Xj_c"][:, :common_k] for r in per_subject], axis=0)

        out[(m1, m2)] = {
            "mode": "subjectwise_window_summary",
            "per_subject": per_subject,
            "mean_corrs": np.nanmean(corr_mat, axis=0).tolist(),
            "Xi_c_all": Xi_c_all_crop,
            "Xj_c_all": Xj_c_all_crop,
            "sub_all": np.asarray(sub_all),
        }

    return out


def cca_results_to_matrix(cca_res, modality_keys=None, mode="first"):
    """
    Convert pairwise subject-wise CCA results to a symmetric matrix.
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

        if "mean_corrs" in v:
            corrs = np.asarray(v["mean_corrs"], dtype=float)
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


# ====================================
# CCA plotting
# ====================================

def plot_cca_pair_scatter(
    cca_res,
    mod1,
    mod2,
    y=None,
    comp=0,
    title=None,
):
    """
    For subject-wise pooled canonical variates after within-subject CCA.
    """
    key, reversed_order = _get_pair_key(cca_res, mod1, mod2)
    res = cca_res[key]

    Xi_c = res["Xi_c_all"]
    Xj_c = res["Xj_c_all"]

    if reversed_order:
        Xi_c, Xj_c = Xj_c, Xi_c

    if comp >= Xi_c.shape[1]:
        raise ValueError(f"comp={comp} out of range, only {Xi_c.shape[1]} components")

    if title is None:
        title = f"Subject-wise pooled CCA latent: {mod1} vs {mod2} (comp {comp+1})"

    plt.figure(figsize=(5, 5))

    if y is None:
        plt.plot(Xi_c[:, comp], Xj_c[:, comp], alpha=0.7)
    else:
        y = np.asarray(y)
        if len(y) != Xi_c.shape[0]:
            raise ValueError(f"len(y)={len(y)} must equal pooled latent rows={Xi_c.shape[0]}")
        classes = np.unique(y)

        for c in classes:
            idx = (y == c)
            plt.plot(Xi_c[idx, comp], Xj_c[idx, comp], alpha=0.7, label=f"class {c}")
        plt.legend()

    plt.xlabel(f"{mod1} canonical variate {comp+1}")
    plt.ylabel(f"{mod2} canonical variate {comp+1}")
    plt.title(title)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()


def plot_cca_corr_heatmap_from_matrix(
    cca_res,
    modality_keys=None,
    mode="first",
    title="CCA correlation matrix",
):
    mods, M = cca_results_to_matrix(cca_res, modality_keys=modality_keys, mode=mode)
    plot_matrix(
        M,
        labels=mods,
        title=title,
        cbar_label="CCA corr",
        fmt=".3f",
        vmin=0,
        vmax=1,
    )


# ====================================
# optional: subject-wise time-trajectory PCA from raw sequences
# ====================================

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
                f"Found unequal window shape: first shape={ref_shape}, current shape={x.shape}"
            )

        keep_idx_local.append(i)
        X_list.append(x.reshape(-1))

    if len(X_list) == 0:
        return None, []

    return np.asarray(X_list, dtype=float), keep_idx_local


def run_subjectwise_trajectory_pca(
    concat_data,
    modality,
    n_components=5,
    sub_col="sub_id",
    verbose=True,
):
    """
    PCA on raw trajectories within each subject for one modality.

    Returns
    -------
    results_by_sub : dict
        {
            sub: {
                "modality": modality,
                "pca": pca,
                "explained_variance_ratio": ...,
                "Z": [N_keep, K],
                "keep_idx_local": [...],
                "window_df_kept": ...
            }
        }
    """
    if "window_df" not in concat_data:
        raise ValueError("concat_data must contain 'window_df'")
    if "sequences" not in concat_data:
        raise ValueError("concat_data must contain 'sequences'")
    if modality not in concat_data["sequences"]:
        raise ValueError(f"{modality} not in concat_data['sequences']")

    df = concat_data["window_df"].copy().reset_index(drop=True)
    seq_list = concat_data["sequences"][modality]

    if len(df) != len(seq_list):
        raise ValueError(
            f"len(window_df)={len(df)} does not match "
            f"len(sequences[{modality}])={len(seq_list)}"
        )

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")

    results_by_sub = {}

    for sub in sorted(df[sub_col].dropna().unique()):
        idx = np.where(df[sub_col].values == sub)[0]
        df_sub = df.iloc[idx].reset_index(drop=True)
        seq_sub = [seq_list[i] for i in idx]

        try:
            X, keep_idx_local = _stack_equal_length_sequences(seq_sub)
        except Exception as e:
            if verbose:
                print(f"[skip trajectory PCA] sub={sub}, modality={modality}: {e}")
            continue

        if X is None or X.shape[0] < 2:
            if verbose:
                print(f"[skip trajectory PCA] sub={sub}, modality={modality}: too few valid windows")
            continue

        imp = SimpleImputer(strategy="mean")
        sc = StandardScaler()

        Xz = imp.fit_transform(X)
        Xz = sc.fit_transform(Xz)

        k = min(n_components, Xz.shape[0] - 1, Xz.shape[1])
        if k < 1:
            continue

        pca = PCA(n_components=k)
        Z = pca.fit_transform(Xz)

        results_by_sub[sub] = {
            "modality": modality,
            "pca": pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "Z": Z,
            "keep_idx_local": keep_idx_local,
            "window_df_kept": df_sub.iloc[keep_idx_local].reset_index(drop=True),
        }

        if verbose:
            print(
                f"[trajectory PCA] sub={sub}, modality={modality}, "
                f"X.shape={X.shape}, Z.shape={Z.shape}"
            )

    return results_by_sub

# =========================================================
# helpers
# =========================================================

def _ensure_2d_time_feat(x):
    """
    x: [T] or [T, D]
    return: [T, D]
    """
    if x is None:
        return None

    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]

    if x.ndim != 2:
        raise ValueError(f"x must be 1D or 2D, got shape={x.shape}")

    if x.size == 0 or np.isnan(x).all():
        return None

    return x


def _collect_equal_shape_windows(seq_list):
    """
    seq_list: list of arrays, each [T, D] or [T]

    Returns
    -------
    X : [N_keep, T, D]
    keep_idx_local : kept local indices
    ref_shape : (T, D)
    """
    keep_idx_local = []
    X_list = []
    ref_shape = None

    for i, x in enumerate(seq_list):
        x = _ensure_2d_time_feat(x)
        if x is None:
            continue

        if ref_shape is None:
            ref_shape = x.shape

        if x.shape != ref_shape:
            raise ValueError(
                f"Found unequal window shape: first shape={ref_shape}, current shape={x.shape}"
            )

        keep_idx_local.append(i)
        X_list.append(x)

    if len(X_list) == 0:
        return None, [], None

    X = np.asarray(X_list, dtype=float)   # [N, T, D]
    return X, keep_idx_local, ref_shape


def _index_map_from_keep_idx(keep_idx_local):
    """
    local original idx -> new compact idx
    """
    return {orig_i: new_i for new_i, orig_i in enumerate(keep_idx_local)}


def _safe_corr(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 3:
        return np.nan
    if np.nanstd(a[mask]) < 1e-10 or np.nanstd(b[mask]) < 1e-10:
        return np.nan

    return np.corrcoef(a[mask], b[mask])[0, 1]


def _remove_low_variance_cols(X, eps=1e-8):
    X = np.asarray(X, dtype=float)
    var = np.nanvar(X, axis=0)
    keep = var > eps
    return X[:, keep], keep


def _get_pair_key(cca_res, mod1, mod2):
    if (mod1, mod2) in cca_res:
        return (mod1, mod2), False
    elif (mod2, mod1) in cca_res:
        return (mod2, mod1), True
    else:
        raise ValueError(f"No CCA result for pair: {mod1}, {mod2}")


# =========================================================
# 1) subject-wise time-preserving PCA
# =========================================================

def run_subjectwise_time_trajectory_pca(
    concat_data,
    modality,
    n_components=5,
    sub_col="sub_id",
    standardize=True,
    verbose=True,
):
    """
    For each subject:
        X: [N, T, D]
        -> reshape to [N*T, D]
        -> PCA on feature dimension D
        -> reshape back to [N, T, K]

    Returns
    -------
    results_by_sub : dict
        {
            sub: {
                "modality": modality,
                "X": [N, T, D],
                "Z": [N, T, K],
                "pca": fitted PCA,
                "explained_variance_ratio": ...,
                "imputer": ...,
                "scaler": ... or None,
                "keep_idx_local": [...],
                "window_df_kept": ...,
                "shape": (N, T, D),
            }
        }
    """
    if "window_df" not in concat_data:
        raise ValueError("concat_data must contain 'window_df'")
    if "sequences" not in concat_data:
        raise ValueError("concat_data must contain 'sequences'")
    if modality not in concat_data["sequences"]:
        raise ValueError(f"{modality} not in concat_data['sequences']")

    df = concat_data["window_df"].copy().reset_index(drop=True)
    seq_list = concat_data["sequences"][modality]

    if len(df) != len(seq_list):
        raise ValueError(
            f"len(window_df)={len(df)} does not match "
            f"len(sequences[{modality}])={len(seq_list)}"
        )

    if sub_col not in df.columns:
        raise ValueError(f"`{sub_col}` not found in concat_data['window_df']")

    results_by_sub = {}

    for sub in sorted(df[sub_col].dropna().unique()):
        idx = np.where(df[sub_col].values == sub)[0]
        df_sub = df.iloc[idx].reset_index(drop=True)
        seq_sub = [seq_list[i] for i in idx]

        try:
            X, keep_idx_local, ref_shape = _collect_equal_shape_windows(seq_sub)
        except Exception as e:
            if verbose:
                print(f"[skip time-PCA] sub={sub}, modality={modality}: {e}")
            continue

        if X is None or X.shape[0] < 2:
            if verbose:
                print(f"[skip time-PCA] sub={sub}, modality={modality}: too few valid windows")
            continue

        N, T, D = X.shape
        X2 = X.reshape(N * T, D)

        imp = SimpleImputer(strategy="mean")
        X2z = imp.fit_transform(X2)

        sc = None
        if standardize:
            sc = StandardScaler()
            X2z = sc.fit_transform(X2z)

        k = min(n_components, X2z.shape[1], X2z.shape[0] - 1)
        if k < 1:
            if verbose:
                print(f"[skip time-PCA] sub={sub}, modality={modality}: k < 1")
            continue

        pca = PCA(n_components=k)
        Z2 = pca.fit_transform(X2z)   # [N*T, K]
        Z = Z2.reshape(N, T, k)

        results_by_sub[sub] = {
            "modality": modality,
            "X": X,
            "Z": Z,
            "pca": pca,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "imputer": imp,
            "scaler": sc,
            "keep_idx_local": keep_idx_local,
            "window_df_kept": df_sub.iloc[keep_idx_local].reset_index(drop=True),
            "shape": (N, T, D),
        }

        if verbose:
            print(
                f"[time-PCA] sub={sub}, modality={modality}, "
                f"X={X.shape}, Z={Z.shape}"
            )

    return results_by_sub


def run_multimodal_subjectwise_time_trajectory_pca(
    concat_data,
    modality_keys=None,
    n_components=5,
    sub_col="sub_id",
    standardize=True,
    verbose=True,
):
    """
    Returns
    -------
    out : dict
        {
            sub: {
                mod1: {...},
                mod2: {...},
                ...
            }
        }
    """
    if modality_keys is None:
        modality_keys = list(concat_data["sequences"].keys())

    out = {}

    for mod in modality_keys:
        res_mod = run_subjectwise_time_trajectory_pca(
            concat_data=concat_data,
            modality=mod,
            n_components=n_components,
            sub_col=sub_col,
            standardize=standardize,
            verbose=verbose,
        )

        for sub, sub_res in res_mod.items():
            if sub not in out:
                out[sub] = {}
            out[sub][mod] = sub_res

    return out


# =========================================================
# 2) subject-wise time-CCA on [N, T, K]
# =========================================================

def compute_pairwise_time_cca_subjectwise(
    subject_time_pca_res,
    modality_keys=None,
    n_components=3,
    min_windows=10,
    max_iter=2000,
    eps=1e-8,
    verbose=True,
):
    """
    For each subject and each modality pair:
        Xi: [N, T, Ki]
        Xj: [N, T, Kj]

    keep common windows by original local indices,
    flatten timepoints:
        [N, T, K] -> [N*T, K]
    do CCA on all timepoints pooled within subject,
    then reshape back:
        Xi_c: [N, T, Kc]
        Xj_c: [N, T, Kc]

    Returns
    -------
    cca_res[(m1, m2)] = {
        "mode": "subjectwise_time_cca",
        "per_subject": [
            {
                "sub": sub,
                "corrs": [Kc],
                "n_windows": N_common,
                "T": T,
                "Xi_c": [N, T, Kc],
                "Xj_c": [N, T, Kc],
                "common_orig_idx": [...],
                "cca_model": fitted CCA,
            },
            ...
        ],
        "mean_corrs": [Kc],
        "grand_mean_Xi": [T, K_common],
        "grand_mean_Xj": [T, K_common],
    }
    """
    all_mods = sorted(list(set(
        m for sub in subject_time_pca_res
        for m in subject_time_pca_res[sub].keys()
    )))
    if modality_keys is None:
        modality_keys = all_mods

    out = {}

    for m1, m2 in combinations(modality_keys, 2):
        per_subject = []

        for sub, sub_res in subject_time_pca_res.items():
            if m1 not in sub_res or m2 not in sub_res:
                continue

            ri = sub_res[m1]
            rj = sub_res[m2]

            Xi_full = ri["Z"]   # [Ni, T, Ki]
            Xj_full = rj["Z"]   # [Nj, T, Kj]

            map_i = _index_map_from_keep_idx(ri["keep_idx_local"])
            map_j = _index_map_from_keep_idx(rj["keep_idx_local"])

            common_orig_idx = sorted(set(map_i.keys()) & set(map_j.keys()))
            if len(common_orig_idx) < min_windows:
                if verbose:
                    print(f"[skip time-CCA] sub={sub}, {m1} vs {m2}: too few common windows")
                continue

            idx_i = [map_i[k] for k in common_orig_idx]
            idx_j = [map_j[k] for k in common_orig_idx]

            Xi = Xi_full[idx_i]   # [N, T, Ki]
            Xj = Xj_full[idx_j]   # [N, T, Kj]

            if Xi.shape[1] != Xj.shape[1]:
                if verbose:
                    print(f"[skip time-CCA] sub={sub}, {m1} vs {m2}: unequal T")
                continue

            N, T, Ki = Xi.shape
            _, _, Kj = Xj.shape

            Xi2 = Xi.reshape(N * T, Ki)
            Xj2 = Xj.reshape(N * T, Kj)

            Xi2, keep_i = _remove_low_variance_cols(Xi2, eps=eps)
            Xj2, keep_j = _remove_low_variance_cols(Xj2, eps=eps)

            if Xi2.shape[1] == 0 or Xj2.shape[1] == 0:
                if verbose:
                    print(f"[skip time-CCA] sub={sub}, {m1} vs {m2}: no usable cols")
                continue

            rank_i = np.linalg.matrix_rank(Xi2)
            rank_j = np.linalg.matrix_rank(Xj2)

            k = min(
                n_components,
                Xi2.shape[1],
                Xj2.shape[1],
                Xi2.shape[0] - 1,
                rank_i,
                rank_j,
            )
            if k < 1:
                if verbose:
                    print(f"[skip time-CCA] sub={sub}, {m1} vs {m2}: k < 1")
                continue

            cca = CCA(n_components=k, max_iter=max_iter)
            Xi_c2, Xj_c2 = cca.fit_transform(Xi2, Xj2)

            good = []
            corrs = []
            for d in range(k):
                c = _safe_corr(Xi_c2[:, d], Xj_c2[:, d])
                if np.isfinite(c):
                    good.append(d)
                    corrs.append(c)

            if len(good) == 0:
                if verbose:
                    print(f"[skip time-CCA] sub={sub}, {m1} vs {m2}: all components collapsed")
                continue

            Xi_c2 = Xi_c2[:, good]
            Xj_c2 = Xj_c2[:, good]

            Kc = Xi_c2.shape[1]
            Xi_c = Xi_c2.reshape(N, T, Kc)
            Xj_c = Xj_c2.reshape(N, T, Kc)

            per_subject.append({
                "sub": sub,
                "corrs": corrs,
                "n_windows": N,
                "T": T,
                "Xi_c": Xi_c,
                "Xj_c": Xj_c,
                "common_orig_idx": common_orig_idx,
                "cca_model": cca,
            })

            if verbose:
                print(
                    f"[time-CCA] sub={sub}, {m1} vs {m2}, "
                    f"N={N}, T={T}, Kc={Kc}, corrs="
                    + ", ".join(f"{c:.3f}" for c in corrs)
                )

        if len(per_subject) == 0:
            continue

        max_k = max(len(r["corrs"]) for r in per_subject)
        corr_mat = np.full((len(per_subject), max_k), np.nan)
        for i, r in enumerate(per_subject):
            corr_mat[i, :len(r["corrs"])] = r["corrs"]

        common_k = min(r["Xi_c"].shape[2] for r in per_subject)
        common_T = min(r["Xi_c"].shape[1] for r in per_subject)

        Xi_mean_by_sub = []
        Xj_mean_by_sub = []

        for r in per_subject:
            Xi_mean = r["Xi_c"][:, :common_T, :common_k].mean(axis=0)  # [T, K]
            Xj_mean = r["Xj_c"][:, :common_T, :common_k].mean(axis=0)
            Xi_mean_by_sub.append(Xi_mean)
            Xj_mean_by_sub.append(Xj_mean)

        Xi_mean_by_sub = np.asarray(Xi_mean_by_sub, dtype=float)  # [S, T, K]
        Xj_mean_by_sub = np.asarray(Xj_mean_by_sub, dtype=float)

        out[(m1, m2)] = {
            "mode": "subjectwise_time_cca",
            "per_subject": per_subject,
            "mean_corrs": np.nanmean(corr_mat, axis=0).tolist(),
            "grand_mean_Xi": np.nanmean(Xi_mean_by_sub, axis=0),   # [T, K]
            "grand_mean_Xj": np.nanmean(Xj_mean_by_sub, axis=0),   # [T, K]
            "Xi_mean_by_sub": Xi_mean_by_sub,                      # [S, T, K]
            "Xj_mean_by_sub": Xj_mean_by_sub,                      # [S, T, K]
        }

    return out


# =========================================================
# 3) matrix / synergy view for time-CCA
# =========================================================

def time_cca_results_to_matrix(time_cca_res, modality_keys=None, mode="first"):
    """
    mode:
        "first" -> first canonical corr
        "mean"  -> mean canonical corr
    """
    if modality_keys is None:
        mods = sorted(list(set(
            [k[0] for k in time_cca_res.keys()] + [k[1] for k in time_cca_res.keys()]
        )))
    else:
        mods = list(modality_keys)

    M = np.full((len(mods), len(mods)), np.nan, dtype=float)
    for i in range(len(mods)):
        M[i, i] = 1.0

    for (m1, m2), res in time_cca_res.items():
        corrs = np.asarray(res["mean_corrs"], dtype=float)

        if corrs.size == 0 or np.all(np.isnan(corrs)):
            val = np.nan
        else:
            if mode == "first":
                val = corrs[0]
            elif mode == "mean":
                val = np.nanmean(corrs)
            else:
                raise ValueError("mode must be 'first' or 'mean'")

        i = mods.index(m1)
        j = mods.index(m2)
        M[i, j] = val
        M[j, i] = val

    return mods, M


def plot_time_cca_synergy_matrix(
    time_cca_res,
    modality_keys=None,
    mode="first",
    title="Time-CCA synergy matrix",
    vmin=0,
    vmax=1,
):
    mods, M = time_cca_results_to_matrix(
        time_cca_res,
        modality_keys=modality_keys,
        mode=mode,
    )

    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Canonical correlation")
    plt.xticks(range(len(mods)), mods, rotation=45, ha="right")
    plt.yticks(range(len(mods)), mods)
    plt.title(title)

    for i in range(len(mods)):
        for j in range(len(mods)):
            if np.isfinite(M[i, j]):
                plt.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()


# =========================================================
# 4) grand-average time-series plotter
# =========================================================

def plot_time_cca_grandavg_timeseries(
    time_cca_res,
    mod1,
    mod2,
    comp=0,
    with_sem=True,
    title=None,
    figsize=(7, 4),
):
    """
    Plot grand-average canonical time series for one modality pair.

    Uses subject-level mean-over-window first, then averages across subjects.
    """
    key, reversed_order = _get_pair_key(time_cca_res, mod1, mod2)
    res = time_cca_res[key]

    Xi = res["Xi_mean_by_sub"]   # [S, T, K]
    Xj = res["Xj_mean_by_sub"]   # [S, T, K]

    if reversed_order:
        Xi, Xj = Xj, Xi

    if comp >= Xi.shape[2]:
        raise ValueError(f"comp={comp} out of range, only {Xi.shape[2]} components")

    T = Xi.shape[1]
    t = np.arange(T)

    mi = Xi[:, :, comp].mean(axis=0)
    mj = Xj[:, :, comp].mean(axis=0)

    if title is None:
        title = f"Grand-average time-CCA: {mod1} vs {mod2} (comp {comp+1})"

    plt.figure(figsize=figsize)
    plt.plot(t, mi, label=mod1)
    plt.plot(t, mj, label=mod2)

    if with_sem and Xi.shape[0] > 1:
        sei = Xi[:, :, comp].std(axis=0, ddof=1) / np.sqrt(Xi.shape[0])
        sej = Xj[:, :, comp].std(axis=0, ddof=1) / np.sqrt(Xj.shape[0])

        plt.fill_between(t, mi - sei, mi + sei, alpha=0.2)
        plt.fill_between(t, mj - sej, mj + sej, alpha=0.2)

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("time step within window")
    plt.ylabel("canonical variate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 5) optional heatmap plotter
# =========================================================

def plot_time_cca_mean_timeseries_heatmap(
    time_cca_res,
    mod1,
    mod2,
    use_abs=False,
    title=None,
):
    """
    Heatmap of grand mean [T, K] for both modalities.
    """
    key, reversed_order = _get_pair_key(time_cca_res, mod1, mod2)
    res = time_cca_res[key]

    Xi = res["grand_mean_Xi"].T  # [T, K]
    Xj = res["grand_mean_Xj"].T   # [T, K]

    if reversed_order:
        Xi, Xj = Xj, Xi

    if use_abs:
        Xi = np.abs(Xi)
        Xj = np.abs(Xj)

    if title is None:
        title = f"Mean time-CCA latent heatmap: {mod1} vs {mod2}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)

    im0 = axes[0, 0].imshow(Xi, aspect="auto", origin="lower")
    axes[0, 0].set_title(mod1)
    axes[0, 0].set_xlabel("time step")
    axes[0, 0].set_ylabel("CCA component")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(Xj, aspect="auto", origin="lower")
    axes[0, 1].set_title(mod2)
    axes[0, 1].set_xlabel("time step")
    axes[0, 1].set_ylabel("CCA component")
    plt.colorbar(im1, ax=axes[0, 1])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_time_cca_latent_spectrogram(
    time_cca_res,
    mod1,
    mod2,
    feature_mode="first",   # "first" | "mean"
    which="both",           # "mod1" | "mod2" | "both"
    fs=1.0,                 # time axis sampling rate
    nperseg=None,
    noverlap=None,
    detrend="constant",
    scaling="density",
    mode="psd",             # "psd" | "magnitude"
    log_power=True,
    title=None,
):
    """
    对 time-CCA 的 grand mean latent 时序做时频分解并画图。

    Parameters
    ----------
    time_cca_res : dict
        time-CCA result dict

    mod1, mod2 : str
        modality names

    feature_mode : str
        "first" -> 用第1个 CCA component 的时间序列
        "mean"  -> 对所有 CCA component 在 axis=1 上取均值，得到单变量时间序列

    which : str
        "mod1" | "mod2" | "both"

    fs : float
        时间采样率。如果每个 time step 对应 1 个采样点，就设成 1；
        如果你知道真实 Hz，比如 window 内是 25 Hz，就传 25。

    nperseg, noverlap : int or None
        spectrogram 参数

    mode : str
        "psd" 或 "magnitude"

    log_power : bool
        是否画 log(1 + Sxx)

    Returns
    -------
    out : dict
        包含频率、时间、谱图结果
    """
    key, reversed_order = _get_pair_key(time_cca_res, mod1, mod2)
    res = time_cca_res[key]

    Xi = np.asarray(res["grand_mean_Xi"], dtype=float)   # [T, K]
    Xj = np.asarray(res["grand_mean_Xj"], dtype=float)   # [T, K]

    if reversed_order:
        Xi, Xj = Xj, Xi

    def _reduce_latent(X, feature_mode="first"):
        if X.ndim != 2:
            raise ValueError(f"Expected [T, K], got shape={X.shape}")

        if feature_mode == "first":
            return X[:, 0]
        elif feature_mode == "mean":
            return np.nanmean(X, axis=1)
        else:
            raise ValueError("feature_mode must be 'first' or 'mean'")

    xi_1d = _reduce_latent(Xi, feature_mode=feature_mode)
    xj_1d = _reduce_latent(Xj, feature_mode=feature_mode)

    if nperseg is None:
        nperseg = min(64, len(xi_1d))
    if noverlap is None:
        noverlap = nperseg // 2

    def _spec(x):
        f, t, Sxx = spectrogram(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=detrend,
            scaling=scaling,
            mode=mode,
        )
        if log_power:
            Sxx = np.log1p(Sxx)
        return f, t, Sxx

    out = {}

    if which in ["mod1", "both"]:
        f1, t1, S1 = _spec(xi_1d)
        out[mod1] = {
            "signal": xi_1d,
            "freqs": f1,
            "times": t1,
            "spec": S1,
        }

    if which in ["mod2", "both"]:
        f2, t2, S2 = _spec(xj_1d)
        out[mod2] = {
            "signal": xj_1d,
            "freqs": f2,
            "times": t2,
            "spec": S2,
        }

    if which == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)

        pcm0 = axes[0, 0].pcolormesh(
            out[mod1]["times"],
            out[mod1]["freqs"],
            out[mod1]["spec"],
            shading="auto",
        )
        axes[0, 0].set_title(f"{mod1} ({feature_mode})")
        axes[0, 0].set_xlabel("time (s)")
        axes[0, 0].set_ylabel("frequency (Hz)")
        plt.colorbar(pcm0, ax=axes[0, 0])

        pcm1 = axes[0, 1].pcolormesh(
            out[mod2]["times"],
            out[mod2]["freqs"],
            out[mod2]["spec"],
            shading="auto",
        )
        axes[0, 1].set_title(f"{mod2} ({feature_mode})")
        axes[0, 1].set_xlabel("time (s)")
        axes[0, 1].set_ylabel("frequency (Hz)")
        plt.colorbar(pcm1, ax=axes[0, 1])

        if title is None:
            title = f"Time-frequency of time-CCA latent ({feature_mode}): {mod1} vs {mod2}"
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    else:
        mod = mod1 if which == "mod1" else mod2
        fig, ax = plt.subplots(figsize=(6, 4))

        pcm = ax.pcolormesh(
            out[mod]["times"],     # x: time (s)
            out[mod]["freqs"],     # y: frequency (Hz)
            out[mod]["spec"],      # [F, T]
            shading="auto",
        )

        ax.set_title(f"{mod} ({feature_mode})")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("frequency (Hz)")
        plt.colorbar(pcm, ax=ax)

        T = len(out[mod]["signal"])
        duration = T / fs
        ax.set_xlim(0, duration)

        if title is None:
            title = f"Time-frequency of time-CCA latent ({feature_mode}): {mod}"
        plt.suptitle(title)

        plt.tight_layout()
        plt.show()

    return out