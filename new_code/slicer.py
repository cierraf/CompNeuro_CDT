import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def get_movie_fps_nominal(sub_data):
    meta = sub_data.get("meta", {})
    if isinstance(meta, dict) and "movie_fps_nominal" in meta:
        return float(meta["movie_fps_nominal"])

    t_grid = np.asarray(sub_data.get("time_grid", []), dtype=float)
    if len(t_grid) >= 2:
        dt = np.median(np.diff(t_grid))
        if dt > 0 and np.isfinite(dt):
            return float(1.0 / dt)

    raise KeyError(
        "movie_fps_nominal not found in sub_data['meta'] and could not infer fps from time_grid"
    )

def slice_modalities_by_index(sub_data, idx, bands=None):
    t = np.asarray(sub_data["time_grid"], dtype=float)
    idx = np.asarray(idx, dtype=int)

    out_seg = {
        "frame_grid": t[idx],
        "lfp_macro": sub_data["lfp_macro"][idx] if sub_data.get("lfp_macro") is not None else None,
        "lfp_micro": sub_data["lfp_micro"][idx] if sub_data.get("lfp_micro") is not None else None,
        "eye_gaze": sub_data["eye_gaze"][idx] if sub_data.get("eye_gaze") is not None else None,
        "pupil": sub_data["pupil"][idx] if sub_data.get("pupil") is not None else None,
        "firing_rate": sub_data["firing_rate"][idx] if sub_data.get("firing_rate") is not None else None,
    }

    if "lfp_bandpower" in sub_data and sub_data["lfp_bandpower"] is not None:
        if bands is None:
            out_seg["lfp_bandpower"] = {
                k: v[idx] for k, v in sub_data["lfp_bandpower"].items()
            }
        else:
            out_seg["lfp_bandpower"] = {
                k: sub_data["lfp_bandpower"][k][idx] for k in bands
            }

    return out_seg

def get_grid_index_from_movie_frame(sub_data, start_frame, end_frame):
    fps = get_movie_fps_nominal(sub_data)
    t_grid = np.asarray(sub_data["time_grid"], dtype=float)

    start_sec = start_frame / fps
    end_sec = end_frame / fps

    idx = np.where((t_grid >= start_sec) & (t_grid <= end_sec))[0]
    return idx

def slice_modalities_by_movie_frame(sub_data, start_frame, end_frame, bands=None):
    idx = get_grid_index_from_movie_frame(sub_data, start_frame, end_frame)
    return slice_modalities_by_index(sub_data, idx, bands=bands)

def slice_modalities_by_movie_time_sec(sub_data, start_sec, end_sec):
    t_grid = np.asarray(sub_data["time_grid"], dtype=float)
    idx = np.where((t_grid >= start_sec) & (t_grid <= end_sec))[0]
    return slice_modalities_by_index(sub_data, idx)

def resample_face_df_to_timegrid(df_face, t_grid, feature_cols, method_map=None):
    if method_map is None:
        method_map = {}

    df_face = df_face.sort_values("movie_time").reset_index(drop=True)
    t_src = df_face["movie_time"].to_numpy(dtype=float)
    t_tgt = np.asarray(t_grid, dtype=float)

    out_dict = {"time": t_tgt}

    for col in feature_cols:
        x = df_face[col].to_numpy()

        if x.dtype == bool:
            x = x.astype(float)

        method = method_map.get(col, "linear")
        valid = np.isfinite(t_src) & np.isfinite(x.astype(float))

        if valid.sum() < 2:
            out_dict[col] = np.full(len(t_tgt), np.nan, dtype=float)
            continue

        ts = t_src[valid]
        xs = x[valid].astype(float)

        uniq_t, uniq_idx = np.unique(ts, return_index=True)
        xs = xs[uniq_idx]

        if len(uniq_t) == 1:
            out = np.full(len(t_tgt), xs[0], dtype=float)

        elif method == "nearest":
            idx = np.searchsorted(uniq_t, t_tgt, side="left")
            idx = np.clip(idx, 0, len(uniq_t) - 1)

            left_idx = np.clip(idx - 1, 0, len(uniq_t) - 1)
            right_idx = idx

            left_dist = np.abs(t_tgt - uniq_t[left_idx])
            right_dist = np.abs(t_tgt - uniq_t[right_idx])

            choose_left = left_dist <= right_dist
            nn_idx = np.where(choose_left, left_idx, right_idx)
            out = xs[nn_idx]
        else:
            out = np.interp(t_tgt, uniq_t, xs)

        out_dict[col] = out

    return pd.DataFrame(out_dict)