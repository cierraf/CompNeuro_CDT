from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

# fMRI (optional)
from bids import BIDSLayout
import nibabel as nib
import traceback

# signal processing
from scipy.signal import butter, sosfiltfilt, hilbert
from collections.abc import Iterable
from scipy.signal import butter, filtfilt


# -----------------------
# filter function
# -----------------------
def _apply_one_filter(x, fs, filter_spec):
    """
    x: [T] or [T, D]
    fs: sampling rate
    filter_spec: dict
        examples:
        {"type": "bandpass", "low": 0.5, "high": 150.0, "order": 4}
        {"type": "lowpass", "high": 20.0, "order": 4}
        {"type": "highpass", "low": 0.5, "order": 4}
        {"type": "bandstop", "low": 58.0, "high": 62.0, "order": 4}
    """
    if x is None or fs is None:
        return x

    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x_in = x[:, None]
        squeeze_back = True
    else:
        x_in = x
        squeeze_back = False

    nyq = fs / 2.0
    ftype = filter_spec.get("type", "bandpass")
    order = filter_spec.get("order", 4)

    if ftype == "bandpass":
        low = filter_spec["low"] / nyq
        high = filter_spec["high"] / nyq
        b, a = butter(order, [low, high], btype="bandpass")

    elif ftype == "lowpass":
        high = filter_spec["high"] / nyq
        b, a = butter(order, high, btype="lowpass")

    elif ftype == "highpass":
        low = filter_spec["low"] / nyq
        b, a = butter(order, low, btype="highpass")

    elif ftype == "bandstop":
        low = filter_spec["low"] / nyq
        high = filter_spec["high"] / nyq
        b, a = butter(order, [low, high], btype="bandstop")

    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    x_out = np.empty_like(x_in)
    for d in range(x_in.shape[1]):
        col = x_in[:, d]
        if np.isnan(col).all():
            x_out[:, d] = col
            continue

        valid_mean = np.nanmean(col)
        col_fill = np.where(np.isnan(col), valid_mean, col)
        x_out[:, d] = filtfilt(b, a, col_fill)

    if squeeze_back:
        x_out = x_out[:, 0]

    return x_out

def apply_signal_filter(x, fs, filter_spec):
    """
    支持:
    1) 单个 dict
    2) list[dict]，按顺序依次应用
    """
    if x is None or fs is None or filter_spec is None:
        return x

    if isinstance(filter_spec, dict):
        return _apply_one_filter(x, fs, filter_spec)

    if isinstance(filter_spec, (list, tuple)):
        y = x
        for spec in filter_spec:
            y = _apply_one_filter(y, fs, spec)
        return y

    raise TypeError("filter_spec must be a dict or a list/tuple of dicts")

# -----------------------
# Subject mapping (int -> ids)
# -----------------------
def bids_subject_from_int(sub: int) -> str:
    return f"p{sub}cs"   # BIDS folder: sub-p41cs


def nwb_subject_from_int(sub: int) -> str:
    return f"CS{sub}"    # NWB subject: sub-CS41


# -----------------------
# Path finding
# -----------------------
def find_nwb_file_for_subject(nwb_root: Union[str, Path], nwb_sub: str) -> Path:
    nwb_root = Path(nwb_root)
    if not nwb_root.exists():
        raise FileNotFoundError(f"NWB root not found: {nwb_root}")

    patterns = [
        f"**/sub-{nwb_sub}/*.nwb",
        f"**/sub-{nwb_sub}*.nwb",
        f"**/*sub-{nwb_sub}*.nwb",
    ]
    hits: List[Path] = []
    for pat in patterns:
        hits.extend(nwb_root.glob(pat))
    hits = sorted(set(hits))

    if not hits:
        all_nwb = list(nwb_root.rglob("*.nwb"))
        hits = sorted([p for p in all_nwb if f"sub-{nwb_sub}" in str(p)])

    if not hits:
        raise FileNotFoundError(f"No NWB found for subject {nwb_sub} under {nwb_root}")

    return hits[0]


# -----------------------
# Time utilities
# -----------------------
def times_from_rate(T: int, rate: float, starting_time: float = 0.0) -> np.ndarray:
    return starting_time + np.arange(T, dtype=np.float64) / float(rate)


def get_ts_data_and_time(ts, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """
    Return (data, t, rate) for an NWB TimeSeries-like object.
    Tries timestamps first, otherwise uses (starting_time, rate).
    """
    if max_samples is None:
        data = np.asarray(ts.data[:])
    else:
        data = np.asarray(ts.data[:max_samples])

    rate = getattr(ts, "rate", None)
    starting_time = getattr(ts, "starting_time", 0.0)

    if getattr(ts, "timestamps", None) is not None:
        if max_samples is None:
            t = np.asarray(ts.timestamps[:], dtype=np.float64)
        else:
            t = np.asarray(ts.timestamps[:max_samples], dtype=np.float64)
    else:
        if rate is None:
            raise ValueError("TimeSeries has no timestamps and no rate; cannot build time axis.")
        t = times_from_rate(len(data), rate, float(starting_time))

    return data, t, rate


def resample_continuous(x: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """
    x: (T,) or (T,C). Linear interpolation onto t_dst.
    """
    x = np.asarray(x)
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)

    if x.ndim == 1:
        return np.interp(t_dst, t_src, x).astype(np.float32)
    elif x.ndim == 2:
        out = np.empty((len(t_dst), x.shape[1]), dtype=np.float32)
        for c in range(x.shape[1]):
            out[:, c] = np.interp(t_dst, t_src, x[:, c])
        return out
    else:
        raise ValueError(f"resample_continuous expects 1D or 2D, got {x.ndim}D")


def _pick_first_series(container, prefer: Optional[List[str]] = None) -> str:
    """
    container: a MultiContainerInterface-like object with .keys() and __getitem__ expecting str
    prefer: optional list of preferred key substrings (case-insensitive)
    Returns: a key string that works with container[key]
    """
    keys_raw = list(container.keys())
    keys_str = [k if isinstance(k, str) else str(k) for k in keys_raw]

    if prefer:
        prefer_l = [p.lower() for p in prefer]
        keys_str = sorted(
            keys_str,
            key=lambda s: 0 if any(p in s.lower() for p in prefer_l) else 1
        )

    last_err = None
    for k in keys_str:
        try:
            _ = container[k]
            return k
        except Exception as e:
            last_err = e
            continue

    raise TypeError(
        f"Could not index container with any stringified key. "
        f"raw keys={keys_raw[:10]}... last_err={repr(last_err)}"
    )


# -----------------------
# Spikes -> firing rate
# -----------------------
def spikes_to_firing_rate(
    spike_times_list: List[np.ndarray],
    t_grid: np.ndarray,
    bin_width: float,
) -> np.ndarray:
    """
    Bin spikes into firing rate on t_grid (sec).

    We interpret t_grid as bin centers. The bin width is explicitly controlled by
    `bin_width` instead of always using median(diff(t_grid)).

    Output shape: (T, n_units) in Hz.
    """
    T = len(t_grid)
    n_units = len(spike_times_list)
    if T < 1:
        raise ValueError("t_grid must have at least 1 point.")
    if bin_width is None or bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    t_grid = np.asarray(t_grid, dtype=np.float64)
    half_bw = float(bin_width) / 2.0
    edges = np.concatenate(([t_grid[0] - half_bw], t_grid + half_bw))

    fr = np.zeros((T, n_units), dtype=np.float32)
    for i, st in enumerate(spike_times_list):
        if st is None or len(st) == 0:
            continue
        st = np.asarray(st, dtype=np.float64)
        counts, _ = np.histogram(st, bins=edges)
        fr[:, i] = counts.astype(np.float32) / float(bin_width)
    return fr


def convert_lfp_time_to_movie_reference(t_raw, baseline_pre_movie=10.0, atol=1e-3):
    t_raw = np.asarray(t_raw, dtype=np.float64)

    if len(t_raw) == 0:
        return t_raw

    if np.isclose(t_raw[0], -baseline_pre_movie, atol=atol):
        return t_raw

    if np.isclose(t_raw[0], 0.0, atol=atol):
        return t_raw - baseline_pre_movie

    print(
        f"[WARN] Unexpected LFP time start {t_raw[0]:.6f}; "
        f"leaving unchanged. Please verify ElectricalSeries.starting_time."
    )
    return t_raw


# -----------------------
# LFP band power
# -----------------------
def bandpower_hilbert(
    lfp: np.ndarray,
    fs: float,
    band: Tuple[float, float],
) -> np.ndarray:
    """
    lfp: (T,C) or (T,)
    Returns power envelope (T,C) or (T,) using bandpass + Hilbert magnitude^2.
    """
    lo, hi = band
    if fs is None:
        raise ValueError("fs required for bandpower.")

    sos = butter(4, [lo, hi], btype="bandpass", fs=float(fs), output="sos")
    if lfp.ndim == 1:
        y = sosfiltfilt(sos, lfp)
        env = np.abs(hilbert(y)) ** 2
        return env.astype(np.float32)
    elif lfp.ndim == 2:
        out = np.empty_like(lfp, dtype=np.float32)
        for c in range(lfp.shape[1]):
            y = sosfiltfilt(sos, lfp[:, c])
            out[:, c] = (np.abs(hilbert(y)) ** 2).astype(np.float32)
        return out
    else:
        raise ValueError("lfp must be 1D or 2D")


# -----------------------
# NWB modality loaders
# -----------------------
def load_lfp_container(nwbfile, which: str, max_samples: Optional[int] = None):
    ece = nwbfile.processing["ecephys"]
    if hasattr(ece, "data_interfaces") and which in ece.data_interfaces:
        lfp_interface = ece.data_interfaces[which]
    else:
        lfp_interface = ece[which]

    series_key = _pick_first_series(
        lfp_interface.electrical_series,
        prefer=["ElectricalSeries", "LFP"]
    )
    ts = lfp_interface.electrical_series[series_key]

    lfp, t, fs = get_ts_data_and_time(ts, max_samples=max_samples)
    lfp = np.asarray(lfp)
    if lfp.ndim == 1:
        lfp = lfp[:, None]
    return lfp.astype(np.float32), np.asarray(t, dtype=np.float64), fs


def load_eye_gaze_and_pupil(nwbfile, max_samples: Optional[int] = None):
    beh = nwbfile.processing["behavior"]

    gaze = None
    pupil = None
    t_eye = None

    beh_keys_raw = list(beh.keys())
    beh_keys_str = {k if isinstance(k, str) else str(k) for k in beh_keys_raw}

    if "EyeTracking" in beh_keys_str:
        et = beh["EyeTracking"]
        key = _pick_first_series(et.spatial_series, prefer=["SpatialSeries", "gaze"])
        ts = et.spatial_series[key]
        data, t, _ = get_ts_data_and_time(ts, max_samples=max_samples)
        gaze = np.asarray(data).astype(np.float32)
        t_eye = np.asarray(t, dtype=np.float64)

    if "PupilTracking" in beh_keys_str:
        pt = beh["PupilTracking"]
        key = _pick_first_series(pt.time_series, prefer=["TimeSeries", "pupil"])
        ts = pt.time_series[key]
        data, t, _ = get_ts_data_and_time(ts, max_samples=max_samples)
        pupil = np.asarray(data).squeeze().astype(np.float32)
        if t_eye is None:
            t_eye = np.asarray(t, dtype=np.float64)

    return gaze, pupil, t_eye


def load_spikes_units(nwbfile) -> List[np.ndarray]:
    """
    Return list of spike_times arrays (seconds), one per unit.
    """
    if getattr(nwbfile, "units", None) is None:
        return []
    df = nwbfile.units.to_dataframe()
    if "spike_times" not in df.columns:
        return []
    spikes = []
    for st in df["spike_times"].values:
        spikes.append(np.asarray(st, dtype=np.float64))
    return spikes


def load_movie_time(nwbfile, max_samples: Optional[int] = None) -> np.ndarray:
    """
    Use stimulus['movieframe_time'] as movie frame index.
    """
    stim = nwbfile.stimulus
    if "movieframe_time" not in stim:
        raise KeyError("movieframe_time not found in nwbfile.stimulus")
    ts = stim["movieframe_time"]
    data, _, _ = get_ts_data_and_time(ts, max_samples=max_samples)
    data = np.asarray(data).squeeze().astype(np.float64)
    return data


# -----------------------
# NaN filling
# -----------------------
def _to_2d_array(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return x


def _restore_shape(x_filled, original):
    original = np.asarray(original)
    if original.ndim == 1:
        return x_filled[:, 0]
    return x_filled


def _interp_fill(x):
    x2 = _to_2d_array(x)
    df = pd.DataFrame(x2)
    df = df.interpolate(method="linear", limit_direction="both")
    df = df.ffill().bfill()
    return _restore_shape(df.values, x)


def _ffill_bfill(x):
    x2 = _to_2d_array(x)
    df = pd.DataFrame(x2)
    df = df.ffill().bfill()
    return _restore_shape(df.values, x)


def _zero_fill(x):
    x = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0)


def _fill_gaze_pixel(x):
    return _interp_fill(x)

def _drop_all_nan_features(x, verbose=False, name=""):
    """
    x: [T, D] or [T]
    return:
        x_new: 去掉全 NaN 列后的数据
        keep_mask: 保留的列 mask
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        # 1D 情况：要么全 NaN，要么不用 drop
        if np.isnan(x).all():
            if verbose:
                print(f"[drop] {name}: all NaN (1D), return zeros")
            return np.zeros_like(x), np.array([False])
        return x, np.array([True])

    # 2D: [T, D]
    keep = ~np.isnan(x).all(axis=0)

    if verbose:
        n_drop = (~keep).sum()
        if n_drop > 0:
            print(f"[drop] {name}: drop {n_drop}/{x.shape[1]} all-NaN features")

    if keep.sum() == 0:
        if verbose:
            print(f"[WARN] {name}: all features are NaN → return zeros")
        return np.zeros_like(x[:, :1]), keep

    return x[:, keep], keep


def fill_nan_by_modality(sub_dict, fill_bold: bool = False, verbose: bool = False):
    """
    改进版：
    1. 填 NaN
    2. 自动 drop 全 NaN feature（关键改动）
    """
    d = sub_dict.copy()

    # -----------------------
    # spikes
    # -----------------------
    if "spikes" in d and d["spikes"] is not None:
        if isinstance(d["spikes"], list):
            filled_spikes = []
            for s in d["spikes"]:
                if s is None:
                    filled_spikes.append(s)
                else:
                    filled_spikes.append(_zero_fill(s))
            d["spikes"] = filled_spikes
        else:
            d["spikes"] = _zero_fill(d["spikes"])

    # -----------------------
    # firing_rate
    # -----------------------
    if "firing_rate" in d and d["firing_rate"] is not None:
        X = _zero_fill(d["firing_rate"])
        X, _ = _drop_all_nan_features(X, verbose, "firing_rate")
        d["firing_rate"] = X

    # -----------------------
    # LFP（关键问题源）
    # -----------------------
    for k in ["lfp_macro", "lfp_micro"]:
        if k in d and d[k] is not None:
            X = _interp_fill(d[k])
            X, _ = _drop_all_nan_features(X, verbose, k)
            d[k] = X

    # -----------------------
    # bandpower
    # -----------------------
    if "lfp_bandpower" in d and d["lfp_bandpower"] is not None:
        if isinstance(d["lfp_bandpower"], dict):
            filled_bp = {}
            for band_name, bp in d["lfp_bandpower"].items():
                if bp is None:
                    filled_bp[band_name] = None
                else:
                    X = _interp_fill(bp)
                    X, _ = _drop_all_nan_features(
                        X, verbose, f"lfp_bandpower[{band_name}]"
                    )
                    filled_bp[band_name] = X
            d["lfp_bandpower"] = filled_bp
        else:
            X = _interp_fill(d["lfp_bandpower"])
            X, _ = _drop_all_nan_features(X, verbose, "lfp_bandpower")
            d["lfp_bandpower"] = X

    # -----------------------
    # eye_gaze
    # -----------------------
    if "eye_gaze" in d and d["eye_gaze"] is not None:
        X = _fill_gaze_pixel(d["eye_gaze"])
        X, _ = _drop_all_nan_features(X, verbose, "eye_gaze")
        d["eye_gaze"] = X

    # -----------------------
    # pupil
    # -----------------------
    if "pupil" in d and d["pupil"] is not None:
        X = _ffill_bfill(d["pupil"])
        X, _ = _drop_all_nan_features(X, verbose, "pupil")
        d["pupil"] = X

    # -----------------------
    # bold
    # -----------------------
    if fill_bold and "bold" in d and d["bold"] is not None:
        if isinstance(d["bold"], list):
            filled_bold = []
            for b in d["bold"]:
                if b is None:
                    filled_bold.append(None)
                else:
                    X = _interp_fill(b)
                    X, _ = _drop_all_nan_features(X, verbose, "bold")
                    filled_bold.append(X)
            d["bold"] = filled_bold
        else:
            X = _interp_fill(d["bold"])
            X, _ = _drop_all_nan_features(X, verbose, "bold")
            d["bold"] = X

    return d

def remove_dead_channels(sub_dict, key):
    X = sub_dict[key]
    if X is None:
        return sub_dict

    X = np.asarray(X)
    keep = ~np.isnan(X).all(axis=0)

    sub_dict[key] = X[:, keep]
    return sub_dict


# -----------------------
# fMRI optional extraction
# -----------------------
def load_bold_timeseries_wholebrain(
    bids_root: Union[str, Path],
    bids_sub: str,
    max_runs: Optional[int] = None
) -> List[np.ndarray]:
    """
    Returns list of arrays (T,V). Uses nilearn NiftiMasker.
    """
    from nilearn.maskers import NiftiMasker

    layout = BIDSLayout(str(bids_root), validate=False)
    bold_files: List[str] = layout.get(
        subject=bids_sub,
        datatype="func",
        suffix="bold",
        extension=["nii", "nii.gz"],
        return_type="file"
    )
    if max_runs is not None:
        bold_files = bold_files[:max_runs]

    masker = NiftiMasker(standardize=False)
    out = []
    for f in bold_files:
        img = nib.load(f)
        X = masker.fit_transform(img)
        out.append(X.astype(np.float32))
    return out


# -----------------------
# Main multimodal loader
# -----------------------
def load_multimodal_subjects_movie_aligned(
    sub_nums: Union[Iterable[int], set],
    nwb_root: Union[str, Path],
    bids_root: Union[str, Path],
    *,
    max_nwb_samples: Optional[int] = None,
    b_filter_data: Optional[bool] = True,
    filter_config=None,

    # analysis grid
    grid_source: str = "movie_frame_time",   # {"movie_frame_time", "uniform", "lfp_raw"}
    grid_dt: Optional[float] = None,
    uniform_grid_include_baseline: bool = True,

    # feature computation
    compute_firing_rate: bool = True,
    firing_rate_bin_width: Optional[float] = None,
    compute_lfp_bandpower: bool = True,
    lfp_bandpower_on_raw: bool = True,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,

    # optional fmri
    load_fmri: bool = False,
    fmri_max_runs: Optional[int] = None,

    verbose: bool = True,
    **kwargs,
) -> Dict[int, Dict[str, Any]]:
    """
    Paper-consistent movie-aligned loader.

    Time convention:
        movie start = 0
        LFP/iEEG baseline before movie = negative time (typically [-10, 0))
        movieframe_time in NWB is treated as frame index
        movieframe_time_sec = movieframe_time / movie_fps_nominal
        spikes are already referenced to movie start per paper
    """
    if kwargs and verbose:
        print(f"[WARN] ignored kwargs: {sorted(kwargs.keys())}")

    if bands is None:
        bands = {
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (13, 30),
            "gamma": (30, 80),
            "high_gamma": (80, 150),
        }

    out: Dict[int, Dict[str, Any]] = {}

    for sub in sub_nums:
        if not isinstance(sub, int):
            raise TypeError(f"sub_nums must contain int, got {sub} ({type(sub)})")

        bids_sub = bids_subject_from_int(sub)
        nwb_sub = nwb_subject_from_int(sub)

        io = None
        nwb_path = None

        if sub not in out:
            out[sub] = {}

        try:
            nwb_path = find_nwb_file_for_subject(nwb_root, nwb_sub)
            io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
            nwbfile = io.read()

            # =========================================================
            # 1) Load raw LFP / iEEG
            # =========================================================
            lfp_macro, t_macro_raw, fs_macro = load_lfp_container(
                nwbfile, "LFP_macro", max_samples=max_nwb_samples
            )
            lfp_micro, t_micro_raw, fs_micro = load_lfp_container(
                nwbfile, "LFP_micro", max_samples=max_nwb_samples
            )

            t_macro_movie = convert_lfp_time_to_movie_reference(
                t_macro_raw, baseline_pre_movie=10.0
            )
            t_micro_movie = convert_lfp_time_to_movie_reference(
                t_micro_raw, baseline_pre_movie=10.0
            )

            # =========================================================
            # 2) Load movie frame index and derive seconds
            # =========================================================
            movieframe_time = load_movie_time(nwbfile, max_samples=max_nwb_samples)
            movieframe_time = np.asarray(movieframe_time, dtype=np.float64)

            movie_fps_nominal = 25.0
            movieframe_time_sec = movieframe_time / movie_fps_nominal

            # =========================================================
            # 3) Load eye tracking
            # =========================================================
            gaze, pupil, t_eye_raw = load_eye_gaze_and_pupil(
                nwbfile, max_samples=max_nwb_samples
            )

            if t_eye_raw is not None:
                t_eye_raw = np.asarray(t_eye_raw, dtype=np.float64)

            t_eye_movie = t_eye_raw

            fs_eye = None
            if t_eye_movie is not None and len(t_eye_movie) > 1:
                dt_eye = np.median(np.diff(t_eye_movie))
                if dt_eye > 0:
                    fs_eye = 1.0 / dt_eye

            if verbose:
                print(f"[sub {sub}] t_eye_raw range=({t_eye_movie[0]:.3f}, {t_eye_movie[-1]:.3f})")

            # =========================================================
            # 4) Load spikes
            # =========================================================
            spikes_list = load_spikes_units(nwbfile)

            # =========================================================
            # 5) Choose common analysis grid
            # =========================================================
            if grid_source == "movie_frame_time":
                t_grid = movieframe_time_sec

            elif grid_source == "uniform":
                if grid_dt is None:
                    raise ValueError("grid_dt must be provided when grid_source='uniform'.")

                if uniform_grid_include_baseline:
                    t0 = float(min(t_macro_movie[0], movieframe_time_sec[0]))
                else:
                    t0 = float(movieframe_time_sec[0])

                t1 = float(max(
                    movieframe_time_sec[-1],
                    t_macro_movie[-1],
                    t_micro_movie[-1] if len(t_micro_movie) > 0 else movieframe_time_sec[-1]
                ))

                t_grid = np.arange(t0, t1 + 0.5 * grid_dt, float(grid_dt), dtype=np.float64)

            elif grid_source == "lfp_raw":
                t_grid = t_macro_movie

            else:
                raise ValueError(
                    f"Unknown grid_source={grid_source!r}. "
                    f"Use one of: 'movie_frame_time', 'uniform', 'lfp_raw'."
                )
            
            # =========================================================
            # 6) Optional filtering on raw continuous signals
            # =========================================================
            if b_filter_data:
                if filter_config is None:
                    filter_config = {}

                if lfp_macro is not None and "lfp_macro" in filter_config:
                    lfp_macro = apply_signal_filter(lfp_macro, fs_macro, filter_config["lfp_macro"])

                if lfp_micro is not None and "lfp_micro" in filter_config:
                    lfp_micro = apply_signal_filter(lfp_micro, fs_micro, filter_config["lfp_micro"])

                if gaze is not None and fs_eye is not None and "eye_gaze" in filter_config:
                    gaze = apply_signal_filter(gaze, fs_eye, filter_config["eye_gaze"])

                if pupil is not None and fs_eye is not None and "pupil" in filter_config:
                    pupil = apply_signal_filter(pupil, fs_eye, filter_config["pupil"])

            # =========================================================
            # 6) Continuous modalities -> resample
            # =========================================================
            lfp_macro_rs = resample_continuous(lfp_macro, t_macro_movie, t_grid)
            lfp_micro_rs = resample_continuous(lfp_micro, t_micro_movie, t_grid)

            gaze_rs = None
            if gaze is not None and t_eye_movie is not None:
                gaze_rs = resample_continuous(gaze, t_eye_movie, t_grid)
                if gaze_rs.ndim == 2 and gaze_rs.shape[1] >= 2:
                    gaze_rs = gaze_rs[:, :2]

            pupil_rs = None
            if pupil is not None and t_eye_movie is not None:
                pupil_rs = resample_continuous(pupil, t_eye_movie, t_grid).astype(np.float32).squeeze()

            # =========================================================
            # 7) Spikes -> firing rate
            # =========================================================
            if compute_firing_rate:
                if firing_rate_bin_width is None:
                    if len(t_grid) > 1:
                        firing_rate_bin_width = float(np.median(np.diff(t_grid)))
                    else:
                        raise ValueError("Cannot infer firing_rate_bin_width from a grid of length < 2.")

                firing_rate = spikes_to_firing_rate(
                    spike_times_list=spikes_list,
                    t_grid=t_grid,
                    bin_width=firing_rate_bin_width,
                )
            else:
                firing_rate = None

            # =========================================================
            # 8) LFP band power
            # =========================================================
            lfp_bandpower: Dict[str, np.ndarray] = {}

            if compute_lfp_bandpower:
                if lfp_bandpower_on_raw:
                    for name, band in bands.items():
                        bp_raw = bandpower_hilbert(lfp_macro, fs=fs_macro, band=band)
                        bp_rs = resample_continuous(bp_raw, t_macro_movie, t_grid)
                        lfp_bandpower[name] = bp_rs
                else:
                    if len(t_grid) < 2:
                        raise ValueError("t_grid too short for resampled bandpower computation.")
                    dt_grid_local = np.median(np.diff(t_grid))
                    fs_grid = 1.0 / float(dt_grid_local)

                    for name, band in bands.items():
                        lfp_bandpower[name] = bandpower_hilbert(
                            lfp_macro_rs, fs=fs_grid, band=band
                        )

            # =========================================================
            # 9) fMRI
            # =========================================================
            bold_list: List[np.ndarray] = []
            fmri_error = None
            if load_fmri:
                try:
                    bold_list = load_bold_timeseries_wholebrain(
                        bids_root, bids_sub, max_runs=fmri_max_runs
                    )
                except Exception as e:
                    fmri_error = repr(e)
                    bold_list = []

            # =========================================================
            # 10) Output
            # =========================================================
            out[sub] = {
                "spikes": spikes_list,
                "firing_rate": firing_rate,

                "lfp_macro": lfp_macro_rs,
                "lfp_micro": lfp_micro_rs,
                "lfp_bandpower": lfp_bandpower,
                "eye_gaze": gaze_rs,
                "pupil": pupil_rs,

                "time_grid": np.asarray(t_grid, dtype=np.float64),

                # keep exact movie frame index
                "movieframe_time": np.asarray(movieframe_time, dtype=np.float64),
                # derived second-scale movie time
                "movieframe_time_sec": np.asarray(movieframe_time_sec, dtype=np.float64),

                "time_raw": {
                    "lfp_macro": t_macro_movie,
                    "lfp_micro": t_micro_movie,
                    "eye": None if t_eye_movie is None else np.asarray(t_eye_movie, dtype=np.float64),
                },

                "bold": bold_list,

                "meta": {
                    "sub": sub,
                    "bids_sub": bids_sub,
                    "nwb_sub": nwb_sub,
                    "nwb_path": str(nwb_path),

                    "fs_macro_raw": fs_macro,
                    "fs_micro_raw": fs_micro,
                    "fs_eye_raw": fs_eye,
                    "movie_fps_nominal": movie_fps_nominal,

                    "grid_source": grid_source,
                    "grid_dt": None if len(t_grid) < 2 else float(np.median(np.diff(t_grid))),
                    "time_reference": "movie_start_is_zero",

                    "lfp_baseline_seconds_pre_movie": 10.0,
                    "spikes_are_movie_referenced": True,
                    "lfp_is_movie_referenced_after_conversion": True,
                    "movieframe_time_is_frame_index": True,

                    "n_units": len(spikes_list),
                    "fmri_error": fmri_error,
                    "max_nwb_samples": max_nwb_samples,
                },
            }

            out[sub] = fill_nan_by_modality(out[sub])

            if verbose:
                print(f"[sub {sub}] nwb_path={nwb_path}")
                print(f"[sub {sub}] fs_macro={fs_macro}, fs_micro={fs_micro}, fs_eye={fs_eye}")
                print(f"[sub {sub}] t_macro_movie range=({t_macro_movie[0]:.3f}, {t_macro_movie[-1]:.3f})")
                print(f"[sub {sub}] movieframe_time(frame) range=({movieframe_time[0]:.3f}, {movieframe_time[-1]:.3f})")
                print(f"[sub {sub}] movieframe_time_sec range=({movieframe_time_sec[0]:.3f}, {movieframe_time_sec[-1]:.3f})")
                print(f"[sub {sub}] t_grid len={len(t_grid)} source={grid_source}")

        except Exception as e:
            if sub not in out:
                out[sub] = {}
            meta = out[sub].get("meta", {})
            meta.update({
                "sub": sub,
                "bids_sub": bids_sub,
                "nwb_sub": nwb_sub,
                "nwb_path": str(nwb_path) if nwb_path else None,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            out[sub]["meta"] = meta

        finally:
            if io is not None:
                io.close()

    return out