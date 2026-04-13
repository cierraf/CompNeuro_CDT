from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pynwb import NWBHDF5IO

# fMRI (optional)
from bids import BIDSLayout
import nibabel as nib
import traceback

# signal processing
from scipy.signal import butter, sosfiltfilt, hilbert
from collections.abc import Iterable

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
    # t[i] = starting_time + i/rate
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

    # timestamps can be large; only load if present
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

    # optional preference ordering
    if prefer:
        prefer_l = [p.lower() for p in prefer]
        keys_str = sorted(
            keys_str,
            key=lambda s: 0 if any(p in s.lower() for p in prefer_l) else 1
        )

    last_err = None
    for k in keys_str:
        try:
            _ = container[k]   # probe
            return k
        except Exception as e:
            last_err = e
            continue

    raise TypeError(f"Could not index container with any stringified key. "
                    f"raw keys={keys_raw[:10]}... last_err={repr(last_err)}")

# -----------------------
# Spikes -> firing rate
# -----------------------
def spikes_to_firing_rate(spike_times_list: List[np.ndarray], t_grid: np.ndarray) -> np.ndarray:
    """
    Bin spikes into firing rate on t_grid (sec).
    We interpret t_grid as bin centers; derive bin edges from midpoints.
    Output shape: (T, n_units) in Hz.
    """
    T = len(t_grid)
    n_units = len(spike_times_list)
    if T < 2:
        raise ValueError("t_grid must have at least 2 points.")

    # bin edges from centers
    dt = np.median(np.diff(t_grid))
    edges = np.concatenate(([t_grid[0] - dt/2], t_grid + dt/2))
    fr = np.zeros((T, n_units), dtype=np.float32)

    for i, st in enumerate(spike_times_list):
        if st is None or len(st) == 0:
            continue
        counts, _ = np.histogram(st, bins=edges)
        fr[:, i] = counts.astype(np.float32) / float(dt)  # Hz
    return fr


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

    # IMPORTANT: do NOT use `if "EyeTracking" in beh` (can trigger bad __contains__)
    beh_keys_raw = list(beh.keys())
    beh_keys_str = {k if isinstance(k, str) else str(k) for k in beh_keys_raw}

    if "EyeTracking" in beh_keys_str:
        et = beh["EyeTracking"]  # now safe, we only index with str
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
        # st is typically an array-like of spike times
        spikes.append(np.asarray(st, dtype=np.float64))
    return spikes


def load_movie_time(nwbfile, max_samples: Optional[int] = None) -> np.ndarray:
    """
    Use stimulus['movieframe_time'] as the common time grid (seconds).
    """
    stim = nwbfile.stimulus
    if "movieframe_time" not in stim:
        raise KeyError("movieframe_time not found in nwbfile.stimulus")
    ts = stim["movieframe_time"]
    data, _, _ = get_ts_data_and_time(ts, max_samples=max_samples)
    data = np.asarray(data).squeeze().astype(np.float64)
    return data


# -----------------------
# fMRI optional extraction
# -----------------------
def load_bold_timeseries_wholebrain(bids_root: Union[str, Path], bids_sub: str, max_runs: Optional[int] = None) -> List[np.ndarray]:
    """
    Returns list of arrays (T,V). Uses nilearn NiftiMasker.
    """
    from nilearn.maskers import NiftiMasker

    layout = BIDSLayout(str(bids_root), validate=False)
    bold_files: List[str] = layout.get(subject=bids_sub, datatype="func", suffix="bold",
                                       extension=["nii", "nii.gz"], return_type="file")
    if max_runs is not None:
        bold_files = bold_files[:max_runs]

    masker = NiftiMasker(standardize=False)
    out = []
    for f in bold_files:
        img = nib.load(f)
        X = masker.fit_transform(img)  # (T,V)
        out.append(X.astype(np.float32))
    return out


# -----------------------
# Main multimodal loader
# -----------------------
def load_multimodal_subjects(
    sub_nums: Union[Iterable[int], set],
    nwb_root: Union[str, Path],
    bids_root: Union[str, Path],
    *,
    max_nwb_samples: Optional[int] = None,
    resample_lfp: bool = False,
    use_movie_time_as_grid: bool = True,
    grid_dt: Optional[float] = None,
    compute_firing_rate: bool = True,
    compute_lfp_bandpower: bool = True,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    load_fmri: bool = False,
    fmri_max_runs: Optional[int] = None,
    verbose: bool = True,
    **kwargs,   # <-- 兼容多余参数，避免 TypeError
) -> Dict[int, Dict[str, Any]]:

    if kwargs and verbose:
        print(f"[WARN] load_multimodal_subjects ignored kwargs: {sorted(kwargs.keys())}")

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

        nwb_path = None
        io = None

        try:
            nwb_path = find_nwb_file_for_subject(nwb_root, nwb_sub)

            io = NWBHDF5IO(str(nwb_path), "r", load_namespaces=True)
            nwbfile = io.read()

            # -----------------------------
            # 1) 先读 LFP（拿到原始时间轴）
            # -----------------------------
            lfp_macro, t_macro, fs_macro = load_lfp_container(
                nwbfile, "LFP_macro", max_samples=max_nwb_samples
            )
            lfp_micro, t_micro, fs_micro = load_lfp_container(
                nwbfile, "LFP_micro", max_samples=max_nwb_samples
            )

            if verbose:
                print(f"[sub {sub}] nwb_path={nwb_path}")
                print(f"[sub {sub}] lfp_macro shape={lfp_macro.shape} dtype={lfp_macro.dtype}")
                print(f"[sub {sub}] t_macro len={len(t_macro)} range=({t_macro[0]:.3f}, {t_macro[-1]:.3f}) fs={fs_macro}")
                print(f"[sub {sub}] lfp_macro min/max=({float(np.min(lfp_macro)):.3f}, {float(np.max(lfp_macro)):.3f})")

            # -----------------------------
            # 2) 决定公共时间网格 t_grid
            #    - 你现在要“先观测原始数据”
            #    - resample_lfp=False => t_grid = t_macro（完整 LFP 采样）
            # -----------------------------
            movie_time = None
            movie_time_error = None

            if not resample_lfp:
                t_grid = t_macro
            else:
                if use_movie_time_as_grid:
                    # 用 movie frame time 作为网格（会很稀疏，仅在你明确要这样时用）
                    t_grid = load_movie_time(nwbfile, max_samples=max_nwb_samples)
                else:
                    if grid_dt is None:
                        raise ValueError("If use_movie_time_as_grid=False you must provide grid_dt.")
                    t0, t1 = float(t_macro[0]), float(t_macro[-1])
                    t_grid = np.arange(t0, t1, float(grid_dt), dtype=np.float64)

            # movie_time 单独读出来（用于对齐/标注事件），不再用于强制重采样
            if use_movie_time_as_grid:
                try:
                    movie_time = load_movie_time(nwbfile, max_samples=max_nwb_samples)
                except Exception as e:
                    movie_time_error = repr(e)
                    movie_time = None

            if verbose:
                print(f"[sub {sub}] resample_lfp={resample_lfp} -> t_grid len={len(t_grid)} "
                      f"(expect == len(t_macro) when resample_lfp=False)")
                if movie_time is not None:
                    print(f"[sub {sub}] movie_time len={len(movie_time)} range=({movie_time[0]:.3f}, {movie_time[-1]:.3f})")
                else:
                    print(f"[sub {sub}] movie_time=None ({movie_time_error})")

            # -----------------------------
            # 3) LFP 是否重采样
            # -----------------------------
            if resample_lfp:
                lfp_macro_rs = resample_continuous(lfp_macro, t_macro, t_grid)
                lfp_micro_rs = resample_continuous(lfp_micro, t_micro, t_grid)
            else:
                lfp_macro_rs = lfp_macro
                lfp_micro_rs = lfp_micro

            # -----------------------------
            # 4) Eye + pupil（如果你只看 LFP，可之后再关掉）
            # -----------------------------
            gaze, pupil, t_eye = load_eye_gaze_and_pupil(nwbfile, max_samples=max_nwb_samples)

            gaze_rs = None
            if gaze is not None and t_eye is not None:
                gaze_rs = resample_continuous(gaze, t_eye, t_grid)
                if gaze_rs.ndim == 2 and gaze_rs.shape[1] >= 2:
                    gaze_rs = gaze_rs[:, :2]

            pupil_rs = None
            if pupil is not None and t_eye is not None:
                pupil_rs = resample_continuous(pupil, t_eye, t_grid).astype(np.float32).squeeze()

            # -----------------------------
            # 5) Spikes -> firing rate（t_grid 很密时会很稀疏，但能跑）
            # -----------------------------
            spikes_list = load_spikes_units(nwbfile)
            firing_rate = spikes_to_firing_rate(spikes_list, t_grid) if compute_firing_rate else None

            # -----------------------------
            # 6) LFP band power（注意：如果你 resample 且 grid_dt 改了，fs 也要跟着变）
            # -----------------------------
            lfp_bandpower: Dict[str, np.ndarray] = {}
            if compute_lfp_bandpower:
                fs_for_bp = fs_macro if (not resample_lfp or grid_dt is None) else (1.0 / float(grid_dt))
                for name, band in bands.items():
                    lfp_bandpower[name] = bandpower_hilbert(lfp_macro_rs, fs=fs_for_bp, band=band)

            # -----------------------------
            # 7) fMRI（可选）
            # -----------------------------
            bold_list: List[np.ndarray] = []
            fmri_error = None
            if load_fmri:
                try:
                    bold_list = load_bold_timeseries_wholebrain(bids_root, bids_sub, max_runs=fmri_max_runs)
                except Exception as e:
                    fmri_error = repr(e)
                    bold_list = []

            # -----------------------------
            # 8) 输出
            # -----------------------------
            out[sub] = {
                "spikes": spikes_list,
                "firing_rate": firing_rate,
                "lfp_macro": lfp_macro_rs,
                "lfp_micro": lfp_micro_rs,
                "lfp_bandpower": lfp_bandpower,
                "eye_gaze": gaze_rs,
                "pupil": pupil_rs,
                "time_grid": np.asarray(t_grid, dtype=np.float64),  # <-- 真实公共时间轴
                "movie_time": None if movie_time is None else np.asarray(movie_time, dtype=np.float64),
                "bold": bold_list,
                "meta": {
                    "sub": sub,
                    "bids_sub": bids_sub,
                    "nwb_sub": nwb_sub,
                    "nwb_path": str(nwb_path),
                    "fs_macro": fs_macro,
                    "fs_micro": fs_micro,
                    "n_units": len(spikes_list),
                    "fmri_error": fmri_error,
                    "movie_time_error": movie_time_error,
                    "resample_lfp": resample_lfp,
                    "max_nwb_samples": max_nwb_samples,
                },
            }

        except Exception as e:
            out[sub] = {
                "meta": {
                    "sub": sub,
                    "bids_sub": bids_sub,
                    "nwb_sub": nwb_sub,
                    "nwb_path": str(nwb_path) if nwb_path else None,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            }

        finally:
            if io is not None:
                io.close()

    return out