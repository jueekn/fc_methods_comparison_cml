# Basic
import numpy as np
import scipy
import scipy.stats
import os
import itertools
import warnings
import sys
from copy import deepcopy

# Data Loading
import cmlreaders as cml #Penn Computational Memory Lab's library of data loading functions

# Data Handling
import os
from os import listdir as ld
import os.path as op
from os.path import join, exists as ex
import time
import datetime

# Data Analysis
import pandas as pd
import xarray as xr

# EEG & Signal Processing
import ptsa
from ptsa.data.readers import BaseEventReader, EEGReader, CMLEventReader, TalReader
from ptsa.data.filters import MonopolarToBipolarMapper, MorletWaveletFilter
from ptsa.data.timeseries import TimeSeries

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Parallelization
import cmldask.CMLDask as da
from cmldask.CMLDask import new_dask_client_slurm as cl
import cmldask
from cluster import get_exceptions_quiet as get_ex

# Custom
from cstat import * #circular statistics
from misc import * #helper functions for loading and saving data, and for other purposes
from matrix_operations import * #matrix operations
import os
from simulate_eeg import AVAILABLE_SIMULATIONS, NULL_SIMULATION_TAGS, simulation_parameters, sample_eeg
from mne_connectivity import spectral_connectivity_epochs, envelope_correlation

root_dir = f'/scratch/jueenaik/retrieval_connectivity_revision'

import helper
from helper import *
helper.root_dir = root_dir

metrics = ["coh", "plv", "ppc", "ciplv", "pli", "wpli", "dpli", "aec"]
asymm = {"dpli"}               
#multivar = {"mim", "gc", "gc_tr"} 

bands = {"alpha": (8,12), "theta": (3, 9)}

regionlabels = list(helper.get_region_information("region_labels"))  
reg2i = {r: i for i, r in enumerate(regionlabels)}

def symmetrize_dense(mat, diag_value=1.0, eps=1e-12):
    '''Creates symmetrical regionxregion matrices if unsymmetrical (mne outputs are lower triangular by default)'''
    mat = np.asarray(mat)
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1]
    n = mat.shape[0]

    iu = np.triu_indices(n, 1)
    upper = mat[iu]

    upper_empty = (
        np.all(~np.isfinite(upper)) or
        np.nanmax(np.abs(np.nan_to_num(upper))) < eps
    )

    if upper_empty:
        out = mat + mat.T         
    else:
        out = 0.5 * (mat + mat.T)  

    np.fill_diagonal(out, diag_value)
    return out
 

def compute_metric_matrix(data, sfreq, m, fmin, fmax):
    '''Self explanatory'''
    if m == "aec":
        return compute_aec(data)

    if m in asymm:
        return compute_spectral_fc_directed(data, sfreq, method=m, fmin=fmin, fmax=fmax, faverage=True)

    return compute_spectral_fc(data, sfreq, method=m, fmin=fmin, fmax=fmax, faverage=True)

def compute_spectral_fc(data, sfreq, method, fmin, fmax, faverage=True):
    '''Computes all MNE phase based functions except multivariate ones like GC/MIM'''
    con = spectral_connectivity_epochs(
        data, method=method, mode="multitaper",
        sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=faverage,
        mt_adaptive=False, n_jobs=1, verbose=False,
    )
    out = con.get_data(output="dense")
    #print(f"con data: {out}")
    if faverage:
        out = out[..., 0]
    #print(f"con data': {out}")
    out = np.asarray(out)
    if out.ndim == 2:
        out = symmetrize_dense(out, diag_value=np.nan)
    return out


def compute_aec(data, orthogonalize="pairwise"):
    '''Computes Amplitude Envelope Correlations'''
    conn = envelope_correlation(data, orthogonalize=orthogonalize, verbose=False)  
    out = np.asarray(conn.get_data(output="dense"))
    out = np.squeeze(out)

    if out.ndim == 3:
        out = np.nanmean(out, axis=0)

    out = symmetrize_dense(out, diag_value=np.nan)
    return out

def compute_session_fc(dfrow, beh=beh, band="theta",
                       metrics=("coh","plv","ppc","ciplv","pli","wpli","aec","dpli"),
                       do_aec=False, simulation_tag=None):
    '''Main pipeline to load EEG + events and compute FC for a session'''
    events = load_events(dfrow, beh)
    if events is None:
        return None

    eeg, mask = get_beh_eeg(dfrow, events, save=False, simulation_tag=simulation_tag) 
    sfreq = float(eeg.samplerate)
    data = np.asarray(eeg.data)                     
    #data = timebin_timeseries(data, sfreq, np.mean)  
    #data, good_ch = subset_good_channels(dfrow, data)

    fmin, fmax = bands[band]

    out = {
        "meta": {
            "sub": dfrow["sub"], "exp": dfrow["exp"], "sess": dfrow["sess"], "loc": dfrow["loc"], "mon": dfrow["mon"],
            "beh": beh, "band": band, "sfreq": sfreq, "n_epochs": data.shape[0], "n_ch": data.shape[1], "simulation_tag": simulation_tag,
        },
        #"good_ch_mask": good_ch,
        "metrics": {},
    }

    data_succ = data[mask]
    data_fail = data[~mask]
    #print(f"data succ: {len(data_succ)}")

    for m in metrics:
        C_succ = compute_metric_matrix(data_succ, sfreq, m, fmin, fmax)
        C_fail = compute_metric_matrix(data_fail, sfreq, m, fmin, fmax)
        print(f"C succ: {C_succ}")
        print(f"C fail: {C_fail}")
        out["metrics"][m] = {"succ": C_succ, "fail": C_fail, "diff": C_succ - C_fail}
    return out

def _all_ordered_pairs(n_ch):
    '''Only for directed methods'''
    seeds, targets = np.where(~np.eye(n_ch, dtype=bool))
    return seeds, targets  

def compute_spectral_fc_directed(data, sfreq, method, fmin, fmax, faverage=True, **kwargs):
    '''Only for directed methods'''
    n_ch = data.shape[1]
    seeds, targets = _all_ordered_pairs(n_ch)
    #print(seeds, targets)
    con = spectral_connectivity_epochs(
        data, method=method, mode="multitaper",
        sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=faverage,
        indices=(seeds, targets),
        mt_adaptive=False, n_jobs=1, verbose=False,
        **kwargs
    )
    vals = con.get_data() 
    if faverage:
        vals = vals[:, 0]
    M = np.full((n_ch, n_ch), np.nan, float)
    M[seeds, targets] = vals
    print(M)
    np.fill_diagonal(M, np.nan)
    return M

def electrode_to_region_mean(el_mat, reg_full, reg_order=regionlabels, fill_value=np.nan):
    '''Averages over electrodes in a region to return region average FC value'''
    n = el_mat.shape[0]
    #print(f"n electrodes: {n}")
    assert el_mat.shape == (n, n)
    assert len(reg_full) == n

    idx = np.array([reg2i.get(r, -1) for r in reg_full], dtype=int)
    keep = idx >= 0
    #print(f"idx: {idx}")

    R = len(reg_order)
    out = np.full((R, R), fill_value, dtype=float)

    for i in range(R):
        ai = np.where(keep & (idx == i))[0]
        if ai.size == 0:
            continue
        for j in range(R):
            bj = np.where(keep & (idx == j))[0]
            if bj.size == 0:
                continue
            block = el_mat[np.ix_(ai, bj)]
            out[i, j] = np.nanmean(block)
    #print(f"region mat: {out}")

    return out
