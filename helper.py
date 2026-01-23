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

bands = {"alpha": (8,12), "theta": (3, 9)}

beh_to_event_windows = {'en': [250-1000, 1250+1000],
                     'en_all': [250-1000, 1250+1000],
                     'rm': [-1000, 0],
                     'ri': [-1000, 0]}

beh_to_epochs = {'en': np.arange(250, 1250, 200),
              'en_all': np.arange(250, 1250, 200),
              'rm': np.arange(-1000, 0, 200),
              'ri': np.arange(-1000, 0, 200)}

behavioral_names = {'en': 'Encoding',
                    'rm': 'Retrieval',
                    'ri': 'Recall Accuracy'}

# root_dir set in main analysis notebook
def print_root_dir():
    print(root_dir)

def load_events(dfrow, beh):
    
    '''
    Loads behavioral events for a particular experimental session and behavioral contrast.
    Requires that the events data and metadata file have already been saved out (see the "Get behavioral events" section in WholeBrainConnectivityPPCRevision.ipynb).
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri').
        
    Returns:
        events : pandas.DataFrame
            Behavioral events for a particular experimental session and behavioral contrast.
    '''
    
    evs_path = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events.json')
    evs_metadata_path = join(root_dir, beh, 'events', f'{ftag(dfrow)}_events_metadata.json')
    if not ex(evs_path): return None
    events = pd.read_json(evs_path)
    events_metadata = pd.read_json(evs_metadata_path, typ='series')
    events.attrs = events_metadata
    events.attrs['mask'] = np.asarray(events.attrs['mask'])
    
    return events

def get_eeg(dfrow, events, start, end, simulation_tag=None):
    
    '''
    Returns EEG signal for a particular session and set of behavioral events.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        start : float
            Time (ms) at which returned EEG clip should begin, relative to a particular event.
        end : float
            Time (ms) at which returned EEG clip should end, relative to a particular event.
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
    
    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
        numpy.array
            List of boolean variables indicating whether the event was a successful memory event (True) or an unsuccessful memory event (False).
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    sess_list_df = pd.read_json(join(root_dir, 'sess_list_df.json'))
    sess_list_df.set_index(['sub', 'exp', 'sess', 'loc', 'mon'], inplace=True)
    eeg_data_source = sess_list_df.loc[(sub, exp, sess, loc, mon), 'eeg_data_source']
    
    events['event_idx'] = np.arange(len(events))
    events.sort_values(by=['mstime', 'eegoffset'], inplace=True)
    events.attrs['mask'] = events.attrs['mask'][events['event_idx']]
    events.drop('event_idx', axis=1, inplace=True)
    
    if eeg_data_source == 'cmlreaders':
        reader = cml.CMLReader(subject=sub, 
                               experiment=exp, 
                               session=sess,
                               localization=loc,
                               montage=mon)
        pairs = get_pairs(dfrow)
        eeg = reader.load_eeg(events, start, end, scheme=pairs)
        eeg = eeg.to_ptsa()
        time_unit = 'millisecond'
        
    elif eeg_data_source == 'ptsa':
        eeg = get_ptsa_eeg(dfrow, events, start, end)
        time_unit = 'second'
    
    sr = float(eeg.samplerate)
    if 'sr' in events.attrs: 
        assert events.attrs['sr'] == sr, 'sampling rate is wrong'
        
    dim_map = {}
    if 'events' in eeg.dims: dim_map['events'] = 'event'
    if 'channels' in eeg.dims: dim_map['channels'] = 'channel'
    eeg = eeg.rename(dim_map)
    eeg = eeg.transpose('event', 'channel', 'time')
    
    if simulation_tag not in ['standard', '', None]:
        # replace experimentally recorded EEG with simulated EEG to validate analysis pipeline
        eeg = replace_w_simulated_EEG(eeg,
                                      dfrow,
                                      eeg_data_source=eeg_data_source,
                                      time_unit=time_unit,
                                      condition_mask=events.attrs['mask'],
                                      simulation_tag=simulation_tag)
    
    return eeg, events.attrs['mask']

def get_ptsa_eeg(dfrow, events, start, end):
    
    '''
    Returns EEG signal for a particular session and set of behavioral events, using the ptsa readers. Used to load EEG for pyFR experimental sessions whose data could not be loaded with cmlreaders.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        start : float
            Time (ms) at which returned EEG clip should begin, relative to a particular event.
        end : float
            Time (ms) at which returned EEG clip should end, relative to a particular event.
    
    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    mon_ = '' if mon==0 else f'_{mon}' #for tal_reader path name

    events = events.to_records()
    tal_reader = TalReader(filename=f'/data/eeg/{sub}{mon_}/tal/{sub}{mon_}_talLocs_database_bipol.mat')
    channels = tal_reader.get_monopolar_channels()
    eeg = EEGReader(events=events, channels=channels,
                    start_time=start/1000, end_time=end/1000).read()

    bipolar_pairs = tal_reader.get_bipolar_pairs()
    pairs = get_pairs(dfrow)
    pair_tuples_select = [tuple((int(pair[0]), int(pair[1]))) for pair in pairs[['contact_1', 'contact_2']].values]
    bipolar_pairs = np.asarray([pair for pair in bipolar_pairs if tuple((int(pair[0]), int(pair[1]))) in pair_tuples_select], dtype=[('ch0', 'S3'), ('ch1', 'S3')]).view(np.recarray)
    mapper= MonopolarToBipolarMapper(bipolar_pairs=bipolar_pairs)
    eeg = mapper.filter(timeseries=eeg)
    
    return eeg

def get_beh_eeg(dfrow, events, save=True, simulation_tag=None):
    
    '''
    Returns processed EEG signal to be analyzed for a particular behavioral contrast.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        save : bool
            Whether to save out the loaded raw EEG signal (True) or not (False).
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
    
    Returns:
        eeg : ptsa.data.TimeSeries
            EEG clip.
    '''

    beh = events.attrs['beh']
    start, end = beh_to_event_windows[beh]
    
    eeg, mask = get_eeg(dfrow, events, start, end, simulation_tag=simulation_tag)
    if save: np.save(join(root_dir, beh, 'eeg', f'{ftag(dfrow)}_raw_eeg.npy'), eeg.data)
    
    if beh in ['rm', 'ri']: eeg = mirror_buffer(eeg, 1000)
    eeg = eeg.resampled(250)
    eeg = notch_filter(eeg, dfrow['sub'])

    return eeg, mask

def notch_filter(eeg, sub):
    
    '''
    Applies a Butterworth filter to EEG signal at 60 or 50 Hz to remove line noise.
    
    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG signal to be notch-filtered.
        sub : str
            Subject code. Used to decide notch filter frequency: if a German (Freiburg) subject, filter at 50 Hz, else at 60 Hz. 
        
    Returns:
        eeg : ptsa.data.TimeSeries
            Notch-filtered EEG signal.
    '''
    
    filter_freqs = [48., 52.] if 'FR' in sub else [58., 62.]
    
    from ptsa.data.filters import ButterworthFilter
    b_filter = ButterworthFilter(freq_range=filter_freqs, filt_type='stop', order=4)
    eeg = b_filter.filter(timeseries=eeg)

    return eeg

def mirror_buffer(eeg, buffer_length, axis=-1):
    
    '''
    Append a mirror buffer to EEG signal.
    If the EEG signal passed to the function is of the form (x_1, ..., x_n), then the buffered signal will be of the form (x_i, ..., x_1, x_1, ..., x_n, x_n, ..., x_(n-i+1)), where 1 <= i <= n.
    
    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG signal to which to append the mirror buffer. 
        buffer_length : float
            Duration (ms) of buffer.
        
    Returns:
        buffered_eeg : ptsa.data.TimeSeries
            EEG signal with mirror buffer appended on both sides.
    '''
    
    sr = float(eeg.samplerate)
    tmpt_length = int(buffer_length * (1/1000) * sr)
    coords=eeg.coords
    coords['time'] = coords['time'][..., tmpt_length::-1]
    buffer = TimeSeries(data=eeg[..., tmpt_length::-1], dims=eeg.dims, coords=coords)
    buffered_eeg = xr.concat([buffer, eeg, buffer], dim='time')
    
    return buffered_eeg

def get_pairs(dfrow):
    
    '''
    Returns the bipolar electrode pairs data for a session.
    Requires that the bipolar electrode pairs data have been already saved out (see 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
    
    Returns:
        pandas.DataFrame
            Bipolar electrode pairs data.
    
    '''
    
    path = join(root_dir, 'electrode_information', 'pairs', f'{ftag(dfrow)}_pairs.json')
    if ex(path): return pd.read_json(path).fillna('nan')
    else: return None

def get_localization(dfrow):
    
    '''
    Returns the localization data for a session. Requires that the localization data have already been saved out (see the 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
    
    Returns:
        localization : pandas.DataFrame
            Localization data.
    '''
    
    path = join(root_dir, 'electrode_information', 'localization', f'{ftag(dfrow)}_localization.json')
    if ex(path): localization = pd.read_json(path).fillna('nan')
    else: return []
    localization['level_1'] = localization.apply(lambda r: tuple(r['level_1']) if isinstance(r['level_1'], list) else r['level_1'], axis=1)
    localization = localization.set_index(['level_0', 'level_1']).rename_axis([None, None], axis='index')
    
    return localization

def get_sr(dfrow):
    
    '''
    Returns the sampling rate of a session.
    Requires that the localization data have been already saved out in the session list DataFrame (see 'Check data availability' section in WholeBrainConnectivityPPCRevision.ipynb). 
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        
    Returns:
        sr : float
            Sampling rate.
    '''
    
    sub, exp, sess, loc, mon = dfrow[['sub', 'exp', 'sess', 'loc', 'mon']]
    sess_list_df = pd.read_json(join(root_dir, 'sess_list_df_data_check.json'))
    sess_list_df.set_index(['sub', 'exp', 'sess', 'loc', 'mon'], inplace=True)
    sr = sess_list_df.loc[(sub, exp, sess, loc, mon), 'sr']
    
    return sr

def find_overlapping_pairs(pairs):
    
    '''
    Returns a list of bipolar pairs that share a monopolar contact.
    
    Parameters:
        pairs : pandas.DataFrame
            Bipolar pairs data.
    
    Returns:
        overlapping_pairs : list
            List of tuples of the form (i, j), where i is the row index of a bipolar pair in the pairs DataFrame and j is the row index of a bipolar pair that shares a monopolar contact.
    '''

    overlapping_pairs = []
    for elec1 in np.arange(len(pairs)):
        for elec2 in np.arange(len(pairs)):

            electrode_pair1 = pairs.iloc[elec1]
            electrode_pair2 = pairs.iloc[elec2]

            this_pair_channels = [electrode_pair1['contact_1'], electrode_pair1['contact_2'], electrode_pair2['contact_1'], electrode_pair2['contact_2']]

            if len(np.unique(this_pair_channels)) < 4:
                overlapping_pairs.append((elec1, elec2))

    return overlapping_pairs

def get_region_information(key=None):
    
    '''
    Returns information about the regionalization scheme. 
    
    Parameters:
        key (str): Which information to return ('region_translator', 'original_labels', 'unique_region_names', or 'region_labels').
    '''
    
    region_translator = pd.read_csv('region_translator.csv', na_filter=False).set_index('atlas_label')
    original_labels = np.unique(region_translator.index)
    unique_region_names = np.sort(region_translator.query('region != "nan"')['region'].unique())
    region_labels = np.char.add(np.repeat(['L ', 'R '], len(unique_region_names)).astype(str), np.tile(unique_region_names, 2).astype(str))
    
    region_lists = pd.Series({'region_translator': region_translator,
                              'original_labels': original_labels,
                              'unique_region_names': unique_region_names,
                              'region_labels': region_labels})
    
    return region_lists[key] if key is not None else region_lists

def get_atlas_labels(pairs, localization): 
    
    '''
    Returns the label from the best available brain region atlas for a session's electrode channels.
    
    Parameters:
        pairs : pandas.DataFrame
            Bipolar electrode pairs data.
        localization : pandas.DataFrame
            Localization data.
    
    Returns:
        pandas.DataFrame
            Table of bipolar electrode pairs, their best atlas label, and the atlases from which those labels were taken.
    '''
    
    if len(localization) > 0: 

        localization = localization.loc['pairs'].reset_index()
        localization['label'] = localization.apply(lambda r: r['index'][0] + '-' + r['index'][1], axis=1)
        localization.set_index('label', inplace=True)
        for col in ['atlases.mtl', 'atlases.dk', 'atlases.whole_brain']:
            pairs[col] = pairs.apply(lambda pair: localization.loc[pair['label'], col] if ((col in localization.columns) & (pair['label'] in localization.index)) else np.nan, axis=1)

    atlases = pd.DataFrame([(col, iCol) for iCol, col in enumerate(['stein.region', 'das.region', 'atlases.mtl', 'atlases.whole_brain', 'wb.region', 'mni.region', 'atlases.dk', 'dk.region', 'ind.corrected.region', 'mat.ind.corrected.region', 'ind.snap.region', 'mat.ind.snap.region', 'ind.dural.region', 'mat.ind.dural.region', 'ind.region', 'mat.ind.region', 'avg.corrected.region', 'avg.mat.corrected.region', 'avg.snap.region', 'avg.mat.snap.region', 'avg.dural.region', 'avg.mat.dural.region', 'avg.region', 'avg.mat.region', 'mat.tal.region'])], columns=['atlas', 'priority']).query('atlas in @pairs.columns').sort_values(by='priority', ascending=True, axis=0)['atlas'].values
    
    def label_pair(pair):
        
        for atlas in atlases:
            test_region = str(pair[atlas])
            if (atlas in ['tal.region', 'mat.tal.region']) and np.any([test_region in label for label in ['Parahippocampal Gyrus', 'Uncus', 'Lentiform Nucleus', 'Caudate', 'Thalamus']]):
                continue
            if test_region.lower() not in ['nan', '[nan]', 'none', 'unknown', 'misc', '', ' ', 'left tc', '*']:
                return test_region, atlas
        return 'nan', 'no atlas'
    
    pairs[['atlas_label', 'atlas']] = pairs.apply(lambda pair: label_pair(pair), axis=1, result_type='expand')
    return pairs.rename({'label': 'pair_label'}, axis=1)[['pair_label', 'atlas_label', 'atlas']]

def regionalize_electrodes(pairs, localization):
    
    '''
    Returns a list of each channel's regionalization, in order of the channels. 
    
    Parameters:
        pairs : pandas.DataFrame
            Bipolar electrode pairs data.
        localization : pandas.DataFrame
            Localization data.
        
    Returns:
        regionalizations : array_like
            List of each channel's regionalization, in order of the channels. 
    '''

    regionalizations = get_atlas_labels(pairs, localization)
    region_translator = get_region_information('region_translator')
    original_labels = get_region_information('original_labels')
    regionalizations['region'] = regionalizations.apply(lambda r: region_translator.loc[r['atlas_label'], 'region'] if r['atlas_label'] in original_labels else 'nan', axis=1)    

    def get_hemisphere_region_label(r):
        
        if 'hemisphere' in pairs.columns: 
            hemisphere = pairs.loc[r.name, 'hemisphere']
            if hemisphere in ['L', 'R']: return hemisphere
        
        if r['atlas_label'] in original_labels:
            hemisphere = region_translator.loc[r['atlas_label'], 'hemisphere']
            if hemisphere in ['L', 'R']: return hemisphere
        
        atlases_x = pd.DataFrame([(col, iCol) for iCol, col in enumerate(['mni.x','ind.corrected.x', 'ind.snap.x', 'ind.dural.x', 'ind.x', 'avg.corrected.x', 'avg.snap.x', 'avg.dural.x', 'avg.x', 'tal.x', 'x'])], columns=['atlas', 'priority']).query('atlas in @pairs.columns').sort_values(by='priority', ascending=True, axis=0)['atlas'].values
        
        for atlas_x in atlases_x:
            x_coord = pairs.loc[r.name, atlas_x]
            if not isinstance(x_coord, (int, float)): continue
            if x_coord < 0: return 'L'
            elif x_coord > 0: return 'R'
        
    regionalizations['hemisphere'] = regionalizations.apply(lambda r: get_hemisphere_region_label(r), axis=1)
    return (regionalizations['hemisphere'] + ' ' + regionalizations['region']).values

def timebin_phase_timeseries(timeseries, sr): return timebin_timeseries(timeseries, sr, circ_mean)
    
def timebin_power_timeseries(timeseries, sr): return timebin_timeseries(timeseries, sr, np.mean)

def timebin_timeseries(timeseries, sr, average_function, bin_size_ms=200):
    
    '''
    Averages time series within timebins and returns timebinned time series.
    
    Parameters:
        timeseries : numpy.array
            Time series. Timepoints should be along the last dimension.
        sr : float
            Sample rate of the time series.
        average_function : function
            Function used for averaging within the timebins (circ_mean or np.mean).
        bin_size_ms : float
            Duration (ms) represented by a single timebin.
            
    Returns:
        timebinned_timeseries : numpy.array
            Timebinned timeseries.
    '''
    
    bin_size = int(sr * (1/1000) * bin_size_ms)
    bin_count = int(np.round(timeseries.shape[-1] / bin_size))
    
    timebinned_timeseries = []
    for iBin in range(bin_count):
        left_edge = iBin*bin_size
        right_edge = (iBin+1)*bin_size if iBin < bin_count - 1 else None
        this_epoch = average_function(timeseries[..., left_edge:right_edge], axis=-1)
        timebinned_timeseries.append(this_epoch)

    timebinned_timeseries = np.asarray(timebinned_timeseries)
    timebinned_timeseries = np.moveaxis(timebinned_timeseries, 0, -1)
    
    return timebinned_timeseries

def clip_buffer(timeseries, buffer_length):
    
    '''
    Returns signal after clipping buffer.
    
    Parameters:
        timeseries : xarray.DataArray, ptsa.data.TimeSeries
            Time series (EEG, power, phase) with 'time' dimension.
        buffer_length : float
            Number of samples (NOT duration) to clip from both ends of the time series.
        
    Returns
        xarray.DataArray, ptsa.data.TimeSeries
            Time series with buffer clipped.
    '''
    
    return timeseries.isel(time=np.arange(buffer_length, len(timeseries['time'])-buffer_length))

def get_phase(eeg, freqs):

    '''
    Returns time series of spectral phase values from Morlet wavelet convolution.
    
    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG clip.
        freqs : numpy.array
            Wavelet frequencies at which to extract phase values.
            
    Returns
        phase : ptsa.data.TimeSeries
            Time series of phase values.
    '''
    
    wavelet_filter = MorletWaveletFilter(freqs=freqs, width=5, output='phase', complete=True)
    phase = wavelet_filter.filter(timeseries=eeg)
    phase = phase.transpose('event', 'channel', 'frequency', 'time')
    
    return phase 

def process_phase(dfrow, events, freqs, simulation_tag=None):
    
    '''
    Returns the phase time series to be analyzed for a session and set of behavioral events.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        freqs : numpy.array
            Frequencies at which to extract phase values.
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
            
    Returns
        phase : ptsa.data.TimeSeries
            Time series of phase values.
        mask : numpy.array
            List of boolean variables indicating whether the event was a successful memory event (True) or an unsuccessful memory event (False).
        sr : float
            Sample rate.
    '''
    
    eeg, mask = get_beh_eeg(dfrow, events, simulation_tag=simulation_tag)
    phase = get_phase(eeg, freqs)
    
    sr = float(eeg.samplerate)
    buffer_length = int(sr/1000*1000)
    phase = clip_buffer(phase, buffer_length)
    return phase, mask, sr

def get_power(eeg, freqs):
    
    '''
    Returns time series of spectral power values. Performs Morlet wavelet convolution, log-transforms power, clips buffer, and z-scores power values.
    
    Parameters:
        eeg : ptsa.data.TimeSeries
            EEG clip.
        freqs : numpy.array
            Wavelet frequencies at which to extract phase values.
            
    Returns
        power : ptsa.data.TimeSeries
            Time series of power values.
    '''
    
    wavelet_filter = MorletWaveletFilter(freqs=freqs, width=5, output='power', complete=True)
    power = wavelet_filter.filter(timeseries=eeg)
    power = power.transpose('event', 'channel', 'frequency', 'time')
    
    power = np.log10(power)
    
    sr = float(eeg.samplerate)
    buffer_length = int(sr/1000*1000)
    power = clip_buffer(power, buffer_length)
    
    mean = power.mean('time').mean('event')
    std = power.mean('time').std('event')
    power = (power-mean)/std 
    
    return power

def process_power(dfrow, events, freqs, simulation_tag=None):
    
    '''
    Returns the timebinned power time series to be analyzed for a session and set of behavioral events.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        events : pandas.DataFrame
            Behavioral events.
        freqs : numpy.array
            Frequencies at which to extract power values.
        simulation_tag : str
            Label of parameter set used to generate simulated EEG signal.
            
    Returns
        power : ptsa.data.TimeSeries
            Time series of power values.
        mask : numpy.array
            List of boolean variables indicating whether the event was a successful memory event (True) or an unsuccessful memory event (False).
    '''
    
    eeg, mask = get_beh_eeg(dfrow, events, simulation_tag=simulation_tag)
    sr = float(eeg.samplerate)
    power = get_power(eeg, freqs)
    
    power = timebin_power_timeseries(power.data, sr)
    
    return power, mask

def get_elsymx(dfrow, freqs, events, simulation_tag=None):
    
    '''
    Computes the electrode-by-electrode synchrony effects matrix.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        freqs : array_like
            Frequencies at which to extract phase values.
        events : pandas.DataFrame
            Behavioral events.
        simulation_tag : str 
            Label of parameter set used to generate simulated EEG signal.
                
    Returns:
        elsymx : numpy.array
            Electrode-by-electrode synchrony effects matrix.
    '''
    
    overlapping_pairs = find_overlapping_pairs(get_pairs(dfrow))
    
    phase, mask, sr = process_phase(dfrow, events, freqs, simulation_tag=simulation_tag) #get phase timeseries and successful/unsuccessful memory event mask

    electrode_count, freq_count, epoch_count = phase.shape[1], len(freqs), 5
    elsymx = np.full((electrode_count, electrode_count, freq_count, epoch_count, 2), np.nan)
        
    for iElec in np.arange(electrode_count):
        for jElec in np.arange(electrode_count):
            
            if (jElec > iElec) or ((iElec, jElec) in overlapping_pairs):
                continue #phase-locking is symmetric in signals
                
            diff = (phase.isel(channel = jElec) - phase.isel(channel = iElec)).data

            diff = timebin_phase_timeseries(diff, sr) #average phase differences in 200 ms timebins

            elsymx[iElec, jElec, ..., 0] = ppc(diff[mask, ...])
            elsymx[iElec, jElec, ..., 1] = ppc(diff[~mask, ...])
    
    elsymx = symmetrize(elsymx)
    return elsymx

def add_regions_elsymx(elsymx, regionalizations, freqs, beh):
    
    '''
    Labels the channel axes of the electrode-by-electrode synchrony effects matrix with their regions, and the frequency and time axes with frequency and time window labels.
    
    Parameters:
        elsymx : numpy.array
            Electrode-by-electrode synchrony effects matrix.
        regionalizations : array_like
            Region labels.
        freqs : array_like
            Frequency labels. 
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri')
    
    Returns:
        elsymx_regs : xarray.DataArray
            Electrode-by-electrode synchrony effects matrix with labeled axes.
    '''
    
    elsymx_regs = xr.DataArray(elsymx, 
                               dims=['reg1', 'reg2', 'freq', 'epoch', 'success'],
                               coords={'reg1': (['reg1'], regionalizations), 
                                       'reg2': (['reg2'], regionalizations),
                                       'freq': (['freq'], freqs), 
                                       'epoch': (['epoch'], beh_to_epochs[beh]),
                                       'success': (['success'], [True, False])})
    
    return elsymx_regs

def regionalize_electrode_connectivities(elsymx):
    
    '''
    Returns region-by-region matrix of synchrony values, generated from averaging the synchrony values of channels in the same region in the electrode-by-electrode synchrony effects matrix.
    
    Parameters:
        elsymx : xarray.DataArray
            Electrode-by-electrode synchrony effects matrix (labeled array with region labels).
    
    Returns: 
        regsymx : xarray.DataArray
            Region-by-region synchrony effects matrix.
    '''
    
    shape = np.array(elsymx.shape)
    region_labels = get_region_information('region_labels')
    shape[0] = shape[1] = len(region_labels)
    
    regsymx = np.full(shape, np.nan)
                               
    for iRegion, region1 in enumerate(region_labels):
        if region1 not in elsymx.reg1.values: continue
        
        for jRegion, region2 in enumerate(region_labels):
            if region2 not in elsymx.reg2.values: continue
            
            elsymx_sel = elsymx.sel(reg1 = region1, reg2 = region2)
            regsymx[iRegion, jRegion, ...] = elsymx_sel.mean([x for x in elsymx_sel.dims if x not in ['freq', 'epoch', 'success']]).values
        
    dims = list(elsymx.dims)
    coords = {}
    coords['reg1'] = coords['reg2'] = region_labels
    for dim in np.setdiff1d(dims, ['reg1', 'reg2']):
        coords[dim] = tuple(([dim], elsymx.coords[dim].values))
    regsymx = xr.DataArray(regsymx, 
                           dims=dims, 
                           coords=coords)

    return regsymx 

def run_pipeline(dfrow, band_name, beh, events, save_dir, simulation_tag=None):
    '''
    Runs the analysis pipeline for phase synchrony effects.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri').
        events : pandas.DataFrame
            Behavioral events to be analyzed.
        save_dir : str
            Directory to which analysis results should be saved.
        simulation_tag : str 
            Label of parameter set used to generate simulated EEG signal.
    
    Returns: 
        regsymx : xarray.DataArray
            Region-by-region synchrony effects matrix.
    '''
    os.makedirs(join(save_dir,'elsymxs',band_name), exist_ok=True)
    os.makedirs(join(save_dir,'regsymxs',band_name), exist_ok=True)

    out_el = join(save_dir,'elsymxs',band_name,f'{ftag(dfrow)}_elsymx.npy')
    out_rg = join(save_dir,'regsymxs',band_name,f'{ftag(dfrow)}_regsymx.pkl')

    if os.path.exists(out_el) and os.path.exists(out_rg):
        return

    freqs = {'theta': np.arange(3,9),
             'alpha': np.arange(8,13),
             'beta': np.arange(20,50,10),
             'gamma': np.arange(80,175,10)}[band_name]

    elsymx = get_elsymx(dfrow, freqs, events, simulation_tag=simulation_tag)
    np.save(out_el, elsymx)

    pairs = get_pairs(dfrow)
    localization = get_localization(dfrow)
    elsymx_regs = add_regions_elsymx(elsymx, regionalize_electrodes(pairs, localization), freqs, beh)
    regsymx = regionalize_electrode_connectivities(elsymx_regs)
    save_pickle(out_rg, regsymx)
    return regsymx
    
def cohens_d(x, y):
    
    '''
    Returns Cohen's d given two independent samples.
    
    Parameters:
        x (numpy.array): First sample.
        y (numpy.array): Second sample.
        
    Returns:
        d (float): Cohen's d.
    '''
    
    s = np.sqrt(((len(x)-1)*(np.std(x, axis=0, ddof=1)**2) + (len(y)-1)*(np.std(y, axis=0, ddof=1)**2))/(len(x)+len(y)-2))
    d = (np.mean(x, axis=0) - np.mean(y, axis=0))/s
    return d

def welchs_t(x, y): 
    
    '''
    Returns Welch's t-statistic for two independent samples.
    
    Parameters:
        x (numpy.array): First sample.
        y (numpy.array): Second sample.
        
    Returns:
        (float): Welch's t-statistic.
    '''
    
    return scipy.stats.ttest_ind(x, y, axis=0, equal_var=False).statistic

def comp_elpomx(dfrow, freqs, events, simulation_tag=None):
    
    '''
    Computes the electrode-wise power effects matrix.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        freqs : array_like
            Frequencies at which to extract power values.
        events : pandas.DataFrame
            Behavioral events.
        simulation_tag : str 
            Label of parameter set used to generate simulated EEG signal.
                
    Returns:
        elpomx : pandas.Series
            Electrode-wise power effects matrices. elpomx['t'] contains a numpy.array with the Welch's t-statistics and elpomx['d'] contains a numpy.array with the Cohen's d statistics.
    '''
    
    power, mask = process_power(dfrow, events, freqs, simulation_tag=simulation_tag)
    
    elpomx = pd.Series({})
    elpomx['t'] = welchs_t(power[mask, ...], power[~mask, ...])
    elpomx['d'] = cohens_d(power[mask, ...], power[~mask, ...])
    
    return elpomx

def add_regions_elpomx(elpomx, regionalizations, freqs, beh):
    
    '''
    Labels the channel axis of the electrode-wise power effects matrix with region labels, and the frequency and time axes with frequency and time window labels.
    
    Parameters:
        elpomx : numpy.array
            Electrode-wise power effects matrix.
        regionalizations : array_like
            Region labels.
        freqs : array_like
            Frequency labels.
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri')
    
    Returns:
        elpomx_regs : pandas.Series
            Electrode-wise power effects matrices with labeled axes. elpomx_regs['t'] contains an xarray.DataArray with the Welch's t-statistics and elpomx_regs['d'] contains an xarray.DataArray with the Cohen's d statistics.
    '''
    
    elpomx_regs = pd.Series({})
    elpomx_regs = xr.DataArray(elpomx, 
                               dims=['reg1', 'freq', 'epoch'], 
                               coords={'reg1': (['reg1'], regionalizations), 
                                       'freq': (['freq'], freqs),
                                       'epoch': (['epoch'], beh_to_epochs[beh])})
    
    return elpomx_regs

def regionalize_electrode_powers(elpomx):
    
    '''
    Returns region-wise matrix of power effect values, generated from averaging the power effect values of channels in the same region in the electrode-wise power effect matrix.
    
    Parameters:
        elpomx : xarray.DataArray
            Electrode-wise power effects matrix (labeled array with region labels).
    
    Returns: 
        regpomx : xarray.DataArray
            Region-wise matrix of power effect values.
    '''
    
    shape = np.array(elpomx.shape)
    region_labels = get_region_information('region_labels')
    shape[0] = len(region_labels)
    
    regpomx = np.full(shape, np.nan)
                               
    for iRegion, region1 in enumerate(region_labels):
        if region1 not in elpomx.reg1.values: continue
            
        elpomx_sel = elpomx.sel(reg1 = region1)
        regpomx[iRegion, ...] = elpomx_sel.mean([x for x in elpomx_sel.dims if x not in ['freq', 'epoch', 'success']]).values
        
    dims = list(elpomx.dims)
    coords = {}
    coords['reg1'] = region_labels
    for dim in np.setdiff1d(dims, 'reg1'):
        coords[dim] = tuple(([dim], elpomx.coords[dim].values))
    regpomx = xr.DataArray(regpomx, 
                           dims=dims, 
                           coords=coords)
            
    return regpomx 

def run_pipeline_power(dfrow, band_name, beh, events, save_dir, simulation_tag=None):
    
    '''
    Runs the analysis pipeline for spectral power effects.
    
    Parameters:
        dfrow : pandas.Series
            Session label.
        band_name : str
            Frequency band to be analyzed ('theta' or 'gamma').
        beh : str
            Behavioral contrast label ('en', 'en_all', 'rm', or 'ri').
        events : pandas.DataFrame
            Behavioral events to be analyzed.
        save_dir : str
            Directory to which analysis results should be saved.
        simulation_tag : str 
            Label of parameter set used to generate simulated EEG signal.
    
    Returns: 
        None
    '''
    
    if ex(join(save_dir, 'regpomxs', band_name, f'{ftag(dfrow)}_regpomx.pkl')): return

    band_name_to_freqs = {'theta': np.arange(3, 9),
                          'alpha': np.arange(8, 13),
                          'beta': np.arange(20, 50, 10),
                          'gamma': np.arange(80, 175, 10)}
    freqs = band_name_to_freqs[band_name]

    elpomx = comp_elpomx(dfrow, freqs, events, simulation_tag=simulation_tag)

    pairs = get_pairs(dfrow)
    localization = get_localization(dfrow)
    regionalizations = regionalize_electrodes(pairs, localization)
    
    regpomx = pd.Series({})
    for k in ['t', 'd']: 
        elpomx_regs = add_regions_elpomx(elpomx[k], regionalizations, freqs, beh)
        regpomx[k] = regionalize_electrode_powers(elpomx_regs)

        np.save(join(save_dir, 'elpomxs', band_name, f'{ftag(dfrow)}_elpomx{k}.npy'), elpomx[k])
        save_pickle(join(save_dir, 'regpomxs', band_name, f'{ftag(dfrow)}_regpomx{k}.pkl'), regpomx[k])

def replace_w_simulated_EEG(original_eeg,
                            dfrow,
                            condition_mask,
                            simulation_tag=None,
                            eeg_data_source='cmlreaders',
                            time_unit='millisecond',
                            random_state=None,
                            random_state_type='offset_from_eeg_hash',
                            verbose=False):
    assert isinstance(original_eeg, TimeSeries)
    assert simulation_tag in AVAILABLE_SIMULATIONS
    
    if random_state_type == 'offset_from_eeg_hash':
        # fix random state to hash of original EEG
        # ensures unique, reproducible random states for each unique input EEG recording
        eeg_hash = hash(str(original_eeg.data))
        random_state = eeg_hash if random_state is None else eeg_hash + random_state
        random_state %= 2**32 - 1
    elif random_state_type == 'standard':
        pass
    else:
        raise ValueError
    if random_state is not None:
        np.random.seed(random_state)
    
    if simulation_tag in ['standard', '', None]:
        return original_eeg
    eeg = original_eeg.copy()
    
    parameters = simulation_parameters[simulation_tag]
    
    wavelet_amplitude = parameters['wavelet_amplitude']
    get_phase_covariance = parameters['phase_covariance_function']
    if get_phase_covariance == 'within_region_group':
        pairs = get_pairs(dfrow)
        if eeg_data_source == 'ptsa':
            # confirm that EEG channels match pairs dataframe used for localizations
            contact_numbers = [[int(pair.item()[0].decode('utf-8')),
                                int(pair.item()[1].decode('utf-8')),
                                pair]
                               for pair in eeg.channel]
            contact_numbers = pd.DataFrame(contact_numbers, columns=['contact_1', 'contact_2', 'eeg_pair'])
            merge_columns = ['contact_1', 'contact_2']
            pairs = pairs.merge(contact_numbers[merge_columns], on=merge_columns)
            assert len(pairs) == len(contact_numbers)
        elif eeg_data_source != 'cmlreaders':
            raise ValueError
        
        localization = get_localization(dfrow)
        regionalizations = regionalize_electrodes(pairs, localization)
        region_series = pd.Series(regionalizations)
        has_hemisphere_mask = region_series.str.startswith('L ') | region_series.str.startswith('R ') | region_series.isna()
        if not has_hemisphere_mask.all():
            # print(region_series[~has_hemisphere_mask])
            # display(region_series)
            raise ValueError
        # use hemispheres for simple regional grouping (put rare NaN region channels in 'Right' group)
        region_groups = ['Left' if left else 'Right' for left in pd.Series(regionalizations).str.startswith('L ')]
        
        from simulate_eeg import get_block_diagonal_ppc_matrix, ppc_matrix_to_wrapped_normal_covariance

        ppc_matrix0 = get_block_diagonal_ppc_matrix(n_channels=None,
                                                    n_regions=None,
                                                    n_region_groups=None,
                                                    regions=list(regionalizations),
                                                    region_groups=list(region_groups),
                                                    global_ppc=parameters['global_ppc0'],
                                                    within_group_ppc=parameters['within_group_ppc0'],
                                                    within_region_ppc=parameters['within_region_ppc0'],
                                                    verbose=verbose,
                                                   )
        cov0 = ppc_matrix_to_wrapped_normal_covariance(ppc_matrix0)
        
        ppc_matrix1 = get_block_diagonal_ppc_matrix(n_channels=None,
                                                    n_regions=None,
                                                    n_region_groups=None,
                                                    regions=list(regionalizations),
                                                    region_groups=list(region_groups),
                                                    global_ppc=parameters['global_ppc1'],
                                                    within_group_ppc=parameters['within_group_ppc1'],
                                                    within_region_ppc=parameters['within_region_ppc1'],
                                                    verbose=verbose,
                                                   )
        cov1 = ppc_matrix_to_wrapped_normal_covariance(ppc_matrix1)
        
    elif get_phase_covariance is None:
        cov0 = None
        cov1 = None
    else:
        raise NotImplementedError(f'Phase covariance method {get_phase_covariance} is not implemented!')
    oscillation_frequency = parameters['oscillation_frequency']
    morlet_reps = parameters['morlet_reps']
    
    pinknoise_amplitude = parameters['pinknoise_amplitude']
    pinknoise_exponent = parameters['pinknoise_exponent']
    
    start_time_ms = original_eeg.time.min()
    duration_ms = original_eeg.time.max() - start_time_ms
    if time_unit == 'second':
        start_time_ms *= 1000
        duration_ms *= 1000
    
    eeg = eeg.assign_coords(_index=("event", np.arange(len(eeg['event'])))).set_index(event='_index', append=True)
    eeg0 = eeg[~condition_mask]
    eeg1 = eeg[condition_mask]
    
    simulated_eeg0 = sample_eeg(n_events=len(eeg0.event),
                                n_channels=len(eeg0.channel),
                                sample_rate_Hz=eeg0.samplerate,
                                start_time_ms=start_time_ms,
                                duration_ms=duration_ms,
                                connectivity_frequency_Hz=oscillation_frequency,
                                morlet_reps=morlet_reps,
                                wavelet_amplitude=wavelet_amplitude,
                                phase_mean=np.zeros(len(cov0)) if cov0 is not None else None,
                                phase_covariance=cov0,
                                pinknoise_amplitude=pinknoise_amplitude,
                                pinknoise_exponent=pinknoise_exponent,
    )
    
    simulated_eeg1 = sample_eeg(n_events=len(eeg1.event),
                                n_channels=len(eeg1.channel),
                                sample_rate_Hz=eeg1.samplerate,
                                start_time_ms=start_time_ms,
                                duration_ms=duration_ms,
                                connectivity_frequency_Hz=oscillation_frequency,
                                morlet_reps=morlet_reps,
                                wavelet_amplitude=wavelet_amplitude,
                                phase_mean=np.zeros(len(cov1)) if cov1 is not None else None,
                                phase_covariance=cov1,
                                pinknoise_amplitude=pinknoise_amplitude,
                                pinknoise_exponent=pinknoise_exponent,
    )
    simulated_eeg0 = TimeSeries.create(data=simulated_eeg0.values,
                                       coords={'event': eeg0.coords['event'],
                                               'channel': eeg0.coords['channel'],
                                               'time': simulated_eeg0.coords['time']},
                                       dims=simulated_eeg0.dims,
                                       samplerate=simulated_eeg0.samplerate.item())
    simulated_eeg1 = TimeSeries.create(data=simulated_eeg1.values,
                                       coords={'event': eeg1.coords['event'],
                                               'channel': eeg1.coords['channel'],
                                               'time': simulated_eeg1.coords['time']},
                                       dims=simulated_eeg1.dims,
                                       samplerate=simulated_eeg1.samplerate.item())
    
    del eeg0, eeg1
    from matrix_operations import sort_multi_index_coord
    if not np.all(simulated_eeg0.time == simulated_eeg1.time):
        raise ValueError('Time values in simulated EEG do not match across conditions')
    
    simulated_eeg = xr.concat([simulated_eeg0, simulated_eeg1], 'event')
    # sort back into original event order
    simulated_eeg = sort_multi_index_coord(simulated_eeg, 'event', '_index')
    simulated_eeg = simulated_eeg.reset_index('_index', drop=True)
    simulated_eeg.attrs['samplerate'] = original_eeg.samplerate
    
    # n_evs = 0
    # for i_ev, (sim_ev, orig_ev) in enumerate(zip(simulated_eeg.event, original_eeg.event)):
    #     if not sim_ev.equals(orig_ev):
    #         print(i_ev)
    #         print(sim_ev)
    #         print()
    #         print(orig_ev)
    #         print()
    #         print('sim_ev == sim_ev', sim_ev.equals(sim_ev))
    #         print('orig_ev == orig_ev', orig_ev.equals(orig_ev))
    #         print()
    #         print()
    #         n_evs += 1
    #         if n_evs == 5:
    #             break
    
    # print(original_eeg.event)
    # print(simulated_eeg.event)
    # print(original_eeg.item_name)
    # print(simulated_eeg.item_name)
    
    # matches for most sessions, but some get implicitly type cast by xarray in workshop_311 environment
    # assert original_eeg.event.equals(simulated_eeg.event)
    assert original_eeg.channel.equals(simulated_eeg.channel)
    assert original_eeg.samplerate.equals(simulated_eeg.samplerate)
    if 'samplerate' in original_eeg.attrs:
        simulated_eeg.attrs['samplerate'] = simulated_eeg.samplerate
    
    if time_unit == 'second':
        simulated_eeg = simulated_eeg.assign_coords({'time': simulated_eeg['time'] / 1000})

    # attributes match for EEG loaded with cmlreaders but EEG loaded with PTSA has different attributes that appear to not matter
    # assert original_eeg.attrs == simulated_eeg.attrs, f'Attributes of simulated EEG do not match original. '
    #         f'Original attributes:\n{original_eeg.attrs}\n\nReplacement attributes:\n{simulated_eeg.attrs}'
    return simulated_eeg


from matrix_operations import sort_array_across_order, upper_tri_values
from misc import print_ttest_1samp
from matplotlib.colors import SymLogNorm

def plot_ppc_matrix(ppc_matrix, beh, regions=None, subplot_col_num=None, linthresh=None):
    if subplot_col_num is None:
        plt.figure(figsize=(4, 2))
        subplot_col_num = 1
    plt.subplot(2, 3, subplot_col_num)
    if linthresh is not None:
        color_norm = SymLogNorm(linthresh=linthresh,
                                vmin=np.nanpercentile(ppc_matrix, 5),
                                vmax=np.nanpercentile(ppc_matrix, 95)
                                # vmin=np.nanmin(ppc_matrix),
                                # vmax=np.nanmax(ppc_matrix)
                               )
    else:
        color_norm = None
    plt.imshow(ppc_matrix, norm=color_norm)
    plt.xlabel('Region', fontsize=30)
    plt.ylabel('Region', fontsize=30)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar(shrink=0.75)
    cbar.set_label('PPC Difference', fontsize=30)
    cbar.ax.tick_params(labelsize=20)
    behavioral_name = behavioral_names[beh]
    plt.title(f'{behavioral_name}', fontsize=35)
    
    plt.subplot(2, 3, subplot_col_num + 3)
    plt.hist(ppc_matrix.values.reshape(-1), bins=100)
    plt.xticks(fontsize=25, rotation=45)
    plt.yticks(fontsize=25)
    plt.xlabel('PPC Difference', fontsize=30)
    plt.ylabel('Region-Region Combination Count', fontsize=30)
    plt.title(f'{behavioral_name}', fontsize=35)
    plt.suptitle(f'Regional PPC Differences', fontsize=40)

    if regions:
        ppc_matrix_sorted = sort_array_across_order(ppc_matrix, regions, axis=[0, 1])
        plt.figure(figsize=(2, 2))
        plt.imshow(ppc_matrix_sorted)
        cbar = plt.colorbar()
        cbar.set_label('PPC Difference', fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        plt.title(behavioral_name, fontsize=30)
        plt.suptitle('PPC Matrix Sorted by Region', fontsize=35)


def generate_subject_synchrony_results(root_dir, simulation_tag=None, figure_path=''):
    subplot_col_num = 1
    is_simulation = simulation_tag not in ['', 'standard', None]
    plt.figure(1, figsize=(30, 20))
    for i_beh, beh in enumerate(['en', 'rm', 'ri']):
        fname = join(root_dir, f'{beh}_pop_symx.pkl')
        mx = load_pickle(fname)

        mx0 = mx.sel(success=False)
        mx1 = mx.sel(success=True)
        diffs = (mx1 - mx0).mean(['freq', 'epoch', 'sub'])
        mx0_sub = mx0.mean(['freq', 'epoch', 'reg1', 'reg2'])
        mx1_sub = mx1.mean(['freq', 'epoch', 'reg1', 'reg2'])
        sub_diffs = mx1_sub - mx0_sub

        regions = None
        behavioral_name = behavioral_names[beh]
        print(f'---------------- {behavioral_name} ----------------')
        print(f'Averaged across frequencies:')
        print(f'Mean {behavioral_name} PPC (averaged across subjects before regions): {np.nanmean(diffs.values):0.5}')
        print(f'Mean {behavioral_name} PPC (averaged across regions before subjects): {np.nanmean(sub_diffs.values):0.5}')
        print_ttest_1samp(sub_diffs.values)
        plt.figure(2, figsize=(30, 10))
        plt.subplot(1, 3, i_beh + 1)
        plt.hist(sub_diffs.values, bins=40)
        plt.title(f'{behavioral_name}', fontsize=35)
        plt.xlabel('PPC Difference', fontsize=30)
        plt.ylabel('Subject Count', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.suptitle(f'Subject-level Brain-wide Synchrony', fontsize=40)

        if is_simulation:
            params = simulation_parameters[simulation_tag]
            linthresh = None
            
            # deprecated. SymLogNorm did not substantially improve interpretability
            # if simulation_tag in NULL_SIMULATION_TAGS:
            #     linthresh = None
            # else:
            #     # linear threshold for matplotlib.colors.SymLogNorm color normalization
            #     linthresh = np.min([np.abs(params['global_ppc1'] - params['global_ppc0']),
            #                         np.abs(params['within_group_ppc1'] - params['within_group_ppc0']),
            #                         np.abs(params['within_region_ppc1'] - params['within_region_ppc0']),
            #                        ])
            #     linthresh = np.max([linthresh, 0.0005])
            oscillation_frequency = params['oscillation_frequency']
            # plot_ppc_matrix(mx0, regions=regions, beh=beh, subplot_col_num=subplot_col_num)
            # plt.title(f'PPC matrix for unsuccessful {beh} trials at {oscillation_frequency} Hz')
            # plot_ppc_matrix(mx1, regions=regions, beh=beh, subplot_col_num=subplot_col_num)
            # plt.title(f'PPC matrix for successful {beh} trials at {oscillation_frequency} Hz')
            mx0_freq = mx.sel(freq=oscillation_frequency, success=False)
            mx1_freq = mx.sel(freq=oscillation_frequency, success=True)
            diffs_freq = (mx1_freq - mx0_freq).mean(['epoch', 'sub'])
            mx0_sub_freq = mx0_freq.mean(['epoch', 'reg1', 'reg2'])
            mx1_sub_freq = mx1_freq.mean(['epoch', 'reg1', 'reg2'])
            sub_diffs_freq = mx1_sub_freq - mx0_sub_freq

            print(f'Measured at {oscillation_frequency} Hz:')
            print(f'Mean {behavioral_name} PPC (averaged across subjects before regions): {np.nanmean(diffs_freq.values):0.5}')
            print(f'Mean {behavioral_name} PPC (averaged across regions before subjects): {np.nanmean(sub_diffs_freq.values):0.5}')
            print_ttest_1samp(sub_diffs_freq.values)
            print()
            plt.figure(3, figsize=(30, 10))
            plt.subplot(1, 3, i_beh + 1)
            plt.hist(sub_diffs_freq.values, bins=40)
            plt.title(f'{behavioral_name}', fontsize=25)
            plt.xlabel('PPC Difference', fontsize=25)
            plt.ylabel('Subject Count', fontsize=25)
            plt.suptitle(f'Subject-level Brain-wide Synchrony at {oscillation_frequency} Hz', fontsize=25)
        else:
            linthresh = None

        plt.figure(1)
        plot_ppc_matrix(diffs, regions=regions, beh=beh, subplot_col_num=subplot_col_num, linthresh=linthresh)
        subplot_col_num += 1

    title = 'PPC Matrices for Oscillations'
    if is_simulation:
        title = title + f' at {oscillation_frequency} Hz'
    plt.suptitle(title, fontsize=35, y=0.9)
    plt.savefig(os.path.join(figure_path, f'ppc_matrices' + (f'_{oscillation_frequency}Hz_{simulation_tag}' if is_simulation else '') + '.png'))
    
    # # find regions missing across all subjects
    # missing_region_mask = np.isnan(mx0.values).all(axis=(0, 2, 3, 4))
    # missing_regions = mx0.reg1[missing_region_mask]
    # print(missing_regions)


    
def generate_simulation_theoretical_effect_plots(root_dir, simulation_tag, figure_path=''):
    parameters = simulation_parameters[simulation_tag]
    is_simulation = simulation_tag not in ['', 'standard', None]
    oscillation_frequency = parameters['oscillation_frequency']
    if simulation_tag in NULL_SIMULATION_TAGS:
        n_regions = 80
        ppc_matrix = np.zeros((n_regions, n_regions))
    else:
        from simulate_eeg import get_block_diagonal_ppc_matrix

        verbose = False
        ppc_matrix0 = get_block_diagonal_ppc_matrix(n_channels=80,
                                                    n_regions=80,
                                                    n_region_groups=2,
                                                    regions=None,
                                                    region_groups=None,
                                                    global_ppc=parameters['global_ppc0'],
                                                    within_group_ppc=parameters['within_group_ppc0'],
                                                    within_region_ppc=parameters['within_region_ppc0'],
                                                    verbose=verbose,
                                                   )
        ppc_matrix1 = get_block_diagonal_ppc_matrix(n_channels=80,
                                                    n_regions=80,
                                                    n_region_groups=2,
                                                    regions=None,
                                                    region_groups=None,
                                                    global_ppc=parameters['global_ppc1'],
                                                    within_group_ppc=parameters['within_group_ppc1'],
                                                    within_region_ppc=parameters['within_region_ppc1'],
                                                    verbose=verbose,
                                                   )

        ppc_matrix = ppc_matrix1 - ppc_matrix0
        np.fill_diagonal(ppc_matrix, parameters['within_region_ppc1'] - parameters['within_region_ppc0'])

    from matrix_operations import upper_tri_values
    print('Average theoretical PPC difference across regions:', upper_tri_values(ppc_matrix).mean())

    plt.figure(figsize=(10, 10))
    plt.imshow(ppc_matrix)
    plt.xlabel('Region', fontsize=30)
    plt.ylabel('Region', fontsize=30)
    plt.xticks([])
    plt.yticks([])
    cbar = plt.colorbar()
    cbar.set_label('PPC Difference', fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    _ = plt.title(f'Theoretical PPC Differences', fontsize=35)
    plt.savefig(os.path.join(figure_path, f'theoretical_ppc_matrix' + (f'_{oscillation_frequency}Hz_{simulation_tag}' if is_simulation else '') + '.png'))
    return ppc_matrix


def sanity_check_whole_brain_synchrony(theory_ppc_matrix, root_dir, simulation_tag):
    from matrix_operations import upper_tri_values
    ppc_matrix = theory_ppc_matrix
    
    is_simulation = simulation_tag not in ['', 'standard', None]
    
    theoretical_effects = dict()

    for beh in ['en', 'rm', 'ri']:
        fname = join(root_dir, f'{beh}_pop_symx.pkl')
        mx = load_pickle(fname)
        
        behavioral_name = behavioral_names[beh]

        mean_nans = list()
        sub_population_matrices = list()
        subject_effects_nonredundant = list()
        for subject, sub_mx in mx.groupby('sub'):
            # replace synchrony values for missing region-region pairs with NaNs in population PPC matrix
            nan_idx = np.where(np.isnan(sub_mx[..., 0, 0, 0].values))
            ppc_matrix_sub = ppc_matrix.copy()
            ppc_matrix_sub[nan_idx] = np.nan
            sub_population_matrices.append(ppc_matrix_sub)
            mean_nans.append(np.isnan(ppc_matrix_sub).mean())
            subject_effects_nonredundant.append(np.nanmean(upper_tri_values(ppc_matrix_sub)))
        population_ppcs = np.stack(sub_population_matrices, axis=0)
        # subjects x regions x regions
        theoretical_global_synchrony = np.nanmean(np.nanmean(np.nanmean(population_ppcs, axis=-1), axis=-1), axis=0)
        theoretical_effects[beh] = theoretical_global_synchrony
        print(f'---------------- {behavioral_name} ----------------')
        print(f'Theoretical global synchrony effect: {theoretical_global_synchrony:0.5}')
        print(f'Theoretical global synchrony effect with non-redundant region-region combinations: {np.mean(subject_effects_nonredundant):0.5}')
        proportion_nan = np.mean(mean_nans)
        print(f'Proportion of total region-region pairs missing at subject level: {proportion_nan:0.5}')
        print()
        
    return theoretical_effects


def plot_synchrony_frequency_epochs(root_dir, simulation_tag=None, figure_path=''):
    plt.figure(figsize=(30, 10))
    is_simulation = simulation_tag not in ['', 'standard', None]
    if is_simulation:
        parameters = simulation_parameters[simulation_tag]
        oscillation_frequency = parameters['oscillation_frequency']

    for i_beh, beh in enumerate(['en', 'rm', 'ri']):
        behavioral_name = behavioral_names[beh]

        fname = join(root_dir, f'{beh}_pop_symx.pkl')
        mx = load_pickle(fname)
        freq_epoch_mx = mx[0, 0, 0, ..., 0] * np.nan
        for frequency, freq_mx in mx.groupby('freq'):
            for epoch, subset_mx in freq_mx.groupby('epoch'):
                subset_mx = subset_mx.mean('reg2').mean('reg1')
                mean_effect = (subset_mx.sel(success=True).data - subset_mx.sel(success=False).data).mean()
                freq_epoch_mx.loc[dict(freq=frequency, epoch=epoch)] = mean_effect
        plt.subplot(1, 3, i_beh + 1)
        plt.imshow(freq_epoch_mx)
        plt.suptitle(f'Whole-brain Synchrony by Frequency and Epoch', fontsize=40)
        plt.title(f'{behavioral_name}', fontsize=35)
        plt.xticks(np.arange(freq_epoch_mx.shape[1]), freq_epoch_mx.epoch.values, fontsize=25)
        plt.yticks(np.arange(freq_epoch_mx.shape[0]), freq_epoch_mx.freq.values, fontsize=25)
        plt.xlabel('Epoch Start Time (ms)', fontsize=25)
        plt.ylabel('Frequency (Hz)', fontsize=25)
        cbar = plt.colorbar()
        cbar.set_label('PPC Difference', fontsize=20)
        cbar.ax.tick_params(labelsize=20)

    plt.savefig(os.path.join(figure_path, f'ppc_freq_epoch' + (f'_{oscillation_frequency}Hz_oscillation_{simulation_tag}' if is_simulation else '') + '.png'))


# def sanity_check_simulated_synchrony(parameters):
#     diagonal_diff = parameters['within_region_ppc1'] - parameters['within_region_ppc0']
#     offdiagonal_diff = parameters['global_ppc1'] - parameters['global_ppc0']

#     # average proportion of region-region combinations missing across subjects
#     proportion_region_combo_missing = 0.95
#     n_regions = 80 * np.sqrt(1 - proportion_region_combo_missing)
#     n_upper_tri_strict = (n_regions ** 2 - n_regions) / 2
#     n_upper_tri = n_upper_tri_strict + n_regions
#     pop_global_synchrony_upper_tri = (diagonal_diff * n_regions + offdiagonal_diff * n_upper_tri_strict) / n_upper_tri
#     pop_global_synchrony_all_regions = (diagonal_diff * n_regions + offdiagonal_diff * n_upper_tri_strict * 2) / (n_regions ** 2)
#     print(f'Population global synchrony averaged over region-region pairs including redundant off-diagonal elements:\n{pop_global_synchrony_all_regions:0.5}')
#     print(f'Population global synchrony averaged over region-region pairs NOT including redundant off-diagonal elements:\n{pop_global_synchrony_upper_tri:0.5}')

def get_analyzed_channels_mask(dfrow):
    
    pairs, localization = get_pairs(dfrow), get_localization(dfrow)
    regionalizations = regionalize_electrodes(pairs, localization)
    region_labels = get_region_information('region_labels')
    standard_regionalizations_mask = np.isin(regionalizations, region_labels)
    
    return standard_regionalizations_mask

def drop_channels_from_symx(symx, included_channel_mask):
    
    symx = symx[included_channel_mask, ...]
    symx = symx[:, included_channel_mask, ...]
    
    return symx