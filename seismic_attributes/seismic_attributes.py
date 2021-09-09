#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: seismic_attributes.py version 1.0.3
#  Purpose: An ObsPy library for event detection and seismic attribute
#           calculation: preparing waveforms for automated analysis.
#   Author: Ross Turner
#    Email: turner.rj@icloud.com
#
# Copyright (C) 2020-2021 Ross Turner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
# -------------------------------------------------------------------
"""
An ObsPy library for event detection and seismic attribute calculation: preparing waveforms for automated analysis.

:copyright:
    Ross Turner, 2020-2021
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)
"""

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.dates as mdates
import datetime, os, obspy, pytz, warnings
import seaborn as sns
from functools import partial
from math import factorial
from matplotlib import cm, rc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from multiprocessing import cpu_count, Pool
from numpy.fft import rfft, rfftfreq
from obspy import read, read_inventory, Trace, Stream, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy.signal.trigger import trigger_onset
from obspy.signal.cross_correlation import correlate
from obspy.signal.polarization import flinn
from obspy.signal.filter import envelope

# define constants
__chunklength_in_sec = 86400

# define functions to download or find already downloaded waveforms
def get_waveforms(network, station, location, channel, starttime, endtime, event_buffer=3600, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
    """
    Function to download waveform data from an online seismic repository and save to a local directory or external drive. The waveform data are split into files containing a single day of data to enable fast recall and storage across multiple drives. The function checks if data for the requested station, channels (components) and time period are present in the specified local directory before attempting to download new waveform data. The requested waveform data are output as a single obspy Stream object.
        
    Parameters
    ----------
    network : str
        A single SEED or data center defined network code; wildcards are not allowed.
    station : str
        A single SEED station code; wildcards are not allowed.
    location : str
        A single SEED location identifier (often blank or ’0’); wildcards are not allowed.
    channel : str or list
        One or more SEED channel codes; multiple codes are entered as a list; e.g. [’BHZ’, ’HHZ’]; wildcards are not allowed.
    starttime : UTCDateTime
        Start time series on the specified start time (or after that time in the case of a data gap).
    endtime : UTCDateTime
        End time series (one sample) before the specified end time.
    event_buffer : float, optional
        Minimum duration of data to buffer before and after the requested time period; expected units are seconds. This is used to capture the full length of events that extend beyond the time period. The default value is 3600 seconds.
    waveform_name : str or path, optional
        Path to directory to read (check for) and write waveform data (an existing file of the same name will not be overwritten). The default location is a directory named waveforms in the working directory.
    station_name : str or path, optional
        Path to directory to read (check for) and write station data (location coordinates, elevation etc, as provided by data repository). The default location is a directory named stations in the working directory.
    providers : str or list, optional
        One or more clients to use to download the requested waveform data if it does not already exist in the specified local directory. Multiple clients are entered as a list; e.g. [’IRIS’, ’GFZ’]. By default IRIS, LMU and GFZ are queried.
    user : str, optional
        User name of HTTP Digest Authentication for access to restricted data.
    password : str, optional
        Password of HTTP Digest Authentication for access to restricted data.
    download : bool, optional
        Specify whether days with missing waveform data are to be downloaded from client; e.g. True or False, alternatively 1 or 0. Missing data are downloaded by default.
        
    Returns
    -------
    stream : Stream
        A stream object with one or more traces for each component at the requested seismometer.
    """
    # create empty stream to store amplitude of three component waveform
    stream = Stream()

    # read-in waveform data from downloaded files
    if isinstance(channel, list):
        for i in range(0, len(channel)):
            if i == 0:
                stream = __get_waveforms(network, station, location, channel[0], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
            else:
                stream += __get_waveforms(network, station, location, channel[i], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
    else:
        stream = __get_waveforms(network, station, location, channel, starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)

    # merge different days in the stream at the same seismograph and channel; this prevents masked arrays from being created
    stream.merge(method=0, fill_value='interpolate')
    # sort channels into ZNE order for use in some obspy functions (only used for some attributes)
    stream.sort(keys=['network', 'station', 'location', 'channel'], reverse=True)
    # truncate stream at requested time window with buffer either side
    stream = stream.slice(starttime - event_buffer, endtime + event_buffer)
    
    # check all traces are the same length and start at the same time
    for i in range(0, len(stream)):
        if not (stream[i].stats.starttime == stream[0].stats.starttime and len(stream[i].data) == len(stream[0].data)):
            warnings.filterwarnings('always', category=UserWarning)
            warnings.warn('Stream has one or more components with inconsistent start and end times! Download the data again if it exists or select a valid time period.', category=UserWarning)
            warnings.filterwarnings('ignore', category=Warning)

    print(stream)
    return stream

def __get_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
    """
    Private function for get_waveforms() to read existing waveform data on the user computer or download missing waveform data from a client.
    """
    # create empty stream to store waveform
    stream = Stream()

    # set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = os.path.join(os.path.join(os.getcwd(), waveform_name), network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed')

        # if file exists add to stream
        if os.path.isfile(filename):
            stream += read(filename)
        # otherwise attempt to download file then read-in if data exists
        else:
            if download == True:
                __download_waveforms(network, station, location, channel, start_time, end_time, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password)
                if os.path.isfile(filename):
                    stream += read(filename)
            else:
                # issue warning that file is not available in local directory
                warnings.filterwarnings('always', category=UserWarning)
                warnings.warn(filename+' not available in local directory.', category=UserWarning)
                warnings.filterwarnings('ignore', category=Warning)

        # update start and end time of each file
        start_time += __chunklength_in_sec
        end_time += __chunklength_in_sec
            
    return stream

def __download_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None):
    """
    Private function for get_waveforms() to download missing waveform data from a client.
    """
    # specify rectangular domain containing any location in the world.
    domain = RectangularDomain(minlatitude=-90, maxlatitude=90, minlongitude=-180, maxlongitude=180)

    # apply restrictions on start/end times, chunk length, station name, and minimum station separation
    restrictions = Restrictions(
        starttime=t1,
        endtime=t2,
        chunklength_in_sec=__chunklength_in_sec,
        network=network, station=station, location=location, channel=channel,
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0)

    # download requested waveform and station data to specified locations
    if isinstance(providers, list):
        if (not user == None) and (not password == None):
            client = []
            for provider in providers:
                client.append(Client(provider, user=user, password=password))
            mdl = MassDownloader(providers=client)
        else:
            mdl = MassDownloader(providers=providers)
    else:
        if (not user == None) and (not password == None):
            mdl = MassDownloader(providers=[Client(providers, user=user, password=password)])
        else:
            mdl = MassDownloader(providers=[providers])
    mdl.download(domain, restrictions, mseed_storage=waveform_name, stationxml_storage=os.path.join(station_name, '{network}.{station}.'+location+'.'+channel+'.xml'))


# define functions to find events in a single seismometer or coincident events across multiple seismometers based on the quadrature sum of their components
def get_events(stream, starttime, endtime, signal_type='amplitude', station_name='stations', trigger_type='recstalta', avg_wave_speed=2, thr_event_join=0.5, thr_coincidence_sum=-1, thr_on=5, thr_off=1, **options): # avg_wave_speed in km/s
    """
    Function to detect events from seismic waveform data using STA/LTA-type algorithms. The input waveforms are denoted by seismometer and channel (e.g. ‘BH?’, ‘HH?’, ‘LH?’); the signal from seismometers with multiple components (i.e. ‘Z’, ‘N’, ‘E’) are combined into a single waveform using the Euclidean norm. The triggering algorithm is applied to the resulting amplitude or energy waveform. Small gaps between triggered events are removed before the combined event details are written to the (reference) event catalogue. If multiple seismometers are present the user can specify the minimum number of seismometers on which an event must be detected for that event to be included in the catalogue.
    
    Parameters
    ----------
    stream : Stream
        Stream containing waveform data for each component from one or more seismometers.
    starttime : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap).
    endtime : UTCDateTime
        Limit results to time series ending (one sample) before the specified end time.
    signal_type : str, optional
        Apply the event detection algorithm to the ‘amplitude’ (i.e. absolute value) or ‘energy’ (i.e. amplitude-squared) waveform. The event detection algorithm is applied to the amplitude waveform by default.
    station_name : str or path, optional
        Path to directory to read station data, specifically the GPS coordinates. The default location is a directory named stations in the working directory.
    trigger_type : str, optional
        The trigger algorithm to be applied (e.g. ‘recstalta’). See e.g. obspy.core.trace.Trace.trigger() for further details. The recursive STA/LTA algorithm is applied by default.
    avg_wave_speed : float, optional
        The speed at which seismic waves propagate through the local medium, contributing to a delay between detections in elements of a seismic array. The distance between the closest set of n (defined by the thr_coincidence_sum parameter) seismometers defines a critical distance the seismic waves must cover for coincidence triggering to detect events simultaneously at n seismometers. The default value for the average wave speed is 2 km/s.
    thr_event_join : float, optional
        The maximum duration of gaps between triggered events before those events are considered separate. Joined events are reported as a single event in the (reference) event catalogue. The maximum gap duration is assumed to be 0.5 seconds by default.
    thr_coincidence_sum : int, optional
        The number of seismometers, n, on which an event must be detected for that event to be included in the (reference) event catalogue. By default an event must be detected at every seismometer.
    thr_on : float, optional
        Threshold for switching single seismometer trigger on. The default value is a threshold of 5.
    thr_off : float, optional
        Threshold for switching single seismometer trigger off. The default value is a threshold of 1.
    options
        Necessary keyword arguments for the respective trigger algorithm that will be passed on. For example ‘sta’ and ‘lta’ for any STA/LTA variant (e.g. sta=3, lta=10). Arguments ‘sta’ and ‘lta’ (seconds) will be mapped to ‘nsta’ and ‘nlta’ (samples) by multiplying by the sampling rate of trace (e.g. sta=3, lta=10 would call the trigger algorithm with 3 and 10 seconds average, respectively).
        
    Returns
    -------
    events : DataFrame
        A pandas dataframe containing the events in the form of a (reference) event catalogue with eight columns including the reference start time and duration of the event based on coincidence triggering. The format of the event catalogue is detailed in the Event and Trace Catalogues section of this documentation.
    traces : DataFrame
        A pandas dataframe containing the trace (metadata) with eight columns including the start time and duration of the triggers for each trace based on single station triggering. The format of the trace (metadata) catalogue is detailed in the Event and Trace Catalogues section of the documentation.
    """
    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    # and with components added in quadrature (i.e. energy)
    stream_list = group_components(component_list, signal_type=signal_type)
    
    # create new stream of the quadrature streams from each seismometer
    new_stream = None
    for i in range(0, len(stream_list)):
        if i == 0:
            new_stream = stream_list[0]
        else:
            new_stream = new_stream + stream_list[i]
            
    if thr_coincidence_sum <= 0:
        thr_coincidence_sum = len(stream_list)
    else:
        thr_coincidence_sum = min(len(stream_list), thr_coincidence_sum)
    
    # get distances between array elements
    distance = __get_distances(stream, starttime, station_name=station_name, thr_coincidence_sum=thr_coincidence_sum)

    # trigger events using specified event detection algorithm
    events, coincident_events = __coincidence_trigger(trigger_type=trigger_type, thr_on=thr_on, thr_off=thr_off, stream=new_stream, nseismometers=len(stream_list), thr_travel_time=distance/avg_wave_speed, thr_event_join=thr_event_join, thr_coincidence_sum=thr_coincidence_sum, **options)
    
    events_df, traces_df = __make_catalogues(coincident_events, stream, events, stream_list, starttime, endtime, signal_type, thr_travel_time=distance/avg_wave_speed, thr_coincidence_sum=thr_coincidence_sum)

    return events_df, traces_df

def __get_distances(stream, starttime, station_name='stations', thr_coincidence_sum=1):
    """
    Private function for get_events() to calcualte maximum distance between any seismometer in a given array and its closest (n-1) neighbours.
    """
    if thr_coincidence_sum <= 1:
        return 0
    else:
        # create list to store latitude and longitude of each unique seismometer
        coordinates_list = []
        seismometer_list = []
        
        # find coordinates of each unique seismometer, excluding channels
        for i in range(0, len(stream)):
            filename = os.path.join(station_name, stream[i].stats.network+'.'+stream[i].stats.station+'.'+stream[i].stats.location+'.'+stream[i].stats.channel+'.xml')

            location = stream[i].stats.network+'.'+stream[i].stats.station+'.'+stream[i].stats.location
            if i == 0 or not np.any(location == np.asarray(seismometer_list)):
                try:
                    inv = read_inventory(filename)
                    coordinates = inv.get_coordinates(stream[i].id, stream[i].stats.starttime)
                    coordinates['location'] = location
                    # append to lists
                    coordinates_list.append(coordinates)
                    seismometer_list.append(coordinates['location'])
                except:
                    warnings.filterwarnings('always', category=UserWarning)
                    warnings.warn(stream[i].stats.channel+' channel not available in '+filename+' in local directory.', category=UserWarning)
        
        # calculate distances between each pair of seismometers
        distances_list = np.zeros((len(coordinates_list), len(coordinates_list)))
        for i in range(0, len(coordinates_list)):
            for j in range(i + 1, len(coordinates_list)):
                distances_list[i][j] = degrees2kilometers(locations2degrees(coordinates_list[i]['latitude'], coordinates_list[i]['longitude'], coordinates_list[j]['latitude'], coordinates_list[j]['longitude']))
        
        # calculate maximum distance between closest n seismometers
        if thr_coincidence_sum == 2:
            # distance between closest pair
            return np.min(distances_list[distances_list > 0])
        elif thr_coincidence_sum < len(coordinates_list) and thr_coincidence_sum <= 5:
            if thr_coincidence_sum == 3:
                # distance between closest set of three seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[j][k]))
                return np.min(distances_matrix[distances_matrix > 0])
            elif thr_coincidence_sum == 4:
                # distance between closest set of four seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            for l in range(k + 1, len(coordinates_list)):
                                distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[i][l], distances_list[j][k], distances_list[j][l], distances_list[k][l]))
                return np.min(distances_matrix[distances_matrix > 0])
            else:
                # distance between closest set of five seismometers
                distances_matrix = np.zeros((len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list), len(coordinates_list)))
                for i in range(0, len(coordinates_list)):
                    for j in range(i + 1, len(coordinates_list)):
                        for k in range(j + 1, len(coordinates_list)):
                            for l in range(k + 1, len(coordinates_list)):
                                for m in range(l + 1, len(coordinates_list)):
                                    distances_matrix[i][j][k] = np.max((distances_list[i][j], distances_list[i][k], distances_list[i][l], distances_list[i][m], distances_list[j][k], distances_list[j][l], distances_list[j][m], distances_list[k][l], distances_list[k][m], distances_list[l][m]))
                return np.min(distances_matrix[distances_matrix > 0])
        else:
            # too computationally inefficient, so use maximum distance in array
            return np.max(distances_list)

def __make_catalogues(events, stream, events_list, stream_list, starttime, endtime, signal_type='amplitude', thr_travel_time=0, thr_coincidence_sum=1):
    """
    Private function for get_events() to create reference and trace catalogues of the identified events.
    """
    ## REFERENCE CATALOGUE
    # output relevant columns to pandas dataframe
    events_df = pd.DataFrame(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration', 'ref_amplitude', 'ref_energy'])
    if len(events) > 0:
        df = pd.DataFrame(events)[['time', 'duration', 'stations']]
    else:
        raise Exception('Start/end time and stream are for different time periods.')

    # remove events outside requested time window and less than 10 times the sampling rate
    df = df[np.logical_and(df['time'] + df['duration'] > starttime, df['time'] < endtime)]
    df = df[df['duration'] > 10./stream[0].stats.sampling_rate]
    df.reset_index(drop=True, inplace=True)
    
    # add columns with peak amplitude and energy of top N stations, and the event id
    attributes = get_attributes(df, stream, attribute_summary)
    for i in range(0, len(df['time'])):
        event_id = '{:0>4d}'.format(df['time'][i].year)+'{:0>2d}'.format(df['time'][i].month)+'{:0>2d}'.format(df['time'][i].day)+'T'+'{:0>2d}'.format(df['time'][i].hour)+'{:0>2d}'.format(df['time'][i].minute)+'{:0>2d}'.format(df['time'][i].second)+'Z'
        if thr_coincidence_sum > 1:
            amplitude = np.mean(np.asarray(attributes['amplitude'][i])[np.argsort(attributes['amplitude'][i])[-thr_coincidence_sum:]])
            energy = np.mean(np.asarray(attributes['energy'][i])[np.argsort(attributes['energy'][i])[-thr_coincidence_sum:]])
        else:
            amplitude = np.mean(attributes['amplitude'][i])
            energy = np.mean(attributes['energy'][i])
        events_df.loc[i] = list([event_id, df['stations'][i], thr_travel_time, df['time'][i], df['duration'][i], amplitude, energy])
    
    # add column to store signal_type used in triggering
    if signal_type == 'amplitude':
        events_df['signal_type'] = 'amplitude'
    else:
        events_df['signal_type'] = 'energy'
    
    ## TRACE CATALOGUE
    # find start time and duration of event at each seismometer
    if isinstance(stream_list, (list, np.ndarray)):
        traces_df = None
        k = 0
        for i in range(0, len(stream_list)):
            trace_df = pd.DataFrame(columns=['event_id', 'station', 'components', 'time', 'duration'])
            
            # create array of components in stream
            components = []
            for j in range(0, len(stream_list[i][0].stats.channel)):
                if int(j - 2) % 3 == 0:
                    components.append(stream_list[i][0].stats.channel[j - 2:j + 1])
            
            df = pd.DataFrame(events_list)[['time', 'duration', 'stations']]
            df = df[df['stations'] == stream_list[i][0].stats.network+'.'+stream_list[i][0].stats.station+'.'+stream_list[i][0].stats.location]
            df.reset_index(drop=True, inplace=True)

            l = 0
            # find all events within range of reference event, including times extending beyond reference event
            for j in range(0, len(events_df['ref_time'])):
                index = np.logical_and(np.any(np.asarray(events_df['stations'][j]) == stream_list[i][0].stats.network+'.'+stream_list[i][0].stats.station+'.'+stream_list[i][0].stats.location), np.logical_and(df['time'] <= events_df['ref_time'][j] + events_df['ref_duration'][j] + thr_travel_time/2., df['time'] + df['duration'] + thr_travel_time/2. >= events_df['ref_time'][j]))
                if np.sum(index) > 0:
                    trace_df.loc[l] = list([events_df['event_id'][j], df['stations'][0], components, np.min(df['time'][index]), np.max(df['time'][index] + df['duration'][index]) - np.min(df['time'][index])])
                    l = l + 1

            # add columns with peak amplitude and energy
            if len(trace_df) > 0:
                attributes = get_attributes(trace_df, stream_list[i], attribute_summary)
                trace_df['amplitude'] = attributes['amplitude']
                trace_df['energy'] = attributes['energy']
                
                # append dataframe for this seismometer to final catalogue
                if k == 0:
                    traces_df = trace_df
                else:
                    traces_df = traces_df.append(trace_df, ignore_index=True)
                k = k + 1
    
        # add column to store signal_type used in triggering
        if signal_type == 'amplitude':
            traces_df['signal_type'] = 'amplitude'
        else:
            traces_df['signal_type'] = 'energy'
    else:
        traces_df = events_df
        # rename columns to match expected format for trace catalogue
        traces_df.rename(columns = {'ref_time': 'time', 'ref_duration': 'duration', 'ref_amplitude': 'amplitude', 'ref_energy': 'energy'}, inplace = True)
            
    return events_df, traces_df

def group_components(component_list, signal_type='amplitude'):
    """
    Function to calculate the Euclidean norm of the waveform amplitude for seismometers with multiple component mea- surements. The normal can be returned as an absolute value amplitude waveform or an energy waveform.
    
    Parameters
    ----------
    stream : Stream
        Stream containing waveform data for each component from one or more seismometers between the start and end time of the event.
    signal_type : str, optional
        Specify whether components are grouped as an ‘amplitude’ (i.e. absolute value) or ‘energy’ (i.e. amplitude-squared) waveform. The components are grouped as an amplitude waveform by default.
        
    Returns
    -------
    stream_list : list
        Return a list of streams for each seismometer with the data representing the amplitude or energy of the waveform measured by taking the normal of the signal from each component. The first stream is accessed as group_components(...)[0] and the trace of that stream as group_components(...)[0][0].
    """
    # convert stream to a list if it is a Stream object for a single seismometer
    if not isinstance(component_list, (list, np.ndarray)):
        component_list = [component_list]
    
    # create lists to store streams for each seismometer (if applicable)
    stream_list = [] # total amplitude (or energy)
    
    # combine traces at each seismometer in quadrature as appropriate
    for i in range(0, len(component_list)):
        # find latest start time and earlest stop time across the compenents at the given seismometer
        starttime, endtime = 0, 1e99
        for j in range(0, len(component_list[i])):
            if component_list[i][j].stats.starttime > starttime:
                starttime = component_list[i][j].stats.starttime
            if component_list[i][j].stats.endtime < endtime:
                endtime = component_list[i][j].stats.endtime
                
        # find weighted mean for the seismometer and component in this trace
        mean_list = []
        for j in range(0, len(component_list[i])):
            count = len(component_list[i][j].slice(starttime, endtime).data)
            mean = np.sum(component_list[i][j].slice(starttime, endtime).data)
            mean_list.append(float(mean)/count)
    
        for j in range(0, len(component_list[i])):
            # create new stream object to store combined components
            if j == 0:
                new_stream = Stream(component_list[i][0].slice(starttime, endtime))
                new_stream[0].data = (component_list[i][0].slice(starttime, endtime).data - mean_list[0])**2
                stream_list.append(new_stream)
            else:
                # add additional components to stream in quadrature
                stream_list[i][0].data += (component_list[i][j].slice(starttime, endtime).data - mean_list[j])**2
                # modify trace id to terminate in number of components
                stream_list[i][0].stats.channel = stream_list[i][0].stats.channel + component_list[i][j].stats.channel
                
        # if requested output is amplitude convert data to amplitudes
        if signal_type == 'amplitude':
            stream_list[i][0].data = np.sqrt(stream_list[i][0].data)
        
    return stream_list
    
def __group_seismometers(stream):
    """
    Private function for get_events() to separate each seismometer into a unique Stream comprising all channels recorded at that seismometer.
    """
    # create lists to store streams for each seismometer (if applicable)
    component_list = [] # amplitude of each component

    # create a copy of the input stream with components added in quadrature
    for i in range(0, len(stream)):
        # create new stream object to store combined components
        if i == 0:
            component_list.append(Stream(stream[0].copy()))
        else:
            for j in range(0, len(component_list)):
                # test if current seismometer has a stream in the list
                if (stream[i].stats.network == component_list[j][0].stats.network and stream[i].stats.station == component_list[j][0].stats.station and (stream[i].stats.channel)[0:-1] == (component_list[j][0].stats.channel)[0:-1]):
                    # seismometer in the list
                    component_list[j] += Stream(stream[i].copy())
                    break
            else:
                # seismometer not in the list; so add it
                component_list.append(Stream(stream[i].copy()))
                
    return component_list

def __coincidence_trigger(trigger_type, thr_on, thr_off, stream, nseismometers, thr_travel_time=0, thr_event_join=10, thr_coincidence_sum=1, trigger_off_extension=60, **options):
    """
    Private function for get_events(), based on the obspy coincidence_trigger function, to identify events based on simultaneous detections at n seismometers (with possible short gaps of duration thr_event_join). This function outputs two lists with the time, duration and station for (1) triggers at single seismometers and (2) detections of events.
    """
    st = stream.copy()
    # use all traces ids found in stream
    trace_ids = [tr.id for tr in st]
    # we always work with a dictionary with trace ids and their weights later
    if isinstance(trace_ids, list) or isinstance(trace_ids, tuple):
        trace_ids = dict.fromkeys(trace_ids, 1)

    # the single station triggering
    triggers = []
    single_triggers = []
    # prepare kwargs for trigger_onset
    kwargs = {'max_len_delete': False}
    for tr in st:
        if tr.id not in trace_ids:
            msg = "At least one trace's ID was not found in the " + \
                  "trace ID list and was disregarded (%s)" % tr.id
            warnings.warn(msg, UserWarning)
            continue
        if trigger_type is not None:
            tr.trigger(trigger_type, **options)
        max_trigger_length = 1e6
        kwargs['max_len'] = int(
            max_trigger_length * tr.stats.sampling_rate + 0.5)
        tmp_triggers = trigger_onset(tr.data, thr_on, thr_off, **kwargs)
        # find triggers for given station
        prv_on, prv_off = -1000, -1000
        for on, off in tmp_triggers:
            on = tr.stats.starttime + float(on) / tr.stats.sampling_rate
            off = tr.stats.starttime + float(off) / tr.stats.sampling_rate
            # extend previous event if only small gap
            if prv_on < 0:
                # update on and off times for first event
                prv_on = on
                prv_off = off
            elif on <= prv_off + thr_event_join:
                # update off time assuming continuing event
                prv_off = off
            else:
                # add previous trigger to catalogue
                triggers.append([prv_on.timestamp, prv_off.timestamp, tr.id])
                # add trigger to trace catalogue
                event = {}
                event['time'] = UTCDateTime(prv_on)
                event['stations'] = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
                event['trace_ids'] = tr.id
                event['coincidence_sum'] = 1.0
                event['duration'] = prv_off - prv_on
                single_triggers.append(event)
                # update on and off times
                prv_on = on
                prv_off = off
        # add final trigger to catalogue
        if prv_on > 0:
            triggers.append([prv_on.timestamp, prv_off.timestamp, tr.id])
            # add trigger to event catalogue
            event = {}
            event['time'] = UTCDateTime(prv_on)
            event['stations'] = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
            event['trace_ids'] = tr.id
            event['coincidence_sum'] = 1.0
            event['duration'] = prv_off - prv_on
            single_triggers.append(event)
    triggers.sort()

    # the coincidence triggering and coincidence sum computation
    coincidence_triggers = []
    last_off_time = [0.0]
    while triggers != []:
        # remove first trigger from list and look for overlaps
        on, off, tr_id = triggers.pop(0)
        on = on - thr_travel_time
        sta = (tr.id).split(".")[0]+'.'+(tr.id).split(".")[1]+'.'+(tr.id).split(".")[2]
        # add trigger to event catalogue
        event = {}
        event['time'] = [UTCDateTime(on)]
        event['off_time'] = [UTCDateTime(off)]
        event['stations'] = [tr_id.split(".")[0]+'.'+tr_id.split(".")[1]+'.'+tr_id.split(".")[2]]
        event['trace_ids'] = [tr_id]
        # compile the list of stations that overlap with the current trigger
        k = 0
        for trigger in triggers:
            tmp_on, tmp_off, tmp_tr_id = trigger
            tmp_sta = (tmp_tr_id).split(".")[0]+'.'+(tmp_tr_id).split(".")[1]+'.'+(tmp_tr_id).split(".")[2]
            if np.any(tmp_sta == np.asarray(event['stations'])):
                pass # station already included so do not add travel time again
            else:
                tmp_on = tmp_on - thr_travel_time
            # break if there is a gap in between the two triggers
            if tmp_on > off + trigger_off_extension: # place limit on number of triggers; must be within a small time of the last trigger
                break
            if k == 10*nseismometers**2:
                warnings.filterwarnings('always', category=UserWarning)
                warnings.warn('Too many triggers joined together; consider looking at a smaller time window to improve computational efficiency.', category=UserWarning)
                warnings.filterwarnings('ignore', category=Warning)
            event['time'].append(UTCDateTime(tmp_on))
            event['off_time'].append(UTCDateTime(tmp_off))
            event['stations'].append(tmp_sta)
            event['trace_ids'].append(tmp_tr_id)
            # allow sets of triggers that overlap only on subsets of all stations (e.g. A overlaps with B and B overlaps w/ C => ABC)
            off = max(off, tmp_off)
            k = k + 1
        
        # find on and off time of first region with multiple triggers
        trigger_times = event['time'] + event['off_time']
        trigger_stations = event['stations'] + event['stations']
        trigger_traces = event['trace_ids'] + event['trace_ids']
        trigger_sum = np.asarray([1]*len(event['time']) + [-1]*len(event['off_time']))
        index = np.argsort(trigger_times.copy())
        # initialise variables
        coincidence_sum, event['coincidence_sum'], join_time = 0, 0, None
        event['stations'], event['trace_ids'] = [], []
        for i in range(0, len(index)):
            coincidence_sum = coincidence_sum + trigger_sum[index[i]]
            # coincidence sum region
            if coincidence_sum >= thr_coincidence_sum:
                # set start time if over threshold for the first time
                if isinstance(event['time'], list):
                    event['time'] = trigger_times[index[i]]
                # update end time
                event['off_time'] = trigger_times[index[i]]
                event['duration'] = event['off_time'] - event['time']
                # update maximum coincidence sum for detection
                event['coincidence_sum'] = max(coincidence_sum, event['coincidence_sum'])
                # add station and trace_id to event catalogue
                if trigger_sum[index[i]] > 0:
                    event['stations'].append(trigger_stations[index[i]])
                    event['trace_ids'].append(trigger_traces[index[i]])
                # reset join time if coincidence trigger condition met again
                join_time = None
            else:
                # before coincidence sum region
                if isinstance(event['time'], list):
                    # add station and trace_id to event catalogue and remove if it detriggers before coincidence sum region
                    if trigger_sum[index[i]] > 0:
                        event['stations'].append(trigger_stations[index[i]])
                        event['trace_ids'].append(trigger_traces[index[i]])
                    else:
                        event['stations'].remove(trigger_stations[index[i]])
                        event['trace_ids'].remove(trigger_traces[index[i]])
                # after coincidence sum region
                else:
                    # update end time
                    event['off_time'] = trigger_times[index[i]]
                    event['duration'] = event['off_time'] - event['time']
                    if join_time == None:
                        join_time = event['off_time']
                    elif (event['off_time'] - join_time) > thr_event_join or coincidence_sum < 1:
                        # only join if at least one seismometer active
                        break
        # update end time and duration in case coincidence trigger did not join events
        if not join_time == None:
            event['off_time'] = join_time
            event['duration'] = join_time - event['time']
                    
        # remove duplicate stations and trace_ids (as applicable)
        event['stations'] = list(dict.fromkeys(event['stations']))
        event['trace_ids'] = list(dict.fromkeys(event['trace_ids']))

        # skip if both coincidence sum and similarity thresholds are not met
        if event['coincidence_sum'] < thr_coincidence_sum:
            continue
        # skip coincidence trigger if it is just a subset of the previous (determined by a shared off-time, this is a bit sloppy)
        if np.any(np.asarray(last_off_time) - float(event['off_time']) >= 0):
            continue
        
        # add event to catalogue and center times
        event['time'], event['off_time'] = event['time'] + thr_travel_time/2., event['off_time'] + thr_travel_time/2.
        coincidence_triggers.append(event)
        last_off_time.append(event['off_time'])
    
    # remove keys used in computation only
    for trigger in coincidence_triggers:
        trigger.pop('off_time')

    return single_triggers, coincidence_triggers


# define functions to calculate attributes of the waveforms
def get_attributes(events, stream, *attributes):
    """
    Function to calculate characteristics of seismic waveform data at times of identified seismic events. The event catalogue output by the get_events function is used to extract the waveform data at the relevant times from an input obspy Stream object. Pre-defined or user-defined attribute functions are applied to the waveform data from each seismometer and channel (e.g. ‘BH?’, ‘HH?’, ‘LH?’) over the duration of each event. The attribute function specifies the calculation of one or more characteristics based on the component waveform. The value for each requested attribute is tabulated for the event catalogue. If multiple seismometers are present, the average value of each attribute is returned for a given event.
    
    Parameters
    ----------
    events : DataFrame
        List of (reference) events or traces (metadata) stored as a pandas dataframe including at minimum a column named ‘time’ (or ‘ref_time’) for the start time of the event and a column named ‘duration’ (or ‘ref_duration’) for the length of the event.
    stream : Stream
        Stream object containing waveform data for each component from one or more seismometers.
    attributes
        Pre-defined or user defined attribute functions to apply to the waveform data. Each function is included separately and comma separated; e.g. get_attributes(..., seismic_attributes.waveform_attributes, user_defined_attribute, seismic_attributes.polarity_attributes).
        
    Returns
    -------
    attributes : DataFrame
        Return a pandas dataframe with the attributes of each event or trace. The dataframe includes columns for the start time and duration of the event in addition to event data from the event or trace catalogue. The remaining columns are for the requested attributes and are named as specified in their respective attribute function. The format of the attribute catalogue is detailed in the Event and Trace Catalogue section of the documentation.
    """
    # variable to check if first event has been found
    first_event = 0

    # catch error if user inputs tuple with event and trace catalogue
    if isinstance(events, (tuple, list, np.ndarray)):
        events = events[0] # set as event catalogue by default
        
    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    attributes_df = None

    # calculate attributes for each event
    for i in range(0, len(events)):
        # store list of attributes if multiple seismometers are used
        attribute_list, attribute_List = None, None
        
        for j in range(0, len(component_list)):
            # find start and end time of event in seconds from start of stream
            try:
                start_time = UTCDateTime(events['ref_time'][i])
                end_time = UTCDateTime(events['ref_time'][i]) + events['ref_duration'][i]
                
                name_list = ['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration']
                attribute_list = [events['event_id'][i], events['stations'][i], events['network_time'][i], start_time, end_time - start_time]
                if first_event == 0:
                    first_event = 1
            except:
                start_time = UTCDateTime(events['time'][i])
                end_time = UTCDateTime(events['time'][i]) + events['duration'][i]
                
                # only include streams from correct seismometer if using trace catalogue
                try:
                    if not events['station'][i] == component_list[j][0].stats.network+'.'+component_list[j][0].stats.station+'.'+component_list[j][0].stats.location:
                        continue
                    name_list = ['event_id', 'station', 'components', 'time', 'duration']
                    attribute_list = [events['event_id'][i], events['station'][i], events['components'][i], start_time, end_time - start_time]
                    if first_event == 0:
                        first_event = 1
                except: # this path is used by get_events function
                    name_list = ['time', 'duration']
                    attribute_list = [start_time, end_time - start_time]
                    if first_event == 0:
                        first_event = 1
                
            # add waveform data during event to new stream
            component_stream = component_list[j].slice(start_time, end_time)
            
            # add attribute names and values to lists
            if len(component_stream) > 0:
                for l in range(0, len(attributes)):
                    name, attribute = attributes[l](component_stream, start_time, end_time)
                    if isinstance(attribute, (list, np.ndarray)):
                        name_list.extend(name)
                        attribute_list.extend(attribute)
                    else:
                        name_list.append(name)
                        attribute_list.append(attribute)
            
            # add attributes from each seismometer to list
            if attribute_List == None:
                if attribute_list == None:
                    raise Exception('No events found: check event list and stream are for different time periods.')
                else:
                    attribute_List = attribute_list
            else:
                for k in range(2, len(attribute_List)):
                    if not isinstance(attribute_List[k], (list, np.ndarray)):
                        attribute_List[k] = [attribute_List[k]]
                    attribute_List[k].append(attribute_list[k])
        
        # create pandas dataframe to store attributes
        if first_event == 1:
            attributes_df = pd.DataFrame(columns=name_list)
            first_event = 2
        # append attributes for each event to dataframe
        if first_event >= 1 and not attribute_List == None:
            attributes_df.loc[len(attributes_df)] = attribute_List
        
    return attributes_df

def attribute_summary(component_stream, event_start, event_stop):
    """
    Function to calculate the peak amplitude and energy of an event from the waveform data passed on from the get_attributes function.
    """
    # Peak amplitude and energy
    trace = group_components(component_stream.copy(), signal_type='energy')[0][0]

    return ['amplitude', 'energy'], [np.sqrt(np.max(trace.data)), np.sum(trace.data)*component_stream[0].stats.delta]

def waveform_attributes(component_stream, event_start, event_stop):
    """
    Function to calculate the waveform attributes of an event from the waveform data passed on from the get_attributes function. Attributes 1 through 22 are returned as log-scaled (base 10) values.
    
    Parameters
    ----------
    component_stream : Stream
        Stream object containing waveform data for each component from a single seismometer between the start and end time of the event.
    event_start : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap); the stream is already truncated to this range in the get_attributes function.
    event_stop : UTCDateTime
        Limit results to time series samples ending (one sample) before the specified end time; the stream is already truncated to this range in the get_attributes function.
        
    Returns
    -------
    attribute : tuple
        A tuple containing the name of the attributes and their values; i.e. [‘attribute_1’, ..., ‘attribute_22’] as a list of strings and the values for the waveform attributes of the event as a list of floats. Attributes 1 through 22 are returned as log-scaled (base 10) values.
    """
    # Bundle of waveform attributes from Provost et al. (2016)
    name_list, attribute_list = [], []
    names, attributes = attribute_1(component_stream, event_start, event_stop)
    name_list.append(names)
    attribute_list.append(attributes)
    names, attributes = attribute_2_3(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_4(component_stream, event_start, event_stop)
    name_list.append(names)
    attribute_list.append(attributes)
    names, attributes = attribute_5_6(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_7_8(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_10_11_12(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_13_14_15_16_17(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_18_19_20_21_22(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    
    return name_list, attribute_list
    
def attribute_1(component_stream, event_start, event_stop):
    # Duration (log-scale)
    return 'attribute_1', np.log10(event_stop - event_start)

def attribute_2_3(component_stream, event_start, event_stop):
    # Ratio of the mean and median over the maximum of the envelop signal (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_envelope = envelope(trace.data)
    
    return ['attribute_2', 'attribute_3'], [np.log10(np.mean(trace_envelope)/np.max(trace_envelope)), np.log10(np.median(trace_envelope)/np.max(trace_envelope))]

def attribute_4(component_stream, event_start, event_stop):
    # Ratio between ascending and descending time (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    event_max = trace.times()[np.argmax(trace.data)]

    return 'attribute_4', np.log10((event_max)/np.maximum(1e-3*event_max, (event_stop - event_start) - event_max)) # cap ratio at 1000
    
def attribute_5_6(component_stream, event_start, event_stop):
    # Kurtosis of the raw signal and envelope (peakness of the signal) (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_envelope = envelope(trace.data)
    
    kurtosis_trace = np.mean(trace.data**4)/(np.mean(trace.data**2))**2 # mean of rectified signal is already zero
    kurtosis_envelope = np.mean(trace_envelope**4)/(np.mean(trace_envelope**2))**2 # mean of rectified signal is already zero

    return ['attribute_5', 'attribute_6'], [np.log10(kurtosis_trace), np.log10(kurtosis_envelope)]
    
def attribute_7_8(component_stream, event_start, event_stop):
    # Skewness of the raw signal and envelope (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_envelope = envelope(trace.data)
    
    skewness_trace = np.mean(trace.data**3)/(np.mean(trace.data**2))**1.5 # mean of rectified signal is already zero
    skewness_envelope = np.mean(trace_envelope**3)/(np.mean(trace_envelope**2))**1.5 # mean of rectified signal is already zero

    return ['attribute_7', 'attribute_8'], [np.log10(skewness_trace), np.log10(skewness_envelope)]
    
def attribute_10_11_12(component_stream, event_start, event_stop):
    # Energy in the first third and the remaining part of the autocorrelation function (log-scale)
    trace = group_components(component_stream.copy(), signal_type='energy')[0][0]
    acf_third = np.sum(correlate(trace, trace, int(len(trace.data)/3))[int(len(trace.data)/3):-1])*trace.stats.delta
    acf_total = np.sum(correlate(trace, trace, int(len(trace.data)))[int(len(trace.data)):-1])*trace.stats.delta

    return ['attribute_10', 'attribute_11', 'attribute_12'], [np.log10(acf_third), np.log10(acf_total - acf_third), np.log10((acf_total - acf_third)/np.maximum(1e-3*(acf_total - acf_third), acf_third))] # cap ratio at 1000
    
def attribute_13_14_15_16_17(component_stream, event_start, event_stop):
    # Energy of the signal filtered in 5 – 10 Hz, 10 – 50 Hz, 5 – 70 Hz, 50 - 100 Hz, and 5 - 100 Hz (log-scale)
    band_13 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=10), signal_type='energy')[0][0])*component_stream[0].stats.delta
    band_14 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=10, freqmax=50), signal_type='energy')[0][0])*component_stream[0].stats.delta
    band_15 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=70), signal_type='energy')[0][0])*component_stream[0].stats.delta
    try:
        band_16 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=50, freqmax=100), signal_type='energy')[0][0])*component_stream[0].stats.delta
    except:
        band_16 = 1 # these frequencies are too high for a lot of seismometers
    band_17 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=100), signal_type='energy')[0][0])*component_stream[0].stats.delta
    
    return ['attribute_13', 'attribute_14', 'attribute_15', 'attribute_16', 'attribute_17'], [np.log10(band_13), np.log10(band_14), np.log10(band_15), np.log10(band_16), np.log10(band_17)]
    
def attribute_18_19_20_21_22(component_stream, event_start, event_stop):
    # Kurtosis of the signal in 5 – 10 Hz, 10 – 50 Hz, 5 – 70 Hz, 50 - 100 Hz, and 5 - 100 Hz frequency range (log-scale)
    band_18 = group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=10), signal_type='amplitude')[0][0].data
    band_19 = group_components((component_stream.copy()).filter("bandpass", freqmin=10, freqmax=50), signal_type='amplitude')[0][0].data
    band_20 = group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=70), signal_type='amplitude')[0][0].data
    try:
        band_21 = group_components((component_stream.copy()).filter("bandpass", freqmin=50, freqmax=100), signal_type='amplitude')[0][0].data
    except:
        band_21 = None # these frequencies are too high for a lot of seismometers
    band_22 = group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=100), signal_type='amplitude')[0][0].data
    
    kurtosis_18 = np.mean(band_18**4)/(np.mean(band_18**2))**2 # mean of rectified signal is already zero
    kurtosis_19 = np.mean(band_19**4)/(np.mean(band_19**2))**2 # mean of rectified signal is already zero
    kurtosis_20 = np.mean(band_20**4)/(np.mean(band_20**2))**2 # mean of rectified signal is already zero
    if not isinstance(band_21, (list, np.ndarray)) and band_21 == None:
        kurtosis_21 = 1
    else:
        kurtosis_21 = np.mean(band_21**4)/(np.mean(band_21**2))**2 # mean of rectified signal is already zero
    kurtosis_22 = np.mean(band_22**4)/(np.mean(band_22**2))**2 # mean of rectified signal is already zero
    
    return ['attribute_18', 'attribute_19', 'attribute_20', 'attribute_21', 'attribute_22'], [np.log10(kurtosis_18), np.log10(kurtosis_19), np.log10(kurtosis_20), np.log10(kurtosis_21), np.log10(kurtosis_22)]

def spectral_attributes(component_stream, event_start, event_stop):
    """
    Function to calculate the spectral attributes of an event from the waveform data passed on from the get_attributes function. Attributes 24 through 37 are returned as log-scaled (base 10) values.
    
    Parameters
    ----------
    component_stream : Stream
        Stream object containing waveform data for each component from a single seismometer between the start and end time of the event.
    event_start : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap); the stream is already truncated to this range in the get_attributes function.
    event_stop : UTCDateTime
        Limit results to time series samples ending (one sample) before the specified end time; the stream is already truncated to this range in the get_attributes function.
        
    Returns
    -------
    attribute : tuple
        A tuple containing the name of the attributes and their values; i.e. [‘attribute_24’, ..., ‘attribute_40’] as a list of strings and the values for the spectral attributes of the event as a list of floats. Attributes 24 through 37 are returned as log-scaled (base 10) values.
    """
    # Bundle of spectral attributes from Provost et al. (2016)
    name_list, attribute_list = [], []
    names, attributes = attribute_24_25_26_27_28_29_30(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_34_35_36_37(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    names, attributes = attribute_38_39_40(component_stream, event_start, event_stop)
    name_list.extend(names)
    attribute_list.extend(attributes)
    
    return name_list, attribute_list
    
def attribute_24_25_26_27_28_29_30(component_stream, event_start, event_stop):
    # Mean and max of the DFT, frequency at the maximum, central frequency of the 1st and 2nd quartile, and median and variance of the normalized DFT (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_fft = np.absolute(rfft(trace.data))
    trace_freq = rfftfreq(len(trace), d=trace.stats.delta)
    
    freq_max = trace_freq[np.argmax(trace_fft)]/2. # halve frequency due for rectified amplitude
    central_27 = np.sum(trace_fft[0:len(trace_fft)//4]*trace_freq[0:len(trace_fft)//4])/np.sum(trace_fft[0:len(trace_fft)//4])
    central_28 = np.sum(trace_fft[len(trace_fft)//4:len(trace_fft)//2]*trace_freq[len(trace_fft)//4:len(trace_fft)//2])/np.sum(trace_fft[len(trace_fft)//4:len(trace_fft)//2])
    trace_norm = trace_fft/np.mean(trace_fft) # amplitude of each measurement normalised to 1
    
    return ['attribute_24', 'attribute_25', 'attribute_26', 'attribute_27', 'attribute_28', 'attribute_29', 'attribute_30'], [np.log10(np.mean(trace_fft)), np.log10(np.max(trace_fft)), freq_max, np.log10(central_27), np.log10(central_28), np.log10(np.median(trace_norm)), np.log10(np.var(trace_norm))]
    
def attribute_34_35_36_37(component_stream, event_start, event_stop):
    # Energy in [0,1/4]Nyf, [1/4,1/2]Nyf, [1/2,3/4]Nyf, [3/4,1]Nyf (log-scale)
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_fft = np.absolute(rfft(trace.data))**2 # squared to give energy
    trace_freq = rfftfreq(len(trace), d=trace.stats.delta)
    nyquist_freq = trace.stats.sampling_rate/2.
    
    band_34 = np.sum(trace_fft[np.logical_and(0 <= trace_freq, trace_freq < nyquist_freq/4.)])*trace.stats.delta
    band_35 = np.sum(trace_fft[np.logical_and(nyquist_freq/4. <= trace_freq, trace_freq < nyquist_freq/2.)])*trace.stats.delta
    band_36 = np.sum(trace_fft[np.logical_and(nyquist_freq/2. <= trace_freq, trace_freq < 3*nyquist_freq/4.)])*trace.stats.delta
    band_37 = np.sum(trace_fft[np.logical_and(3*nyquist_freq/4. <= trace_freq, trace_freq < nyquist_freq)])*trace.stats.delta
        
    return ['attribute_34', 'attribute_35', 'attribute_36', 'attribute_37'], [np.log10(band_34), np.log10(band_35), np.log10(band_36), np.log10(band_37)]
    
def attribute_38_39_40(component_stream, event_start, event_stop):
    # Spectral centroid, gyration radius and spectral centroid width
    trace = group_components(component_stream.copy(), signal_type='amplitude')[0][0]
    trace_fft = np.absolute(rfft(trace.data))
    trace_freq = rfftfreq(len(trace), d=trace.stats.delta)
    
    gamma1 = np.mean(trace_fft*trace_freq**2)/np.mean(trace_fft*trace_freq)
    gamma2 = np.sqrt(np.mean(trace_fft*trace_freq**3)/np.mean(trace_fft*trace_freq**2))
    
    return ['attribute_38', 'attribute_39', 'attribute_40'], [gamma1, gamma2, np.sqrt(gamma1**2 - gamma2**2)]

def polarity_attributes(component_stream, event_start, event_stop):
    """
    Function to calculate the polarity attributes of an event from the waveform data passed on from the get_attributes function. This function is only applicable for seismometers with a three component signal (i.e. ‘Z’, ‘N’ and ‘E’ components).
    
    Parameters
    ----------
    component_stream : Stream
        Stream object containing waveform data for each component from a single seismometer between the start and end time of the event.
    event_start : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap); the stream is already truncated to this range in the get_attributes function.
    event_stop : UTCDateTime
        Limit results to time series samples ending (one sample) before the specified end time; the stream is already truncated to this range in the get_attributes function.
        
    Returns
    -------
    attribute : tuple
        A tuple containing the name of the attributes and their values; i.e. [‘attribute_68’, ..., ‘attribute_71’] as a list of strings and the values for the polarity attributes of the event as a list of floats.
    """
    # Bundle of polarity attributes from Provost et al. (2016)
    return attribute_68_69_70_71(component_stream, event_start, event_stop)
    
def attribute_68_69_70_71(component_stream, event_start, event_stop):
    # Rectilinearity, azimuth, dip and planarity
    if not len(component_stream) >= 3 or (not component_stream[0].id[-1] == 'Z' or not component_stream[1].id[-1] == 'N' or not component_stream[2].id[-1] == 'E'):
        raise Exception('Polarity attributes cannot be derived without at least one \'Z\', \'N\' and \'E\' component.')
    else:
        try:
            azimuth, incidence, rectillinearity, planarity = flinn(component_stream)
        except:
            azimuth, incidence, rectillinearity, planarity = np.nan, np.nan, np.nan, np.nan

    return ['attribute_68', 'attribute_69', 'attribute_70', 'attribute_71'], [rectillinearity, azimuth, incidence, planarity]


# define functions to plot identified events in waveform data
def plot_events(events, stream, starttime, endtime, filename=None):
    """
    Generates a plot of the seismic waveform used in event detection from first seismometer included in the obspy Stream with identified events highlighted. The signal is plotted as either an amplitude or energy waveform based on the signal type used in the event triggering algorithm in the get_events function. Note that the purpose of this plot is to gain an overview of the events detected (see also following function, if a manual visual check on a given event is desired). Multiple days of data are presented on separate pages in the output .pdf document.
    
    Parameters
    ----------
    events : DataFrame
        List of events or traces stored as a pandas dataframe including as a minimum a column named ‘time’ (or ‘ref_time’) for the start time of the event and a column named ‘duration’ (or ‘ref_duration’) for the length of the event.
    stream : Stream
        Stream object containing waveform data for each component from one or more seismometers. The waveform for the first seismometer in the stream is plotted using the signal type (amplitude or energy) specified in the event detection function.
    starttime : UTCDateTime
        Limit results to time series samples starting on the specified start time (or after that time in the case of a data gap).
    endtime : UTCDateTime
        Limit results to time series samples ending (one sample) before the specified end time.
    filename : str or path, optional
        Path to directory to write the .pdf document of plots. The default filename/path is ‘[station name]_event_plot.pdf’ located in the working directory.
        
    Returns
    -------
    None
    """
    warnings.filterwarnings('ignore', category=Warning)

    # catch error if user inputs tuple with event and trace catalogue
    if isinstance(events, (tuple, list, np.ndarray)):
        events = events[0] # set as event catalogue by default
    
    # set start and end time of each calendar day
    start_of_day = UTCDateTime(starttime.year, starttime.month, starttime.day)
    start_time = starttime
    end_time = min(endtime, start_of_day + __chunklength_in_sec)

    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    # and with components added in quadrature (i.e. energy)
    stream_list = group_components(component_list, signal_type=events['signal_type'][0])
    new_stream = stream_list[0] # first stream in list

    # create empty .pdf file
    if filename == None:
        pdf = PdfPages(os.path.abspath(new_stream[0].stats.station+'_event_plot.pdf'))
    else:
        pdf = PdfPages(os.path.abspath(filename+'.pdf'))

    # create stream including only times identified as events
    event_stream = Stream()
    for i in range(0, len(events)):
        try:
            event_stream += new_stream.slice(events['ref_time'][i], events['ref_time'][i] + events['ref_duration'][i])
        except:
            if events['station'][i] == new_stream[0].stats.network+'.'+new_stream[0].stats.station+'.'+new_stream[0].stats.location:
                event_stream += new_stream.slice(events['time'][i], events['time'][i] + events['duration'][i])

    # plot waveform data for each calendar day
    while (start_time < endtime):
        # create plot of waveform data
        fig = __plot_events(event_stream, new_stream, start_time, end_time, signal_type=events['signal_type'][0])
        
        # show first plot to confirm waveform data is as expected
        if start_time < starttime + __chunklength_in_sec:
            plt.show()
            
        # save figure frame to .pdf file
        pdf.savefig(fig)
        plt.clf()

        # update start and end time for each calendar day
        start_of_day = UTCDateTime(start_time.year, start_time.month, start_time.day)
        start_time = start_of_day + __chunklength_in_sec
        end_time = min(endtime, start_time + __chunklength_in_sec)
        
    # close .pdf file
    pdf.close()

def __plot_events(event_stream, stream, start_time, end_time, signal_type='amplitude'):
    """
    Private function for plot_events() to create plot of events overlaid on waveform data form a single day.
    """
    # create new figure object for single day plot
    my_dpi = 150
    fig = plt.figure(figsize=(9, 9), dpi=my_dpi)

    # set font sizes and use of Latex fonts
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # add entire waveform over calendar day in light grey
    if signal_type == 'amplitude':
        stream.plot(fig=fig, type="dayplot", interval=15, vertical_scaling_range=0.5*np.max(stream[0].data), color='lightgrey', starttime=start_time, endtime=end_time, title='')
    else:
        stream.plot(fig=fig, type="dayplot", interval=15, vertical_scaling_range=0.2*np.max(stream[0].data), color='lightgrey', starttime=start_time, endtime=end_time, title='')
    # add events over calendar day in red
    if len(event_stream) > 0:
        if signal_type == 'amplitude':
            event_stream.plot(fig=fig, type="dayplot", interval=15, vertical_scaling_range=0.5*np.max(stream[0].data), color='crimson', starttime=start_time, endtime=end_time, title='')
        else:
            event_stream.plot(fig=fig, type="dayplot", interval=15, vertical_scaling_range=0.2*np.max(stream[0].data), color='crimson', starttime=start_time, endtime=end_time, title='')
    
    # correct axis labels and ticks
    for item in (plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontsize(11)
    plt.xlabel(r'Time in minutes', fontsize=12)
    if signal_type == 'amplitude':
        plt.ylabel(r'Amplitude at '+str(stream[0].stats.station)+' on '+str(start_time.day)+' '+start_time.strftime('%B')+' '+str(start_time.year), fontsize=12, labelpad=6)
    else:
        plt.ylabel(r'Energy at '+str(stream[0].stats.station)+' on '+str(start_time.day)+' '+start_time.strftime('%B')+' '+str(start_time.year), fontsize=12, labelpad=6)
    fig.tight_layout()

    return fig

# define function to plot waveforms from trace database
def plot_waveforms(events, event_id, start_buffer=10, end_buffer=30, filename=None, waveform_name='waveforms', station_name='stations', providers=['IRIS', 'LMU', 'GFZ'], user=None, password=None, download=True):
    """
    Generates a plot of waveforms at all seismometers with a detection of a given event in the trace (metadata) catalogue. The purpose of this function is to enable a manual, visual check on a given event. The waveforms are separated into panels by the recording channel; only seismometers with an identical set of channels to that of the first seismometer (in order of input with a detection) are included on the plot. The function searches for the required waveform data in the specified local directory before attempting to download new waveform data.
    
    Parameters
    ----------
    events : DataFrame
        List of traces stored as a pandas dataframe including as a minimum a column named ‘time’ for the start time of the event and a column named ‘duration’ for the length of the event. The input must be either the trace (metadata) catalogue or trace attribute catalogue, not the (reference) event catalogue.
    event_id : str
        Unique time-based identifier for an event of the form ‘yyyymmddThhmmssZ’, for the year ‘yyyy’, month ‘mm’, day of month ‘dd’, hour ‘hh’, minute ‘mm’ and second ‘ss’. These are found in the (reference) event and trace (metadata) catalogues.
    start_buffer : float, optional
        Number of seconds of waveform data to display before the identified start time of event at each seismometer (in trace catalogue). The default value is 10 seconds.
    end_buffer : float, optional
        Number of seconds of waveform data to display after the identified end time of event at each seismometer (in trace catalogue). The default value is 30 seconds.
    filename : str or path, optional
        Path to directory to write the .pdf document of plots. The default filename/path is ‘[event_id]_plot.pdf’ located in the working directory.
    waveform_name : str or path, optional
        Path to directory to read (check for) and write waveform data (an existing file of the same name will not be overwritten). The default location is a directory named waveforms in the working directory.
    station_name : str or path, optional
        Path to directory to read (check for) and write station data (location coordinates, elevation etc, as provided by data repository). The default location is a directory named stations in the working directory.
    providers : str or list, optional
        One or more clients to use to download the requested waveform data if it does not already exist in the specified local directory. Multiple clients are entered as a list; e.g. [’IRIS’, ’GFZ’].
    user : str, optional
        User name of HTTP Digest Authentication for access to restricted data.
    password : str, optional
        Password of HTTP Digest Authentication for access to restricted data.
    download : bool, optional
        Specify whether days with missing waveform data are to be downloaded from client; e.g. True or False, alternatively 1 or 0. Missing data are downloaded by default.
        
    Returns
    -------
    None
    """
    # create new figure object for single day plot
    my_dpi = 150
    fig = plt.figure(figsize=(7.5, 6.5), dpi=my_dpi)

    # set font sizes and use of Latex fonts
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=10)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    # define colour scheme
    colormap = cm.get_cmap('ocean_r', 256)
    new_colormap = ListedColormap(colormap(np.linspace(0.5, 1, 256)))
        
    # catch error if user inputs tuple with event and trace catalogue
    if isinstance(events, (tuple, list, np.ndarray)):
        events = events[1] # set as trace catalogue by default
    try:
        a, b = events['station'], events['components']
    except:
        raise Exception('Input the trace catalogue of events or attributes, not the reference event catalogue.')

    # create copy of event catalogue and extract relevant waveforms
    events_df = events.loc[events['event_id'] == event_id].reset_index(drop=True)
    
    # download waveform data for each seismometer and channel
    stream_list = []
    channel_list = []
    x_min, x_max, y_min, y_max = 1e99, -1e99, 1e99, -1e99
    for i in range(0, len(events_df)):
        event = events_df.loc[i]
        # only plot seismometers with the same channels as the first seismometer
        if i == 0 or np.array_equal(event['components'], channel_list):
            channel_list = event['components']
            stream = get_waveforms(event['station'].split(".")[0], event['station'].split(".")[1], event['station'].split(".")[2], event['components'], event['time'] - start_buffer, event['time'] + event['duration'] + end_buffer, event_buffer=0, waveform_name=waveform_name, station_name=station_name, providers=providers, user=user, password=password, download=download)
            stream_list.append(stream)
            # find domain and range of traces
            for trace in stream:
                if event['time'] - start_buffer < x_min:
                    x_min = event['time'] - start_buffer
                if event['time'] + event['duration'] + end_buffer > x_max:
                    x_max = event['time'] + event['duration'] + end_buffer
                if np.min(trace.data) < y_min:
                    y_min = np.min(trace.data)
                if np.max(trace.data) > y_max:
                    y_max = np.max(trace.data)
        else:
            msg = "The channels in at least one seismometer differ from those at the other sites (%s)" % event['station']
            warnings.filterwarnings('always', category=UserWarning)
            warnings.warn(msg, UserWarning)
    
    for i in range(0, 2):
        # plot waveforms
        trace_id = []
        for j in range(0, len(stream_list)):
            if j%2 == 0:
                linestyle = '-'
            else:
                linestyle = '--'
            channels = []
            for trace in stream_list[j]:
                channels.append(trace.id)
                # modify names to display only channel
                if i == 1:
                    trace.id = '...'+trace.id.split(".")[3]
            trace_id.append(channels)
            warnings.filterwarnings('ignore', category=Warning)
            if len(events_df) > 1:
                stream_list[j].plot(fig=fig, color=new_colormap(j/(len(events_df) - 1)), linewidth=0.75, linestyle=linestyle)
            else:
                stream_list[j].plot(fig=fig, color=new_colormap(0), linewidth=0.75, linestyle=linestyle)

        # modify axes limits
        allaxes = fig.get_axes()
        for ax in allaxes:
            ax.set_xlim([datetime.datetime.fromtimestamp(x_min.timestamp, pytz.UTC), datetime.datetime.fromtimestamp(x_max.timestamp, pytz.UTC)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.set_ylim([y_min - 0.05*(y_max - y_min), y_max + 0.05*(y_max - y_min)])
        
        # add legend labels for each seismometer
        for j in range(0, len(stream_list)):
            if j%2 == 0:
                linestyle = '-'
            else:
                linestyle = '--'
            if len(events_df) > 1:
                allaxes[0].plot(0, 0, color=new_colormap(j/(len(events_df) - 1)), linewidth=0.75, linestyle=linestyle, label=trace_id[j][0].split(".")[0]+'.'+trace_id[j][0].split(".")[1]+'.'+trace_id[j][0].split(".")[2])
            else:
                 allaxes[0].plot(0, 0, color=new_colormap(0), linewidth=0.75, linestyle=linestyle, label=trace_id[j][0].split(".")[0]+'.'+trace_id[j][0].split(".")[1]+'.'+trace_id[j][0].split(".")[2])
        if len(stream_list) > 8:
            allaxes[0].legend(fontsize = 8, loc='upper right')
        else:
            allaxes[0].legend(fontsize = 10, loc='upper right')
        plt.title('')
        
        # add axes labels
        plt.xlabel(r'UTC Time on '+str(x_min.day)+' '+x_min.strftime('%B')+' '+str(x_min.year), fontsize=12)
        
        # find location of bottom left and top right plot in tight layout (doubles computation time)
        if i == 0:
            plt.tight_layout()
            left, bottom, right, top = 1, 1, 0, 0
            for ax in allaxes:
                pos = ax.get_position()
                if pos.x0 < left:
                    left = pos.x0
                if pos.y0 < bottom:
                    bottom = pos.y0
                if pos.x1 > right:
                    right = pos.x1
                if pos.y1 > top:
                    top = pos.y1
                ax.cla()
        if i == 1:
            plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=None, hspace=None)
    
    # create .pdf file
    if filename == None:
        plt.savefig(os.path.abspath(event_id+'_plot.pdf'))
    else:
        plt.savefig(os.path.abspath(filename+'.pdf'))

    # display figure
    plt.show()

# define functions to plot distribution and correlation between attributes
def plot_attributes(attributes, filename=None):
    """
    Box-and-whisker plot of the distribution of values for each attribute derived for the event catalogue using the get_attributes function.
    
    Parameters
    ----------
    attributes : DataFrame
        List of events and their attributes stored as a pandas dataframe including columns named ‘time’ (or ‘ref_time’) and ‘duration’ (or ‘ref_duration’) for the start time and duration of the event.
    filename : str or path, optional
        Specify path to directory and filename to write the .pdf plot. The default filename/path is ‘attribute_plot.pdf’ located in the working directory.
        
    Returns
    -------
    None
    """
    # create new figure
    my_dpi = 150
    fig = plt.figure(figsize=(6, 6), dpi=my_dpi)

    # set font sizes and use of Latex fonts
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # set axis labels and scale
    plt.xlabel('Normalised attribute')
    plt.xlim([-5, 5])
        
    attributes = attributes.copy()
    # scale attributes to gaussian with median of zero and standard deviation corresponding to the 16/84th percentiles
    for i in range(5, len(attributes.columns)):
        for j in range(0, len(attributes.iloc[:, i])):
            attributes.iloc[j, i] = np.mean(attributes.iloc[j, i])
        attributes.iloc[:, i] = (attributes.iloc[:, i] - np.quantile(attributes.iloc[:, i], 0.5))/((np.quantile(attributes.iloc[:, i], 0.84) - np.quantile(attributes.iloc[:, i], 0.16))/2.)
        
    # replace non-Latex friendly characters
    for i in range(5, len(attributes.columns)):
        attributes.rename(columns = {attributes.columns[i]:str(attributes.columns[i]).replace('_', '\_')}, inplace=True)

    # plot boxplot using seaborn
    try:
        sns.boxplot(data=attributes.drop(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration']), orient='h', ax=plt.gca())
    except:
        sns.boxplot(data=attributes.drop(columns=['event_id', 'station', 'components', 'time', 'duration']), orient='h', ax=plt.gca())
    plt.tight_layout()

    # create .pdf file
    if filename == None:
        plt.savefig(os.path.abspath('attribute_plot.pdf'))
    else:
        plt.savefig(os.path.abspath(filename+'.pdf'))
    
    # display figure
    plt.show()

def plot_correlations(attributes, filename=None, plot_type='matrix'):
    """
    Correlation matrix, or alternatively a corner plot, of attributes derived for the event catalogue using the get_attributes function. The correlation matrix is a lower triangular matrix coloured by the value of the correlation coefficient; the numerical value of the Pearson correlation coefficient is also shown in text. The corner plot is also a lower triangular matrix with scatter plots of the correlation between each attribute and bar charts of the distribution of each attribute along the diagonal.
    
    Parameters
    ----------
    attributes : DataFrame
        List of events and their attributes stored as a pandas dataframe including columns named ‘time’ (or ‘ref_time’) and ‘duration’ (or ‘ref_duration’) for the start time and duration of the event.
    filename : str or path, optional
        Specify path to directory and filename to write the .pdf plot. The default filename/path is ‘correlation_plot.pdf’ located in the working directory.
    plot_type : str, optional
        Specify whether correlations are to be plotted as a correlation matrix (’matrix’) or corner (or pair) plot (’corner’). By default a correlation matrix is plotted.
        
    Returns
    -------
    None
    """
    # create new figure
    my_dpi = 150
    fig = plt.figure(figsize=(9, 9), dpi=my_dpi)

    # set font sizes and use of Latex fonts
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    attributes = attributes.copy()
    # replace non-Latex friendly characters
    for i in range(5, len(attributes.columns)):
        for j in range(0, len(attributes.iloc[:, i])):
            attributes.iloc[j, i] = np.mean(attributes.iloc[j, i])
        attributes.iloc[:, i] = pd.to_numeric(attributes.iloc[:, i], downcast='float')
        attributes.rename(columns = {attributes.columns[i]:str(attributes.columns[i]).replace('_', '\_')}, inplace=True)

    if plot_type == 'matrix':
        # create correlation matrix and mask for upper triangular matrix
        try:
            corrMatt = attributes.drop(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration']).corr()
        except:
            corrMatt = attributes.drop(columns=['event_id', 'station', 'components', 'time', 'duration']).corr()
        mask = np.array(corrMatt)
        mask[np.tril_indices_from(mask)] = False
        
        # create correlation plot
        sns.heatmap(corrMatt, mask=mask, vmin=-1, vmax=1, square=True, annot=True, annot_kws={'size': 9}, fmt='.2f', cbar=False)
        plt.tight_layout()
    else:
        warnings.filterwarnings('ignore', category=Warning)
        rc('font', size=0) # stop text from resizing plots
        
        # create pair plot
        try:
            pairs = attributes.drop(columns=['event_id', 'stations', 'network_time', 'ref_time', 'ref_duration'])
        except:
            pairs = attributes.drop(columns=['event_id', 'station', 'components', 'time', 'duration'])
        
        for i in range(0, 2):
            pd.plotting.scatter_matrix(pairs, ax=plt.gca(), c='crimson', hist_kwds={'bins': 10, 'color': 'black', 'alpha': 0.65})
            allaxes = fig.get_axes()
            # remove tickmarks
            for ax in allaxes:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(ax.get_xlabel(), rotation=90, fontsize=12)
                ax.set_ylabel(ax.get_ylabel(), rotation=0, horizontalalignment='right', verticalalignment='center', fontsize=12)
            # find location of bottom left plot in tight layout (doubles computation time)
            if i == 0:
                plt.tight_layout()
                left, bottom = 1, 1
                for ax in allaxes:
                    pos = ax.get_position()
                    if pos.x0 < left:
                        left = pos.x0
                    if pos.y0 < bottom:
                        bottom = pos.y0
                plt.cla()
            # remove upper triangular matrix
            if i == 1:
                j = 0
                sqrt = np.sqrt(len(allaxes))
                for ax in allaxes:
                    if j%sqrt > j//sqrt:
                        ax.cla()
                        ax.axis('off')
                    j = j + 1
        # resize plot to fit window
        plt.subplots_adjust(left=left, bottom=bottom, right=1, top=1, wspace=None, hspace=None)
        
    # create .pdf file
    if filename == None:
        if plot_type == 'matrix':
            plt.savefig(os.path.abspath('correlation_matrix.pdf'))
        else:
            plt.savefig(os.path.abspath('corner_plot.pdf'))
    else:
        plt.savefig(os.path.abspath(filename+'.pdf'))

    # display figure
    plt.show()
