# seismic_attributes module
# Ross Turner, 25 September 2020

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys, os, obspy, warnings
import seaborn as sns
from functools import partial
from obspy import read, Trace, Stream, UTCDateTime
from obspy.clients.fdsn.mass_downloader import RectangularDomain, Restrictions, MassDownloader
from obspy.signal.trigger import coincidence_trigger
from obspy.signal.cross_correlation import correlate
from obspy.signal.polarization import flinn
from obspy.signal.freqattributes import spectrum
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import cpu_count, Pool

# define constants
__chunklength_in_sec = 86400

## define functions to download or find already downloaded waveforms
def get_waveforms(network, station, location, channel, starttime, endtime, event_buffer=3600, waveform_name='waveforms', station_name='stations', client=['IRIS', 'LMU', 'GFZ'], download=True):
        
    # create empty stream to store amplitude of three component waveform
    stream = Stream()

    # read-in waveform data from downloaded files
    if isinstance(channel, list):
        for i in range(0, len(channel)):
            if i == 0:
                stream = __get_waveforms(network, station, location, channel[0], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, provider=client, download=download)
            else:
                stream += __get_waveforms(network, station, location, channel[i], starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, provider=client, download=download)
    else:
        stream = __get_waveforms(network, station, location, channel, starttime - event_buffer, endtime + event_buffer, waveform_name=waveform_name, station_name=station_name, provider=client, download=download)

    # merge different days in the stream at the same seismograph and channel
    stream.merge(method=0, fill_value=None)
    # sort channels into ZNE order for use in some obspy functions (only used for some attributes)
    stream.sort(keys=['network', 'station', 'location', 'channel'], reverse=True)

    print(stream)
    return stream

def __get_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', provider=['IRIS', 'LMU', 'GFZ'], download=True):

    # create empty stream to store waveform
    stream = Stream()

    # set start and end time of each file; these start and end on calendar dates
    start_time = UTCDateTime(t1.year, t1.month, t1.day)
    end_time = start_time + __chunklength_in_sec

    # read-in waveform data from downloaded files
    while (start_time < t2):
        filename = waveform_name+'/'+network+'.'+station+'.'+location+'.'+channel+'__'+start_time.strftime("%Y%m%dT%H%M%SZ")+'__'+end_time.strftime("%Y%m%dT%H%M%SZ")+'.mseed'

        # if file exists add to stream
        if os.path.isfile(filename):
            stream += read(filename)
        # otherwise attempt to download file then read-in if data exists
        else:
            if download == True:
                __download_waveforms(network, station, location, channel, start_time, end_time, waveform_name=waveform_name, station_name=station_name, provider=provider)
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

def __download_waveforms(network, station, location, channel, t1, t2, waveform_name='waveforms', station_name='stations', provider=['IRIS', 'LMU', 'GFZ']):

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
    if isinstance(provider, list):
        mdl = MassDownloader(providers=provider)
    else:
        mdl = MassDownloader(providers=[provider])
    mdl.download(domain, restrictions, mseed_storage=waveform_name, stationxml_storage=station_name)


# define functions to find events in a single seismometer or coincident events across multiple seismometers based on the quadrature sum of their components
def get_events(stream, starttime, endtime, signal_type='amplitude', trigger_type='recstalta', thr_event_join=0.5, thr_travel_time=0.01, thr_coincidence_sum=-1, thr_on=5, thr_off=1, **options):

    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    # and with components added in quadrature (i.e. energy)
    stream_list = group_components(component_list, signal_type=signal_type)

    # find list of events for each seismometer based on the stream of amplitudes/energies
    events_list = []
    for i in range(0, len(stream_list)):
        # trigger events using specified event detection algorithm
        events = coincidence_trigger(trigger_type=trigger_type, thr_on=thr_on, thr_off=thr_off, stream=stream_list[i], thr_coincidence_sum=1, details=True, **options)
        
        # create reporting functions for input into the conicidence triggering algorithms to join events over small gaps
        if thr_event_join > 0:
            # lengthen all events
            report_stream = __reporting_functions(stream_list, events, thr_event_join)
            events = coincidence_trigger(trigger_type='classicstalta', thr_on=1.001, thr_off=1, stream=report_stream, thr_coincidence_sum=1, details=True, sta=1./report_stream[0].stats.sampling_rate, lta=3./report_stream[0].stats.sampling_rate)
            # shorten joined events (i.e. start and end times will match original)
            report_stream = __reporting_functions(stream_list, events, -thr_event_join)
            events = coincidence_trigger(trigger_type='classicstalta', thr_on=1.001, thr_off=1, stream=report_stream, thr_coincidence_sum=1, details=True, sta=1./report_stream[0].stats.sampling_rate, lta=3./report_stream[0].stats.sampling_rate)
            
        # append event list to file
        events_list.append(events)

    # find list of events for array of seismometers based on the event catalogues
    if len(events_list) > 1:
        # create reporting functions for input into the conicidence triggering algorithms to events triggered at multiple seismometers
        report_stream = __reporting_functions(stream_list, events_list, max(0.01, thr_travel_time))

        # apply coincidence triggering to reporting functions
        if thr_coincidence_sum <= 0:
            coincident_events = coincidence_trigger(trigger_type='classicstalta', thr_on=1.001, thr_off=1, stream=report_stream, thr_coincidence_sum=len(events_list), details=True, sta=1./report_stream[0].stats.sampling_rate, lta=3./report_stream[0].stats.sampling_rate)
        else:
            coincident_events = coincidence_trigger(trigger_type='classicstalta', thr_on=1.001, thr_off=1, stream=report_stream, thr_coincidence_sum=min(len(events_list), thr_coincidence_sum), details=True, sta=1./report_stream[0].stats.sampling_rate, lta=3./report_stream[0].stats.sampling_rate)
            
            # if not all seismographs need an event we need to combine the coincident events from different sets of seismographs
            if len(events_list) > thr_coincidence_sum:
                # create new reporting functions to merge events from coincident trigger function
                report_stream = __reporting_functions(stream_list, coincident_events, 0.01)
                coincident_events = coincidence_trigger(trigger_type='classicstalta', thr_on=1.001, thr_off=1, stream=report_stream, thr_coincidence_sum=1, details=True, sta=1./report_stream[0].stats.sampling_rate, lta=3./report_stream[0].stats.sampling_rate)
        
    else:
        # single set of events; no coincidence triggering
        coincident_events = events_list[0]
    
    # output relevant columns to pandas dataframe
    if len(coincident_events) > 0:
        events_df = pd.DataFrame(coincident_events)[['time', 'duration']]
    else:
        raise Exception('Start/end time and stream are for different time periods.')

    # remove events outside requested time window and less than 10 times the sampling rate
    events_df = events_df[np.logical_and(events_df['time'] + events_df['duration'] > starttime, events_df['time'] < endtime)]
    events_df = events_df[events_df['duration'] > 10./stream[0].stats.sampling_rate]
    events_df.reset_index(drop=True, inplace=True)
    
    # add column to store signal_type used in triggering
    if signal_type == 'amplitude':
        events_df['signal_type'] = 'amplitude'
    else:
        events_df['signal_type'] = 'energy'

    return events_df
    
def __reporting_functions(stream_list, events_list, thr_event_offset):

    # create reporting functions with one everywhere except at events
    k = 0
    for i in range(0, len(stream_list)):
        for j in range(0, len(stream_list[i])):
            if i == 0 and j == 0:
                report_stream = Stream(stream_list[0][0].copy())
                report_stream[0].data = (stream_list[0][0].data)*0.0 + 1
            else:
                report_stream += Stream(stream_list[i][j].copy())
                report_stream[-1].data = (stream_list[i][j].data)*0.0 + 1
                k = k + 1 # update number of traces

            # add linear functions at times with events
            if len(events_list) > 0 and isinstance(events_list[0], list):
                events = events_list[i]
            else:
                events = events_list
            times = np.asarray(report_stream[k].times())
            for l in range(0, len(events)):
                # find start and end time of event in seconds from start of stream
                start_time = float(events[l]['time']) - float(report_stream[k].stats.starttime) - thr_event_offset/2.
                end_time = float(events[l]['time'] + events[l]['duration']) - float(report_stream[k].stats.starttime) + thr_event_offset/2.
                # find indices of each event
                index = np.searchsorted(times, np.arange(start_time, end_time, 1./report_stream[k].stats.sampling_rate))
                # remove indices outside the range of the stream
                index = np.delete(index, np.where(index == len(times)))
                if np.sum(np.where(index == 1)) > 0 and np.sum(np.where(index == 0)) > 0: # keep true zero event
                    index = np.delete(index, np.where(index == 0))
                    index = np.append([0], index)
                else:
                    index = np.delete(index, np.where(index == 0))
                # set these indices to linear function
                if len(index) > 0:
                    report_stream[k].data[index] = times[index]/100. + 1
    
    return report_stream

def group_components(component_list, signal_type='amplitude'):
    
    # convert stream to a list if it is a Stream object for a single seismometer
    if not isinstance(component_list, (list, np.ndarray)):
        component_list = [component_list]
    
    # create lists to store streams for each seismometer (if applicable)
    stream_list = [] # total amplitude (or energy)
    
    # combine traces at each seismometer in quadrature as appropriate
    for i in range(0, len(component_list)):
        # find mean of each component at the given seismometer
        mean_list = []
        # calculate weighted mean for the seismometer and component in this trace
        for j in range(0, len(component_list[i])):
            count = len(component_list[i][j].data)
            mean = np.sum(component_list[i][j].data)
            mean_list.append(float(mean)/count)
    
        for j in range(0, len(component_list[i])):
            # create new stream object to store combined components
            if j == 0:
                new_stream = Stream(component_list[i][0].copy())
                new_stream[0].data = (component_list[i][0].data - mean_list[0])**2
                stream_list.append(new_stream)
            else:
                # add additional components to stream in quadrature
                stream_list[i][0].data += (component_list[i][j].data - mean_list[j])**2
            
                # modify trace id to terminate in number of components
                if stream_list[i][0].id[-1].isnumeric() == True:
                    stream_list[i][0].id = stream_list[i][0].id[:-1]+str(int(stream_list[i][0].id[-1]) + 1)
                else:
                    stream_list[i][0].id = stream_list[i][0].id[:-1]+'2'
                
        # if requested output is amplitude convert data to amplitudes
        if signal_type == 'amplitude':
            stream_list[i][0].data = np.sqrt(stream_list[i][0].data)
        
    return stream_list
    
def __group_seismometers(stream):

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


# define functions to plot identified events in waveform data
def plot_events(events, stream, starttime, endtime, filename=None):

    warnings.filterwarnings('ignore', category=Warning)

    # set start and end time of each calendar day
    start_of_day = UTCDateTime(starttime.year, starttime.month, starttime.day)
    start_time = starttime
    end_time = min(endtime, start_of_day + __chunklength_in_sec)

    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)
    # and with components added in quadrature (i.e. energy); 'amplitude' is very computationally intensive
    stream_list = group_components(component_list, signal_type=events['signal_type'][0])
    new_stream = stream_list[0]
    print(new_stream)

    # create empty .pdf file
    if filename == None:
        pdf = PdfPages(new_stream[0].stats.station+'_event_plot.pdf')
    else:
        pdf = PdfPages(filename+'.pdf')

    # create stream including only times identified as events
    event_stream = Stream()
    for i in range(0, len(events)):
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
    plt.xlabel('Time in minutes', fontsize=12)
    if signal_type == 'amplitude':
        plt.ylabel('Amplitude at '+str(stream[0].stats.station)+' on '+str(start_time.day)+' '+start_time.strftime('%B')+' '+str(start_time.year), fontsize=12, labelpad=6)
    else:
        plt.ylabel('Energy at '+str(stream[0].stats.station)+' on '+str(start_time.day)+' '+start_time.strftime('%B')+' '+str(start_time.year), fontsize=12, labelpad=6)
    fig.tight_layout()

    return fig

def __lighten_color(color, alpha=0.25):

    # lighten existing named colour or hex code by alpha; alpha=1 is original and alpha=0 is white
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], 1 - alpha*(1 - c[1]), c[2])


# define functions to calculate attributes of the waveforms
def get_attributes(events, stream, *attributes):

    # create a copy of the input stream separated into streams for each seismometer
    component_list = __group_seismometers(stream)

    # calculate attributes for each event
    for i in range(0, len(events)):
        # find start and end time of event in seconds from start of stream
        start_time = UTCDateTime(events['time'][i])
        end_time = UTCDateTime(events['time'][i] + events['duration'][i])
        
        # find mean of attributes if multiple seismometers are used
        attribute_list, attribute_List = None, None
        k = 0
        for j in range(0, len(component_list)):
            # add waveform data during event to new stream
            component_stream = component_list[j].slice(start_time, end_time)

            # add attribute names and values to lists
            if len(component_stream) > 0:
                name_list = ['start_time', 'stop_time']
                attribute_list = [start_time, end_time]
                for l in range(0, len(attributes)):
                    name, attribute = attributes[l](component_stream, start_time, end_time)
                    if isinstance(attribute, (list, np.ndarray)):
                        name_list.extend(name)
                        attribute_list.extend(attribute)
                    else:
                        name_list.append(name)
                        attribute_list.append(attribute)
            
            # calculate mean of attributes
            if attribute_List == None:
                if attribute_list == None:
                    raise Exception('No events found: check event list and stream are for different time periods.')
                else:
                    attribute_List = attribute_list
                    k = k + 1
            else:
                attribute_List[2:] = np.asarray(attribute_List[2:])*float(k)/float(k + 1) + np.asarray(attribute_list[2:])/float(k + 1)
                k = k + 1
        
        # create pandas dataframe to store attributes
        if i == 0:
            attributes_df = pd.DataFrame(columns=name_list)
        # append attributes for each event to dataframe
        attributes_df.loc[len(attributes_df)] = attribute_List
        
    return attributes_df

def attribute_1(component_stream, event_start, event_stop):
    # Duration
    return 'attribute_1', event_stop - event_start
    
def attribute_4(component_stream, event_start, event_stop):
    # Ratio between ascending and descending time; use only first seismometer in list
    trace = group_components(component_stream, signal_type='energy')[0][0]
    event_max = trace.times()[np.argmax(trace.data)]

    return 'attribute_4', (event_max)/np.maximum(1e-9, (event_stop - event_start) - event_max)
    
def attribute_10_11_12(component_stream, event_start, event_stop):
    # Energy in the first third and the remaining part of the autocorrelation function; use only first seismometer in list
    trace = group_components(component_stream, signal_type='energy')[0][0]
    acf_third = np.sum(correlate(trace, trace, int(len(trace.data)/3))[int(len(trace.data)/3):-1])*trace.stats.delta
    acf_total = np.sum(correlate(trace, trace, int(len(trace.data)))[int(len(trace.data)):-1])*trace.stats.delta

    return ['attribute_10', 'attribute_11', 'attribute_12'], [acf_third, acf_total - acf_third, (acf_total - acf_third)/np.maximum(1e-9, acf_third)]
    
def attribute_13_14_15_17(component_stream, event_start, event_stop):
    # Energy of the signal filtered in 5 – 10 Hz, 10 – 50 Hz, 5 – 70 Hz, 1 - 20 Hz
    band_13 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=10), signal_type='energy'))*component_stream[0].stats.delta
    band_14 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=10, freqmax=50), signal_type='energy'))*component_stream[0].stats.delta
    band_15 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=5, freqmax=70), signal_type='energy'))*component_stream[0].stats.delta
    band_17 = np.sum(group_components((component_stream.copy()).filter("bandpass", freqmin=1, freqmax=20), signal_type='energy'))*component_stream[0].stats.delta
    
    return ['attribute_13', 'attribute_14', 'attribute_15', 'attribute_17'], [band_13, band_14, band_15, band_17]

def attribute_68_69_70_71(component_stream, event_start, event_stop):
    # Rectilinearity, azimuth, dip and planarity
    azimuth, incidence, rectillinearity, planarity = flinn(component_stream)

    return ['attribute_68', 'attribute_69', 'attribute_70', 'attribute_71'], [rectillinearity, azimuth, incidence, planarity]


# define functions to plot distribution and correlation between attributes
def plot_attributes(attributes, attribute_scale=None, filename=None):

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
    # determine if any attributes should have logarithmic scaling
    if not attribute_scale == None:
        if isinstance(attribute_scale, str):
            try:
                if (np.min(attributes[attribute_scale]) > 0):
                    attributes[attribute_scale] = np.log10(attributes[attribute_scale])
                else:
                    print(attribute_scale+' is has non-positive values.')
            except:
                print(attribute_scale+' is not an attribute.')
        else:
            for i in range(0, len(attribute_scale)):
                try:
                    if (np.min(attributes[attribute_scale[i]]) > 0):
                        attributes[attribute_scale[i]] = np.log10(attributes[attribute_scale[i]])
                    else:
                        print(attribute_scale[i]+' is has non-positive values.')
                except:
                    print(attribute_scale[i]+' is not an attribute.')
            
    # scale attributes to gaussian with median of zero and standard deviation corresponding to the 16/84th percentiles
    for i in range(2, len(attributes.columns)):
        attributes.iloc[:, i] = (attributes.iloc[:, i] - np.quantile(attributes.iloc[:, i], 0.5))/((np.quantile(attributes.iloc[:, i], 0.84) - np.quantile(attributes.iloc[:, i], 0.16))/2.)
        
    # replace non-Latex friendly characters
    for i in range(2, len(attributes.columns)):
        attributes.rename(columns = {attributes.columns[i]:str(attributes.columns[i]).replace('_', '\_')}, inplace=True)

    # plot boxplot using seaborn
    sns.boxplot(data=attributes.drop(columns=['start_time', 'stop_time']), orient='h', ax=plt.gca())
    plt.tight_layout()

    # create .pdf file
    if filename == None:
        plt.savefig('attribute_plot.pdf')
    else:
        plt.savefig(filename+'.pdf')

def plot_correlations(attributes, attribute_scale=None, filename=None):

    # create new figure
    my_dpi = 150
    fig = plt.figure(figsize=(9, 9), dpi=my_dpi)

    # set font sizes and use of Latex fonts
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    attributes = attributes.copy()
    # determine if any attributes should have logarithmic scaling
    if not attribute_scale == None:
        if isinstance(attribute_scale, str):
            try:
                if (np.min(attributes[attribute_scale]) > 0):
                    attributes[attribute_scale] = np.log10(attributes[attribute_scale])
                else:
                    print(attribute_scale+' is has non-positive values.')
            except:
                print(attribute_scale+' is not an attribute.')
        else:
            for i in range(0, len(attribute_scale)):
                try:
                    if (np.min(attributes[attribute_scale[i]]) > 0):
                        attributes[attribute_scale[i]] = np.log10(attributes[attribute_scale[i]])
                    else:
                        print(attribute_scale[i]+' is has non-positive values.')
                except:
                    print(attribute_scale[i]+' is not an attribute.')

    # replace non-Latex friendly characters
    for i in range(2, len(attributes.columns)):
        attributes.rename(columns = {attributes.columns[i]:str(attributes.columns[i]).replace('_', '\_')}, inplace=True)
        
    # create correlation matrix and mask for upper triangular matrix
    corrMatt = attributes.drop(columns=['start_time', 'stop_time']).corr()
    mask = np.array(corrMatt)
    mask[np.tril_indices_from(mask)] = False

    # create correlation plot
    sns.heatmap(corrMatt, mask=mask, vmin=-1, vmax=1, square=True, annot=True, annot_kws={'size': 9}, fmt='.2f', cbar=False)
    plt.tight_layout()

    # create .pdf file
    if filename == None:
        plt.savefig('correlation_plot.pdf')
    else:
        plt.savefig(filename+'.pdf')
