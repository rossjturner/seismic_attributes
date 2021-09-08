#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: tests.py
#  Purpose: Functions to test if the seismic_attributes package performs as expected.
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
Functions to test if the seismic_attributes package performs as expected.

:copyright:
    Ross Turner, 2020-2021
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)
"""

import pytest

# test seismic_attributes package is installed
def test_one():
    try:
        # package installation using pip or Python distutils
        from seismic_attributes import seismic_attributes as sa
    except:
        # package copied into directory
        import seismic_attributes as sa
    return sa
        
# test import of obspy packages works
def test_two():
    sa = test_one()
    t1 = sa.UTCDateTime("2018-01-01T02:46:30.0Z")
    t2 = sa.UTCDateTime("2018-01-01T02:47:30.0Z")
    return t1, t2
    
# test download of data from IRIS client works
def test_three():
    sa = test_one()
    t1, t2 = test_two()
    stream = sa.get_waveforms('DK', 'ILULI', '', ['BHE', 'BHN', 'BHZ'], t1, t2, waveform_name='test_waveforms', station_name='test_stations', event_buffer=60, providers='IRIS', download=True)

# test output from event catalogue is correct
def test_four():
    sa = test_one()
    t1, t2 = test_two()
    stream = sa.get_waveforms('DK', 'ILULI', '', ['BHE', 'BHN', 'BHZ'], t1, t2, waveform_name='test_waveforms', station_name='test_stations', event_buffer=60, download=False)
    events = sa.get_events(stream, t1, t2, station_name='test_stations', trigger_type='recstalta', sta=1, lta=100, thr_on=3, thr_off=1, thr_event_join=5)
    assert events[0]['event_id'][0] == '20180101T024709Z' or events[0]['event_id'][0] == '20180101T024710Z' or events[0]['event_id'][0] == '20180101T024711Z'
    return events, stream

# test output from attribute catalogue is correct
def test_five():
    sa = test_one()
    events, stream = test_four()
    attributes = sa.get_attributes(events, stream, sa.polarity_attributes)
    assert abs(attributes['attribute_70'][0] - 37.731470226460914) < 0.001
    return attributes

# test waveform plotting function works
def test_six():
    sa = test_one()
    events, stream = test_four()
    sa.plot_waveforms(events, events[0]['event_id'][0], start_buffer=15, end_buffer=30, filename='test_plot', waveform_name='test_waveforms', station_name='test_stations', download=False)
