# seismic_attributes

This package is an extension for the observational seismology obspy software package that provides a streamlined tool tailored to the processing of seismic signals from non-earthquake sources, in particular those from deforming systems such as glaciers and landslides. This seismic attributes library provides functionality to: (1) download and/or pre-process seismic waveform data; (2) detect and catalogue seis- mic events using multi-component signals from one or more seismometers; and (3) calculate characteristics ('attributes'/'features') of the identified events. The workflow is controlled by three main functions that have been tested for the breadth of data types expected from permanent and campaign-deployed seismic instrumentation. A selected STA/LTA-type (short-term average/long-term average), or other, event detection algorithm can be applied to the waveforms and user-defined functions implemented to calculate any required characteristics of the detected events. The code is written in Python 2/3 and has detailed documentation and worked examples available on GitHub (https://github.com/rossjturner/seismic_attributes).

## Installation

This package can either be installed using _pip_ or from a .zip file downloaded from the GitHub repository using the standard Python distutils.

### Install using pip
The following command will install the latest version of the _seismic attributes_ library from the Python Packaging Index (PyPI):

```bash
pip install seismic_attributes
```

### Install from GitHub repository

The package can be downloaded from the GitHub repository at https://github.com/rossjturner/seismic_attributes, or cloned with _git_ using:

```bash
git clone https://github.com/rossjturner/seismic_attributes.git
```

The package is installed by running the following command as an administrative user:

```bash
python setup.py install
```

## Contact

Ross Turner <<turner.rj@icloud.com>>

