# An ObsPy library for event detection and seismic attribute calculation: preparing waveforms for automated analysis 
[![DOI](https://zenodo.org/badge/303865623.svg)](https://zenodo.org/badge/latestdoi/303865623)

This package is an extension for the observational seismology _obspy_ software package that provides a streamlined tool tailored to the processing of seismic signals from non-earthquake sources, in particular those from deforming systems such as glaciers and landslides. This _seismic attributes_ library provides functionality to: (1) download and/or pre-process seismic waveform data; (2) detect and catalogue seismic events using multi-component signals from one or more seismometers; and (3) calculate characteristics ('attributes'/'features') of the identified events. The workflow is controlled by three main functions that have been tested for the breadth of data types expected from permanent and campaign-deployed seismic instrumentation. A selected STA/LTA-type (short-term average/long-term average), or other, event detection algorithm can be applied to the waveforms and user-defined functions implemented to calculate any required characteristics of the detected events. The code is written in Python 2/3 and has detailed documentation and worked examples available on GitHub (https://github.com/rossjturner/seismic_attributes).

## Installation

This package can either be installed using _pip_ or from a .zip file downloaded from the GitHub repository using the standard Python package _distutils_.

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

## Unit testing

The successful installation of the package can be verified using the Python package _pytest_. The unit tests require internet access to run as data is downloaded from the Incorporated Research Institutions for Seismology (IRIS) server; downloaded data will be saved in the working directory.

### Before the test

Find the path to the installed _seismic_attributes_ package. If the package was installed using _pip_ this is achieved as follows:

```bash
pip show seismic_attributes
```

The path to the _seismic_attributes_ directory is shown next to 'Location:'.

### Running the test

The unit test is run with _pytest_ (which should be installed with _seismic_attributes_ by default) using the following command:

```bash
pytest path_to_directory/seismic_attributes
```

The unit test takes approximately 10-20 seconds to run. Five tests are conducted which verify the _seismic_attributes_ package is correctly linked with _obspy_, can download data from the IRIS server, produces an event catalogue with the expected output, and calculates attributes for events as expected.

## Documentation and Examples

Full documentation of the functions included in the seismic_attributes package, in addition to worked examples, is included in [seismic_attributes_user.pdf](https://github.com/rossjturner/seismic_attributes/blob/main/seismic_attributes_user.pdf) on the GitHub repository. The worked examples are additionally included in the following Jupyter notebook: [seismic_attributes_example.ipynb](https://github.com/rossjturner/seismic_attributes/blob/main/seismic_attributes_example.ipynb).

## Contact

Ross Turner <<turner.rj@icloud.com>>

