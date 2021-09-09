from setuptools import setup, Extension

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='seismic_attributes',
    version='1.0.3-1',
    author = "Ross Turner",
    author_email = "turner.rj@icloud.com",
    description = ("An ObsPy library for event detection and seismic attribute calculation: preparing waveforms for automated analysis."),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages=['seismic_attributes'],
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'obspy', 'pytz', 'seaborn', 'pytest'
    ],
)
