from setuptools import setup

setup(
    name='seismic_attributes',
    version='1.0.1',
    author = "Ross Turner",
    author_email = "turner.rj@icloud.com",
    description = ("An ObsPy library for event detection and seismic attribute calculation: preparing waveforms for automated analysis."),
    packages=['seismic_attributes'],
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'obspy', 'pytz', 'seaborn'
    ],
)
