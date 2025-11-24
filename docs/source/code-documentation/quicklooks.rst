Quicklook Generation
===========================

This document provides a high-level overview of the workflow and usage
of the ``QuicklookGenerator`` system. It is intended to help developers
understand how quicklook plots are produced from IMAP instrument data files.

Overview
--------

Each instrument implements its own plotting logic
while sharing a common initialization, data-loading,
and dispatch workflow.

The system is built around four major components:

1. **Dataset loading** – converting a CDF file into an ``xarray.Dataset``.
2. **Instrument detection** – determining the correct quicklook
   generator class.
3. **Abstract quicklook interface** – common API for all instruments.
4. **Instrument-specific subclasses** – actual plotting implementations.

Workflow
--------

1. **User supplies a CDF file path**::

       quicklook = get_instrument_quicklook("path/to/file.cdf")

2. **The filename is parsed** using
   ``ScienceFilePath.extract_filename_components`` to extract the
   ``instrument`` field.

3. **The correct Quicklook subclass is selected** via the
   ``QuicklookGeneratorType`` enum. For example::

       MAG  → MagQuicklookGenerator

   The pattern continues for each individual instrument.

4. **The chosen class is instantiated**, and the constructor:

   - Loads the dataset using ``dataset_into_xarray``.
   - Stores instrument metadata (such as the instrument name).
   - Initializes plotting-related attributes.

5. **The user calls** ``two_dimensional_plot(variable="...")``.
   This method is implemented by each subclass and acts as a routing
   mechanism that decides *which* quicklook plot to generate.

   Example for MAG::

       mag_ql = MagQuicklookGenerator("imap_mag_20250101_v01.cdf")
       mag_ql.two_dimensional_plot(variable="mag sensor co-ord")

6. **The requested plot function runs**, accessing data from the internal
   ``xarray.Dataset`` and generating figures using Matplotlib.

7. **Plots are displayed**, and the workflow ends. No data is returned—
   the quicklook system produces visualizations only.

Core Components
---------------

Abstract Base Class: ``QuicklookGenerator``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``QuicklookGenerator`` abstract base class provides:

- Unified dataset loading via ``dataset_into_xarray``.
- Common metadata attributes (instrument, title, axis labels, etc.).
- Basic validation to ensure that a dataset is present.
- An abstract method ``two_dimensional_plot``, which each instrument
  subclass must implement as a dispatch mechanism.

Time & Coordinate Handling
--------------------------

Instrument data often stores time in **J2000 nanoseconds**. These values
are converted to UTC timestamps using::

    convert_j2000_to_utc(time_array)

This conversion internally uses:

- A fixed epoch: ``2000-01-01 11:58:55.816``
- NumPy ``timedelta64`` arithmetic
- Output as ``datetime64[ns]`` UTC timestamps

This routine is used across all instruments providing an ``epoch`` field.

Instrument Dispatch Logic
-------------------------

Users should begin quicklook generation by calling::

    quicklook = get_instrument_quicklook(filename)

This function:

1. Extracts the instrument identifier from the filename.
2. Uses ``QuicklookGeneratorType`` to map the instrument to a class.
3. Returns an instantiated quicklook object.

Example::

    >>> ql = get_instrument_quicklook("imap_swapi_20250101_v02.cdf")
    >>> ql.two_dimensional_plot("count rates")

Adding Support for New Instruments
----------------------------------

To extend the quicklook system:

1. Create a new subclass::

       class NewInstQuicklookGenerator(QuicklookGenerator):
           ...

2. Implement ``two_dimensional_plot`` as a dispatch method.
3. Add plot functions as needed.
4. Register the class in the ``QuicklookGeneratorType`` enum.
5. Ensure that ``ScienceFilePath.extract_filename_components`` correctly
   identifies the instrument.

Instrument Team Support
-----------------------

To implement a new instrument-specific quicklook, instrument teams must
provide a minimal set of information that defines what plots are
required and how the underlying data should be interpreted.

Required Information
~~~~~~~~~~~~~~~~~~~~

1. **List of quicklook plots**
   - A high-level description of each plot to be generated.
   - Whether each plot is 1-D (line), 2-D (spectrogram), or multi-panel.

2. **Variables required for each plot**
   - CDF variable names.
   - Which CDF files contain the required variables.

3. **Time-axis requirements (if applicable)**
   - Desired time range (full file, event-based selection, rolling window).

4. **Plot formatting preferences**
   - Preferred units or scaling (linear, log).
   - Desired titles, axis labels, and annotations.

5. **Special processing rules**
   - Required calibrations or unit conversions.
   - Masking rules for invalid ranges or quality flags.
   - Any filtering or smoothing applied before plotting.
