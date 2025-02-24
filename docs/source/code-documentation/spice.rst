.. _spice:

SPICE
==========

.. currentmodule:: imap_processing.spice

This is the SPICE module. It contains code that uses NASA's Navigation and Ancillary Information
Facility (`NAIF <https://naif.jpl.nasa.gov/naif/index.html>`_) `SPICE (Spacecraft, Planet,
Instrument, C-matrix, Events) Toolkit <https://naif.jpl.nasa.gov/naif/toolkit.html>`_ to help
interpret IMAP science data. `imap_processing` does not directly interact with the SPICE Toolkit,
but leverages the `spiceypy <https://github.com/AndrewAnnex/SpiceyPy>`_ python package which
wraps the SPICE C library.

The `imap_processing` spice module contains functions for kernel management, time conversions,
calculation of spacecraft and instrument geometry, and utilities for querying spin and pointing
table data.

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    kernels
    geometry
    spin
    time
