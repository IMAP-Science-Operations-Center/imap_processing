.. _codice:

CoDICE
======

.. currentmodule:: imap_processing.codice

This is the CoDICE (Compact Dual Ion Composition Experiment) Instrument module,
which contains the code for processing data from the CoDICE instrument.

The processing code to decommutate the CCSDS packets (L0) and create L1a data
products can be found below:

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    codice_l0
    codice_l1a

The modules below contain various utility classes and functions to support L0
and L1a processing:

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    utils
    decompress
