.. _swe:

SWE
====

.. currentmodule:: imap_processing.swe

This is the SWE (Solar Wind Electrons) Instrument module, which contains the code for processing
data from the SWE instrument.

The L1A code to unpack electron counts can be found below.

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    l1a.swe_l1a
    l1a.swe_science

The L1B code to process electron counts to rates and then
organize data by voltage step and spin angle can be
found below.

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    l1b.swe_l1b
    l1b.swe_l1b_science

The modules below contains various utility classes and functions used for processing SWE data.

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    utils.swe_utils
