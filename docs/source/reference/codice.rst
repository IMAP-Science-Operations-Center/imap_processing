.. _codice:

CoDICE (Compact Dual Ion Composition Experiment)
================================================

.. currentmodule:: imap_processing.codice

This is the CoDICE Instrument module, which contains the code for processing
data from the CoDICE instrument.

The L0 code to decommutate the CCSDS packet data can be found below:

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    l0.decom_codice

The code below is used to decompress an 8 bit science value to a 24- or 32-bit
uncompressed value:

.. autosummary::
    :toctree: generated/
    :template: autosummary.rst
    :recursive:

    l0.decompress_codice
