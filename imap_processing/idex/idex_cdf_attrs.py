"""Contains CDF attribute definitions for IDEX."""

import dataclasses

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    FloatAttrs,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.idex import __version__
from imap_processing.idex.idex_constants import IdexConstants

# Global Attributes
text = (
    "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
    "dust impact ionization mass spectrometer on the IMAP mission that "
    "provides the elemental composition, speed, and mass distributions "
    "of interstellar dust and interplanetary dust particles. Each record "
    "contains the data from a single dust impact. See "
    "https://imap.princeton.edu/instruments/idex for more details."
)

idex_base = GlobalInstrumentAttrs(
    __version__, "IDEX>Interstellar Dust Experiment", text
)

idex_l1_global_attrs = GlobalDataLevelAttrs(
    "L1>Level-1",
    "imap_idex_l1",
    "IMAP Mission IDEX Instrument Level-1 Data.",
    idex_base,
)

idex_l2_global_attrs = GlobalDataLevelAttrs(
    "L2>Level-2", "imap_idex_l2", "IMAP Mission IDEX Instrument Level-2 Data", idex_base
)

l1_data_base = ScienceAttrs(
    IdexConstants.DATA_MIN,
    IdexConstants.DATA_MAX,
    display_type="spectrogram",
    depend_0="Epoch",
    format="I12",
    units="dN",
    var_type="data",
    variable_purpose="PRIMARY",
)

# L1 variables base dictionaries
# (these need to be filled in by the variable dictionaries below)
l1_tof_base = dataclasses.replace(l1_data_base, depend_1="Time_High_SR")
l1_target_base = dataclasses.replace(l1_data_base, depend_1="Time_Low_SR")

sample_rate_base = FloatAttrs(
    IdexConstants.SAMPLE_RATE_MIN,
    IdexConstants.SAMPLE_RATE_MAX,
    depend_0="Epoch",
    format="F12.5",
    label_axis="Time",
    units="microseconds",
    var_notes=(
        "The number of microseconds since the event.  "
        "0 is the start of data collection, negative "
        "numbers represent data collected prior to a dust event"
    ),
)

# TODO: is "VALIDMAX" not required here? Added INT_MAXVAL, incl the existing CDF for ref
trigger_base = ScienceAttrs(
    0,
    GlobalConstants.INT_MAXVAL,
    depend_0="Epoch",
    format="I12",
    var_type="data",
    display_type="no_plot",
)
#     {
#     "DEPEND_0": "Epoch",
#     "FILLVAL": Constants.INT_FILLVAL,
#     "FORMAT": "I12",
#     "VALIDMIN": 0,  # All values are positive integers or 0 by design
#     "VAR_TYPE": "data",
#     "DISPLAY_TYPE": "no_plot",
# }

# L1 Attribute Dictionaries
low_sr_attrs = dataclasses.replace(
    sample_rate_base,
    catdesc="Low sample rate time steps for a dust " "event.",
    fieldname="Low Sample Rate Time",
    var_notes=(
        "The low sample rate in microseconds. Steps are approximately 1/4.025 "
        "microseconds in duration. Used by the Ion_Grid, Target_Low, and "
        "Target_High variables."
    ),
)

high_sr_attrs = dataclasses.replace(
    sample_rate_base,
    catdesc="High sample rate time steps for a dust event.",
    fieldname="High Sample Rate Time",
    var_notes=(
        "The high sample rate in microseconds. Steps are approximately 1/260 "
        "microseconds in duration. Used by the TOF_High, TOF_Mid, and "
        "TOF_Low variables."
    ),
)

tof_high_attrs = dataclasses.replace(
    l1_tof_base,
    catdesc="Time of flight waveform on the high-gain channel",
    fieldname="High Gain Time of Flight",
    label_axis="TOF High Ampl.",
    var_notes=(
        "High gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
)

tof_mid_attrs = dataclasses.replace(
    l1_tof_base,
    catdesc="Time of flight waveform on the mid-gain channel",
    fieldname="Mid Gain Time of Flight",
    label_axis="TOF Mid Ampl.",
    var_notes=(
        "Mid gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
)

tof_low_attrs = dataclasses.replace(
    l1_tof_base,
    catdesc="Time of flight waveform on the low-gain channel",
    fieldname="Low Gain Time of Flight",
    label_axis="TOF Low Ampl.",
    var_notes=(
        "Low gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
)

target_low_attrs = dataclasses.replace(
    l1_target_base,
    catdesc="Target low charge sensitive amplifier waveform",
    fieldname="Low Target Signal",
    label_axis="Low Target Ampl.",
    var_notes=(
        "Low gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge. "
    ),
)

target_high_attrs = dataclasses.replace(
    l1_target_base,
    catdesc="Ion grid charge sensitive amplifier waveform",
    fieldname="High Target Signal",
    label_axis="High Target Ampl.",
    var_notes=(
        "High gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
)

ion_grid_attrs = dataclasses.replace(
    l1_target_base,
    catdesc="Ion grid charge sensitive amplifier waveform data",
    fieldname="Ion Grid Signal",
    label_axis="Ion Grid Ampl.",
    var_notes=(
        "This is the ion grid signal from IDEX. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
)
