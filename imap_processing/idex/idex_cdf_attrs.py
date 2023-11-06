import dataclasses

from imap_processing.cdfutils.global_base import (
    Constants,
    DataLevelBase,
    FloatBase,
    InstrumentBase,
    IntBase,
    ScienceBase,
)
from imap_processing.idex import __version__

# Valid min/maxes

# Data is in a 12 bit unsigned INT
DATA_MIN = 0  # It could go down to 0 in theory
DATA_MAX = 4096  # It cannot exceed 4096 (2^12)

# Samples span 130 microseconds at the most, and values are allowed to be negative
SAMPLE_RATE_MIN = -130  # All might be negative
SAMPLE_RATE_MAX = 130  # All might be positive

# Global Attributes
text = (
    "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
    "dust impact ionization mass spectrometer on the IMAP mission that "
    "provides the elemental composition, speed, and mass distributions "
    "of interstellar dust and interplanetary dust particles. Each record "
    "contains the data from a single dust impact. See "
    "https://imap.princeton.edu/instruments/idex for more details."
)

idex_base = InstrumentBase(__version__, "IDEX>Interstellar Dust Experiment", text)
# idex_global_base = {
#     "Data_type": "L1>Level-1",
#     "Data_version": __version__,
#     "Descriptor": "IDEX>Interstellar Dust Experiment",
#     "TEXT": (
#         "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
#         "dust impact ionization mass spectrometer on the IMAP mission that "
#         "provides the elemental composition, speed, and mass distributions "
#         "of interstellar dust and interplanetary dust particles. Each record "
#         "contains the data from a single dust impact. See "
#         "https://imap.princeton.edu/instruments/idex for more details."
#     ),
#     "Logical_file_id": "FILL ME IN AT FILE CREATION",
# } | GlobalBase.global_base

idex_l1_global_attrs = DataLevelBase(
    "L1>Level-1",
    "imap_idex_l1",
    "IMAP Mission IDEX Instrument Level-1 Data.",
    idex_base,
)

idex_l2_global_attrs = DataLevelBase(
    "L2>Level-2", "imap_idex_l2", "IMAP Mission IDEX Instrument Level-2 Data", idex_base
)

l1_data_base = ScienceBase(
    DATA_MIN,
    DATA_MAX,
    display_type="spectrogram",
    depend_0="Epoch",
    format="I12",
    units="dN",
    var_type="data",
    variable_purpose="PRIMARY",
)

# L1 variables base dictionaries
# (these need to be filled in by the variable dictionaries below)
# l1_data_base = {
#     "DEPEND_0": "Epoch",
#     "DISPLAY_TYPE": "spectrogram",
#     "FILLVAL": Constants.INT_FILLVAL,
#     "FORMAT": "I12",
#     "UNITS": "dN",
#     "VALIDMIN": DATA_MIN,
#     "VALIDMAX": DATA_MAX,
#     "VAR_TYPE": "data",
#     "SCALETYP": "linear",
#     # "VARIABLE_PURPOSE" tells CDAWeb which variables are worth plotting
#     "VARIABLE_PURPOSE": "PRIMARY",
# }

l1_tof_base = dataclasses.replace(l1_data_base, depend_1="Time_High_SR")

l1_target_base = dataclasses.replace(l1_data_base, depend_1="Time_Low_SR")

sample_rate_base = FloatBase(
    SAMPLE_RATE_MIN,
    SAMPLE_RATE_MAX,
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
trigger_base = IntBase(
    0,
    Constants.INT_MAXVAL,
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
