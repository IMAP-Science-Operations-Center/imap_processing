from imap_processing import cdf_utils
from imap_processing.idex import __version__

# Valid min/maxes

# Data is in a 12 bit unsigned INT
DATA_MIN = 0  # It could go down to 0 in theory
DATA_MAX = 4096  # It cannot exceed 4096 (2^12)

# Samples span 130 microseconds at the most, and values are allowed to be negative
SAMPLE_RATE_MIN = -130  # All might be negative
SAMPLE_RATE_MAX = 130  # All might be positive

# Global Attributes
idex_global_base = {
    "Data_type": "L1>Level-1",
    "Data_version": __version__,
    "Descriptor": "IDEX>Interstellar Dust Experiment",
    "TEXT": (
        "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
        "dust impact ionization mass spectrometer on the IMAP mission that "
        "provides the elemental composition, speed, and mass distributions "
        "of interstellar dust and interplanetary dust particles. Each record "
        "contains the data from a single dust impact. See "
        "https://imap.princeton.edu/instruments/idex for more details."
    ),
    "Logical_file_id": "FILL ME IN AT FILE CREATION",
} | cdf_utils.global_base

idex_l1_global_attrs = {
    "Data_type": "L1>Level-1",
    "Logical_source": "imap_idex_l1",
    "Logical_source_description": "IMAP Mission IDEX Instrument Level-1 Data.",
} | idex_global_base

idex_l2_global_attrs = {
    "Data_type": "L2>Level-2",
    "Logical_source": "imap_idex_l2",
    "Logical_source_description": "IMAP Mission IDEX Instrument Level-2 Data",
} | idex_global_base

# L1 variables base dictionaries
# (these need to be filled in by the variable dictionaries below)
l1_data_base = {
    "DEPEND_0": "Epoch",
    "DISPLAY_TYPE": "spectrogram",
    "FILLVAL": cdf_utils.INT_FILLVAL,
    "FORMAT": "I12",
    "UNITS": "dN",
    "VALIDMIN": DATA_MIN,
    "VALIDMAX": DATA_MAX,
    "VAR_TYPE": "data",
    "SCALETYP": "linear",
    # "VARIABLE_PURPOSE" tells CDAWeb which variables are worth plotting
    "VARIABLE_PURPOSE": "PRIMARY",
}

l1_tof_base = {"DEPEND_1": "Time_High_SR"} | l1_data_base

l1_target_base = {"DEPEND_1": "Time_Low_SR"} | l1_data_base

sample_rate_base = {
    "DEPEND_0": "Epoch",
    "FILLVAL": cdf_utils.DOUBLE_FILLVAL,
    "FORMAT": "F12.5",
    "LABLAXIS": "Time",
    "UNITS": "microseconds",
    "VALIDMIN": SAMPLE_RATE_MIN,
    "VALIDMAX": SAMPLE_RATE_MAX,
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
    "VAR_NOTES": (
        "The number of microseconds since the event.  "
        "0 is the start of data collection, negative "
        "numbers represent data collected prior to a dust event"
    ),
}

trigger_base = {
    "DEPEND_0": "Epoch",
    "FILLVAL": cdf_utils.INT_FILLVAL,
    "FORMAT": "I12",
    "VALIDMIN": 0,  # All values are positive integers or 0 by design
    "VAR_TYPE": "data",
    "DISPLAY_TYPE": "no_plot",
}

# L1 Attribute Dictionaries
low_sr_attrs = {
    "CATDESC": "Low sample rate time steps for a dust event.",
    "FIELDNAM": "Low Sample Rate Time",
    "VAR_NOTES": (
        "The low sample rate in microseconds. "
        "Steps are approximately 1/4.025 microseconds in duration. "
        "Used by the Ion_Grid, Target_Low, and Target_High variables."
    ),
} | sample_rate_base

high_sr_attrs = {
    "CATDESC": "High sample rate time steps for a dust event.",
    "FIELDNAM": "High Sample Rate Time",
    "VAR_NOTES": (
        "The high sample rate in microseconds. "
        "Steps are approximately 1/260 microseconds in duration. "
        "Used by the TOF_High, TOF_Mid, and TOF_Low variables."
    ),
} | sample_rate_base

tof_high_attrs = {
    "CATDESC": "Time of flight waveform on the high-gain channel",
    "FIELDNAM": "High Gain Time of Flight",
    "LABLAXIS": "TOF High Ampl.",
    "VAR_NOTES": (
        "High gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

tof_mid_attrs = {
    "CATDESC": "Time of flight waveform on the mid-gain channel",
    "FIELDNAM": "Mid Gain Time of Flight",
    "LABLAXIS": "TOF Mid Ampl.",
    "VAR_NOTES": (
        "Mid gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

tof_low_attrs = {
    "CATDESC": "Time of flight waveform on the low-gain channel",
    "FIELDNAM": "Low Gain Time of Flight",
    "LABLAXIS": "TOF Low Ampl.",
    "VAR_NOTES": (
        "Low gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

target_low_attrs = {
    "CATDESC": "Target low charge sensitive amplifier waveform",
    "FIELDNAM": "Low Target Signal",
    "LABLAXIS": "Low Target Ampl.",
    "VAR_NOTES": (
        "Low gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge. "
    ),
} | l1_target_base

target_high_attrs = {
    "CATDESC": "Ion grid charge sensitive amplifier waveform",
    "FIELDNAM": "High Target Signal",
    "LABLAXIS": "High Target Ampl.",
    "VAR_NOTES": (
        "High gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
} | l1_target_base

ion_grid_attrs = {
    "CATDESC": "Ion grid charge sensitive amplifier waveform data",
    "FIELDNAM": "Ion Grid Signal",
    "LABLAXIS": "Ion Grid Ampl.",
    "VAR_NOTES": (
        "This is the ion grid signal from IDEX. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
} | l1_target_base
