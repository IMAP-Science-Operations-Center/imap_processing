import numpy as np

from imap_processing import common_cdf_attrs

# Set IDEX software version here for now
__version__ = "1"

# Global Attributes
idex_l1_global_attrs = {
    "Data_type": ["L1>Level-1"],
    "Data_version": [__version__],
    "Descriptor": ["IDEX>Interstellar Dust Experiment"],
    "TEXT": [
        (
            "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
            "dust impact ionization mass spectrometer on the IMAP mission that "
            "provides the elemental composition, speed, and mass distributions "
            "of interstellar dust and interplanetary dust particles. Each record "
            "contains the data from a single dust impact. See "
            "https://imap.princeton.edu/instruments/idex for more details."
        )
    ],
    "Mission_group": ["IMAP"],
    "Logical_source": ["imap_idx_l1"],
    "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
    "Logical_source_description": ["IMAP Mission IDEX Instrument Level-1 Data."],
} | common_cdf_attrs.global_base

idex_l2_global_attrs = {
    "Data_type": ["l2"],
    "Data_version": [__version__],
    "TEXT": [
        (
            "The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF) "
            "dust impact ionization mass spectrometer on the IMAP mission that "
            "provides the elemental composition, speed, and mass distributions "
            "of interstellar dust and interplanetary dust particles. Each record "
            "contains the data from a single dust impact. See "
            "https://imap.princeton.edu/instruments/idex for more details. "
        )
    ],
    "Mission_group": ["IMAP"],
    "Logical_source": ["imap_idx_l2"],
    "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
    "Logical_source_description": ["IMAP Mission IDEX Instrument Level-2 Data"],
} | common_cdf_attrs.global_base

# L1 variables base dictionaries (these are not complete)
l1_data_base = {
    "DEPEND_0": "Epoch",
    "DISPLAY_TYPE": "spectrogram",
    "FILLVAL": np.float32(-1.0e31),
    "FORMAT": "E12.2",
    "UNITS": "dN",
    "VALIDMIN": np.float32(0),
    "VALIDMAX": np.float32(4096),
    "VAR_TYPE": "data",
    "SCALETYP": "linear",
}

l1_tof_base = {"DEPEND_1": "Time_High_SR", "LABLAXIS": "Amplitude"} | l1_data_base

l1_target_base = {
    "DEPEND_1": "Time_Low_SR",
    "LABLAXIS": "Amplitude",
} | l1_data_base

sample_rate_base = {
    "DEPEND_0": "Epoch",
    "FILLVAL": np.float32(-1.0e31),
    "FORMAT": "E12.2",
    "LABLAXIS": "Time",
    "UNITS": "microseconds",
    "VALIDMIN": np.float32(-500),
    "VALIDMAX": np.float32(500),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

trigger_base = {
    "DEPEND_0": "Epoch",
    "FILLVAL": np.iinfo(np.int64).min,
    "FORMAT": "E12.2",
    "VALIDMIN": 0,
    "VAR_TYPE": "support_data",
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
    "VAR_NOTES": (
        "High gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

tof_mid_attrs = {
    "CATDESC": "Time of flight waveform on the mid-gain channel",
    "FIELDNAM": "Mid Gain Time of Flight",
    "VAR_NOTES": (
        "Mid gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

tof_low_attrs = {
    "CATDESC": "Time of flight waveform on the low-gain channel",
    "FIELDNAM": "Low Gain Time of Flight",
    "VAR_NOTES": (
        "Low gain channel of the time-of-flight signal. "
        "Sampled at 260 Megasamples per second, with a 10-bit resolution. "
        "Data is used to quantify dust composition."
    ),
} | l1_tof_base

target_low_attrs = {
    "CATDESC": "Target low charge sensitive amplifier waveform",
    "FIELDNAM": "Low Target Signal",
    "VAR_NOTES": (
        "This is the low gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge. "
    ),
} | l1_target_base

target_high_attrs = {
    "CATDESC": "Ion grid charge sensitive amplifier waveform",
    "FIELDNAM": "High Target Signal",
    "VAR_NOTES": (
        "This is the high gain channel of IDEX's target signal. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
} | l1_target_base

ion_grid_attrs = {
    "CATDESC": "Ion grid charge sensitive amplifier waveform data",
    "FIELDNAM": "Ion Grid Signal",
    "VAR_NOTES": (
        "This is the ion grid signal from IDEX. "
        "Sampled at 3.75 Msps with 12-bit resolution. "
        "Data is used to quantify dust charge."
    ),
} | l1_target_base
