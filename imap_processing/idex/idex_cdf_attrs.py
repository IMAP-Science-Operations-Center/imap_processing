import numpy as np

from imap_processing import common_cdf_attrs

# Set IDEX software version here for now
__version__ = 1

# Global Attributes
idex_l1_global_attrs = (
    {
        "Data_type": ["L1>Level-1"],
        "Data_version": [__version__],
        "Descriptor": ["IDEX>Interstellar Dust Experiment"],
        "TEXT": [
            """The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF)
             dust impact ionization mass spectrometer on the IMAP mission that
            provides the elemental composition, speed, and mass distributions
            of interstellar dust and interplanetary dust particles. Each record
            contains the data from a single dust impact. See
            https://imap.princeton.edu/instruments/idex for more details."""
        ],
        "Mission_group": ["IMAP"],
        "Logical_source": ["imap_idx_l1"],
        "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
        "Logical_source_description": ["IMAP Mission IDEX Instrument Level-1 Data.  "],
    }
    | common_cdf_attrs.global_base
)

idex_l2_global_attrs = (
    {
        "Data_type": ["l2"],
        "Data_version": [__version__],
        "TEXT": [
            """The Interstellar Dust Experiment (IDEX) is a time-of-flight (TOF)
             dust impact ionization mass spectrometer on the IMAP mission that
            provides the elemental composition, speed, and mass distributions
            of interstellar dust and interplanetary dust particles. Each record
            contains the data from a single dust impact.  See
            https://imap.princeton.edu/instruments/idex for more details."""
        ],
        "Mission_group": ["IMAP"],
        "Logical_source": ["imap_idx_l2"],
        "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
        "Logical_source_description": ["IMAP Mission IDEX Instrument Level-2 Data"],
    }
    | common_cdf_attrs.global_base
)

# L1 variables base dictionaries (these are not complete)
l1_data_base = {
    "DEPEND_0": "Epoch",
    "DISPLAY_TYPE": "spectrogram",
    "FILLVAL": np.array([-1.0e31], dtype=np.float32),
    "FORMAT": "E12.2",
    "UNITS": "dN",
    "VALIDMIN": np.array([-1.0e31], dtype=np.float32),
    "VALIDMAX": np.array([1.0e31], dtype=np.float32),
    "VAR_TYPE": "data",
    "SCALETYP": "linear",
}

l1_tof_base = {"DEPEND_1": "Time_High_SR", "LABLAXIS": "Time_[dN]"} | l1_data_base

l1_target_base = {
    "DEPEND_1": "Time_Low_SR",
    "LABLAXIS": "Amplitude_[dN]",
} | l1_data_base

sample_rate_base = {
    "DEPEND_0": "Epoch",
    "FILLVAL": np.array([-1.0e31], dtype=np.float32),
    "FORMAT": "E12.2",
    "LABLAXIS": "Time_[dN]",
    "UNITS": "microseconds",
    "VALIDMIN": np.array([-1.0e31], dtype=np.float32),
    "VALIDMAX": np.array([1.0e31], dtype=np.float32),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
}

trigger_base = {
    "DEPEND_0": "Epoch",
    "DISPLAY_TYPE": "no_plot",
    "FILLVAL": np.array([-1.0e31], dtype=np.float32),
    "FORMAT": "E12.2",
    "LABLAXIS": "Trigger_Info",
    "UNITS": "",
    "VALIDMIN": 0,
    "VALIDMAX": np.array([1.0e31], dtype=np.float32),
    "VAR_TYPE": "metadata",
}

# L1 Attribute Dictionaries
low_sr_attrs = (
    {
        "CATDESC": "Time_Low_SR",
        "FIELDNAM": "Time_Low_SR",
        "VAR_NOTES": """The Low sample rate in microseconds.
                    Steps are approximately 1/4.025 nanoseconds in duration.""",
    }
    | sample_rate_base
)

high_sr_attrs = (
    {
        "CATDESC": "Time_High_SR",
        "FIELDNAM": "Time_High_SR",
        "VAR_NOTES": """The High sample rate in microseconds.
                    Steps are approximately 1/260 nanoseconds in duration.""",
    }
    | sample_rate_base
)

tof_high_attrs = {
    "CATDESC": "TOF_High",
    "FIELDNAM": "TOF_High",
    "VAR_NOTES": "This is the high gain channel of the time-of-flight signal.",
} | l1_tof_base

tof_mid_attrs = {
    "CATDESC": "TOF_Mid",
    "FIELDNAM": "TOF_Mid",
    "VAR_NOTES": "This is the mid gain channel of the time-of-flight signal.",
} | l1_tof_base

tof_low_attrs = {
    "CATDESC": "TOF_Low",
    "FIELDNAM": "TOF_Low",
    "VAR_NOTES": "This is the low gain channel of the time-of-flight signal.",
} | l1_tof_base

target_low_attrs = {
    "CATDESC": "Target_Low",
    "FIELDNAM": "Target_Low",
    "VAR_NOTES": "This is the low gain channel of IDEX's target signal.",
} | l1_target_base

target_high_attrs = {
    "CATDESC": "Target_High",
    "FIELDNAM": "Target_High",
    "VAR_NOTES": "This is the high gain channel of IDEX's target signal.",
} | l1_target_base

ion_grid_attrs = {
    "CATDESC": "Ion_Grid",
    "FIELDNAM": "Ion_Grid",
    "VAR_NOTES": "This is the ion grid signal from IDEX.",
} | l1_target_base
