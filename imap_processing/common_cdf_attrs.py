import numpy as np

global_base = {
    "Project": ["STSP Cluster>Solar Terrestrial Science Programmes, Cluster"],
    "Source_name": ["IMAP>Interstellar Mapping and Acceleration Probe"],
    "Discipline": ["Solar Physics>Heliospheric Physics"],
    "Descriptor": ["IDEX>Interstellar Dust Experiment"],
    "PI_name": ["Dave McComas"],
    "PI_affiliation": ["Princeton"],
    "Instrument_type": ["Particles (space)"],
    "Mission_group": ["IMAP"],
}

epoch_attrs = {
    "CATDESC": "Default time",
    "FIELDNAM": "Epoch",
    "FILLVAL": np.array([-9223372036854775808]),
    "FORMAT": "a2",
    "LABLAXIS": "Epoch",
    "UNITS": "ns",
    "VALIDMIN": np.array([-315575942816000000]),
    "VALIDMAX": np.array([946728069183000000]),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
    "MONOTON": "INCREASE",
    "TIME_BASE": "J2000",
    "TIME_SCALE": "Terrestrial Time",
    "REFERENCE_POSITION": "Rotating Earth Geoid",
}
