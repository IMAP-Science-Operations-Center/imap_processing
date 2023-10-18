import numpy as np

INT_FILLVAL = np.iinfo(np.int64).min
DOUBLE_FILLVAL = np.float64(-1.0e-31)

global_base = {
    "Project": ["STP>Solar-Terrestrial Physics"],
    "Source_name": ["IMAP>Interstellar Mapping and Acceleration Probe"],
    "Discipline": ["Solar Physics>Heliospheric Physics"],
    "PI_name": ["Dr. David J. McComas"],
    "PI_affiliation": [
        "Princeton Plasma Physics Laboratory",
        "100 Stellarator Road, Princeton, NJ 08540",
    ],
    "Instrument_type": ["Particles (space)"],
    "Mission_group": ["IMAP>Interstellar Mapping and Acceleration Probe"],
}

epoch_attrs = {
    "CATDESC": "Default time",
    "FIELDNAM": "Epoch",
    "FILLVAL": INT_FILLVAL,
    "FORMAT": "a2",
    "LABLAXIS": "Epoch",
    "UNITS": "ns",
    "VALIDMIN": np.int64(-315575942816000000),
    "VALIDMAX": np.int64(946728069183000000),
    "VAR_TYPE": "support_data",
    "SCALETYP": "linear",
    "MONOTON": "INCREASE",
    "TIME_BASE": "J2000",
    "TIME_SCALE": "Terrestrial Time",
    "REFERENCE_POSITION": "Rotating Earth Geoid",
}
