from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    CoordinateAttrs,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.mag import __version__

text = (
    "The IMAP magnetometer (MAG) consists of a pair of identical magnetometers "
    "which each measure the magnetic field in three directions in the vicinity of "
    "the spacecraft. "
    "MAG will contribute to our understanding of the acceleration and transport "
    "of charged particles in the heliosphere. "
    "MAG design and assembly is led by Imperial College, London. See "
    "https://imap.princeton.edu/instruments/mag for more details."
)

mag_base = GlobalInstrumentAttrs(
    __version__,
    "MAG>Magnetometer",
    text,
    "Magnetic Fields (space)",
    pi_name=["Tim Horbury"],
    pi_affiliation=["The Blackett Laboratory", "Imperial College London"],
)

mag_l1a_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1A",
    logical_source="imap_mag_l1a",
    logical_source_desc="IMAP Mission MAG Instrument Level-1A Data.",
    instrument_base=mag_base,
)

mag_l1b_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1B",
    logical_source="imap_mag_l1b",
    logical_source_desc="IMAP Mission MAG Instrument Level-1B Data.",
    instrument_base=mag_base,
)

mag_l1c_attrs = GlobalDataLevelAttrs(
    "L1A>Level-1C",
    logical_source="imap_mag_l1c",
    logical_source_desc="IMAP Mission MAG Instrument Level-1C Data.",
    instrument_base=mag_base,
)

# TODO: Supporting data attributes?

# TODO: display type, catdesc, units, format, label_axis
mag_vector_attrs = ScienceAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    depend_0="Epoch",
    depend_1="Direction",
    display_type="time_series",
    fieldname="Magnetic Field Vector",
    fill_val=GlobalConstants.INT_FILLVAL,
    format="F64.5",
    units="counts",
    var_type="data",
)

mag_flag_attrs = ScienceAttrs(
    validmin=0,
    validmax=1,
    depend_0="Epoch",
    display_type="time_series",
    fill_val=255,
    format="I1",
)

direction_attrs = CoordinateAttrs(
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    catdesc="Magnetic field vector",
    fieldname="[x,y,z] magnetic field vector",
    format="F64.5",
    var_type="support_data",
    lablaxis="Magnetic field vector",
)


# "FEE_ICU_IO_STATUS": {
#             "dims": [
#                 "dim_empty"
#             ],
#             "attrs": {
#                 "CATDESC": "Sensor Front End Electronics Instrument Control Unit Input Output Status",
#                 "DEPEND_0": "EPOCH",
#                 "DETECTOR": "OBS>Outboard Sensor",
#                 "DISPLAY_TYPE": "time_series",
#                 "FIELDNAM": "Sensor Front End Electronics ICU IO Status",
#                 "FILLVAL": 255,
#                 "FORMAT": "I1",
#                 "LABLAXIS": "FEE ICU IO Status",
#                 "SCALEMAX": 1,
#                 "SCALEMIN": 0,
#                 "SCALETYP": "linear",
#                 "UNITS": "None",
#                 "VALIDMAX": 1,
#                 "VALIDMIN": 0,
#                 "VAR_TYPE": "metadata",
#                 "standard_name": "Sensor Front End Electronics ICU IO Status",
#                 "long_name": "FEE ICU IO Status",
#                 "units": "None"
#             },
#             "data": []
#         },
