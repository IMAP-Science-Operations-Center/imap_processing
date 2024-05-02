"""Shared attribute values for GLOWS CDF files."""

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.glows import __version__

text = (
    "The GLObal Solar Wind Structure (GLOWS) is a non-imaging single-pixel Lyman-alpha "
    "photometer that will be used to observe the sky distribution of the helioglow to "
    "better understand the evolution of the solar wind structure. "
    "The Lyman-alpha photon counts from these observations can be used to build a more "
    "comprehensive picture of the solar wind structure and how it changes through the "
    "solar cycles. "
    "GLOWS design and assembly is led by the Space Research Center, Warsaw, Poland "
    "(CBK PAN). See https://imap.princeton.edu/instruments/glows for more details."
)

glows_base = GlobalInstrumentAttrs(
    __version__,
    "GLOWS>GLObal Solar Wind Structure",
    text,
    "Photons (space)",  # TODO: is this accurate?
)

# TODO: Do we want a data format version as well as a software version?

glows_l1a_hist_attrs = GlobalDataLevelAttrs(
    "L1A_hist>Level-1A histogram",
    logical_source="imap_glows_l1a_hist",
    logical_source_desc="IMAP Mission GLOWS Histogram Level-1A Data.",
    instrument_base=glows_base,
)

glows_l1a_de_attrs = GlobalDataLevelAttrs(
    "L1A_de>Level-1A direct event",
    logical_source="imap_glows_l1a_de",
    logical_source_desc="IMAP Mission GLOWS Direct Event Level-1A Data.",
    instrument_base=glows_base,
)

bins_attrs = AttrBase(
    validmin=0,
    validmax=70,  # doc says maximum count per bin is 66.7
    catdesc="Counts of direct events for photon impacts.",
    fieldname="Counts of direct events.",
    format="F7.4",  # Float with 4 digits
    var_type="support_data",
    display_type="time_series",
    label_axis="Counts",
)

# TODO Update this
per_second_attrs = AttrBase(
    validmin=0,
    validmax=300,
    catdesc="Direct events recorded approximately per second",
    fieldname="List of direct events ",
    format="F7.4",  # Float with 4 digits
    var_type="support_data",
    display_type="time_series",
    label_axis="Counts",
)

# TODO Update this
# TODO rename this
direct_event_attrs = AttrBase(
    validmin=0,
    validmax=300,  # TODO what is a reasonable max
    catdesc="Direct events recorded approximately per second",
    fieldname="List of direct events ",
    format="F7.4",  # Int with 10 digits
    var_type="data",
    display_type="time_series",
    label_axis="Counts",
)

histogram_attrs = ScienceAttrs(
    validmin=0,
    validmax=70,  # doc says maximum count per bin is 66.7, value has a 71
    catdesc="Histogram of photon counts.",
    depend_0="epoch",
    depend_1="bins",
    fieldname="Histogram of photon counts.",
    format="F7.4",  # Float with 4 digits TODO: are histogram objects float or int?
    display_type="time_series",
    label_axis="Counts",
    fill_val=GlobalConstants.INT_FILLVAL,
    units="counts",
    var_type="data",
)

metadata_attrs = ScienceAttrs(
    depend_0="epoch",
    validmin=GlobalConstants.INT_FILLVAL,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    fill_val=GlobalConstants.INT_FILLVAL,
    var_type="metadata",
    format="I6",
    label_axis="Metadata",
)

# TODO: Validate/update these
catdesc_fieldname_l1a = {
    "flight_software_version": ["Flight software version", "Flight software version"],
    "ground_software_version": [
        "Ground software version in the format vXXX",
        "Ground software version",
    ],
    "pkts_file_name": ["Input filename for packets data", "Input filename"],
    "seq_count_in_pkts_file": ["Sequence counter from packet", "Sequence counter"],
    "last_spin_id": ["The ID of the last spin in the block", "Last spin ID"],
    "imap_start_time": [
        "Start time of the block, in seconds, in IMAP spacecraft clock time",
        "Start time in seconds in IMAP time",
    ],
    "imap_time_offset": [
        "End time of the block, in seconds, in IMAP spacecraft clock time",
        "End time in seconds in IMAP time",
    ],
    "glows_start_time": [
        "Start time of the block, in seconds, in GLOWS clock time",
        "Start time in seconds in GLOWS time",
    ],
    "glows_time_offset": [
        "End time of the block, in seconds, in GLOWS clock time",
        "End time in seconds in GLOWS time",
    ],
    "flags_set_onboard": [
        "Flags for bad data [missing PPS, missing time, missing spin phase, missing "
        "spin periods, overexposure, nonmonotonic event, data collected at night, "
        "HV test, pulse test, memory error]",
        "Flags for missing or bad data",
    ],
    "is_generated_on_ground": [
        "Indicates if the histogram data was generated on the ground",
        "Histogram data generated on ground",
    ],
    "imap_sclk_last_pps": ["IMAP seconds for last PPS", "IMAP seconds for last PPS"],
    "glows_sclk_last_pps": ["GLOWS seconds for last PPS", "GLOWS seconds for last PPS"],
    "glows_ssclk_last_pps": [
        "GLOWS subseconds for last PPS, with a max of 2000000",
        "GLOWS subseconds for last PPS",
    ],
    "imap_sclk_next_pps": ["IMAP seconds for next PPS", "IMAP seconds for next PPS"],
    "catbed_heater_active": ["Indicates if the heater is active", "Heater active flag"],
    "spin_period_valid": [
        "Indicates if the spin phase is valid",
        "Valid spin phase flag",
    ],
    "spin_phase_at_next_pps_valid": [
        "Indicates if the spin phase at next PPS is valid",
        "Spin phase at next PPS is valid",
    ],
    "spin_period_source": ["Flag - Spin period source", "Flag - Spin period source"],
    "spin_period": ["Uint encoded spin period value", "Uint encoded spin period value"],
    "spin_phase_at_next_pps": [
        "Uint encoded next spin phase value",
        "Uint encoded next spin phase value",
    ],
    "number_of_completed_spins": [
        "Number of spins, from onboard",
        "Number of spins, from onboard",
    ],
    "filter_temperature": ["Uint encoded temperature", "Uint encoded temperature"],
    "hv_voltage": ["Uint encoded voltage", "Uint encoded voltage"],
    "glows_time_on_pps_valid": [
        "Indicates if the glows time is valid",
        "Flag - is glows time valid",
    ],
    "time_status_valid": ["Flag - valid time status", "Flag - valid time status"],
    "housekeeping_valid": ["Flag - valid housekeeping", "Flag - valid housekeeping"],
    "is_pps_autogenerated": [
        "Indicates if PPS is autogenerated",
        "Indicates if PPS is autogenerated",
    ],
    "hv_test_in_progress": ["Flag", "Flag"],
    "pulse_test_in_progress": ["Flag", "Flag"],
    "memory_error_detected": ["Flag", "Flag"],
    "number_of_de_packets": ["Number of DE packets", "Number of DE packets"],
    "missing_packet_sequences": [
        "Missing packet sequence numbers",
        "Missing packet sequences",
    ],
}
