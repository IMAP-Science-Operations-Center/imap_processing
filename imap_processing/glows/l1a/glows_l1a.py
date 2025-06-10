"""Methods for GLOWS Level 1A processing and CDF writing."""

from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows.l0.decom_glows import decom_packets
from imap_processing.glows.l0.glows_l0_data import DirectEventL0
from imap_processing.glows.l1a.glows_l1a_data import DirectEventL1A, HistogramL1A
from imap_processing.spice.time import (
    met_to_ttj2000ns,
)


def create_glows_attr_obj() -> ImapCdfAttributes:
    """
    Load in 1la CDF attributes for GLOWS instrument.

    Returns
    -------
    glows_attrs : ImapCdfAttributes
        Imap object with l1a attribute files loaded in.
    """
    # Create ImapCdfAttributes object for cdf attributes management
    glows_attrs = ImapCdfAttributes()
    # Load in files
    glows_attrs.add_instrument_global_attrs("glows")
    glows_attrs.add_instrument_variable_attrs("glows", "l1a")
    return glows_attrs


def glows_l1a(packet_filepath: Path) -> list[xr.Dataset]:
    """
    Will process packets into GLOWS L1A CDF files.

    Outputs Datasets for histogram and direct event GLOWS L1A. This list can be passed
    into write_cdf to output CDF files.

    We expect one input L0 file to be processed into one L1A file, with one
    observational day's worth of data.

    Parameters
    ----------
    packet_filepath : pathlib.Path
        Path to packet file for processing.

    Returns
    -------
    generated_files : list[xr.Dataset]
        List of the L1A datasets.
    """
    # Create ImapCdfAttributes object for cdf attributes management
    glows_attrs = create_glows_attr_obj()

    # Decompose packet file into histogram, and direct event data.
    hist_l0, de_l0 = decom_packets(packet_filepath)

    l1a_de = process_de_l0(de_l0)
    l1a_hists = []
    for hist in hist_l0:
        l1a_hists.append(HistogramL1A(hist))

    # Generate CDF files for each day
    output_datasets = []
    dataset = generate_histogram_dataset(l1a_hists, glows_attrs)
    output_datasets.append(dataset)

    dataset = generate_de_dataset(l1a_de, glows_attrs)
    output_datasets.append(dataset)

    return output_datasets


def process_de_l0(
    de_l0: list[DirectEventL0],
) -> list[DirectEventL1A]:
    """
    Will process Direct Event packets into GLOWS L1A CDF files.

    This involves combining packets with direct event sequences that span multiple
    packets.

    Parameters
    ----------
    de_l0 : list[DirectEventL0]
        List of DirectEventL0 objects.

    Returns
    -------
    de_by_day : list[DirectEventL1A]
        Dictionary with keys of days and values of lists of DirectEventL1A objects.
        Each day has one CDF file associated with it.
    """
    de_list: list[DirectEventL1A] = []

    for de in de_l0:
        # Putting not first data int o last direct event list.
        if de.SEQ != 0:
            # If the direct event is part of a sequence and is not the first,
            # add it to the last direct event in the list
            de_list[-1].merge_de_packets(de)
        else:
            de_list.append(DirectEventL1A(de))

    return de_list


def generate_de_dataset(
    de_l1a_list: list[DirectEventL1A],
    glows_cdf_attributes: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Generate a dataset for GLOWS L1A direct event data CDF files.

    Parameters
    ----------
    de_l1a_list : list[DirectEventL1A]
        List of DirectEventL1A objects for a given day.
    glows_cdf_attributes : ImapCdfAttributes
        Object containing l1a CDF attributes for instrument glows.

    Returns
    -------
    output : xarray.Dataset
        Dataset containing the GLOWS L1A direct event CDF output.
    """
    # TODO: Block header per second, or global attribute?

    # Store timestamps for each DirectEventL1a object.
    time_data = np.zeros(len(de_l1a_list), dtype=np.int64)

    # Each DirectEventL1A class covers 1 second of direct events data
    direct_events = np.zeros((len(de_l1a_list), len(de_l1a_list[0].direct_events), 4))
    missing_packets_sequence = ""

    # First variable is the output data type, second is the list of values
    support_data: dict = {
        # "flight_software_version": [],
        "seq_count_in_pkts_file": [np.uint16, []],
        "number_of_de_packets": [np.uint32, []],
    }

    data_every_second: dict = {
        "imap_sclk_last_pps": [np.uint32, []],
        "glows_sclk_last_pps": [np.float64, []],
        "glows_ssclk_last_pps": [np.float64, []],
        "imap_sclk_next_pps": [np.uint32, []],
        "catbed_heater_active": [np.uint8, []],
        "spin_period_valid": [np.uint8, []],
        "spin_phase_at_next_pps_valid": [np.uint8, []],
        "spin_period_source": [np.uint8, []],
        "spin_period": [np.float64, []],
        "spin_phase_at_next_pps": [np.float64, []],
        "number_of_completed_spins": [np.uint32, []],
        "filter_temperature": [np.float64, []],
        "hv_voltage": [np.float64, []],
        "glows_time_on_pps_valid": [np.uint8, []],
        "time_status_valid": [np.uint8, []],
        "housekeeping_valid": [np.uint8, []],
        "is_pps_autogenerated": [np.uint8, []],
        "hv_test_in_progress": [np.uint8, []],
        "pulse_test_in_progress": [np.uint8, []],
        "memory_error_detected": [np.uint8, []],
    }

    for index, de in enumerate(de_l1a_list):
        # Set the timestamp to the first timestamp of the direct event list
        epoch_time = met_to_ttj2000ns(de.l0.MET)

        # determine if the length of the direct_events numpy array is long enough,
        # and extend the direct_events length dimension if necessary.
        de_len = len(de.direct_events)
        if de_len > direct_events.shape[1]:
            # If the new DE list is longer than the existing shape, first reshape
            # direct_events and pad the existing vectors with zeros.
            direct_events = np.pad(
                direct_events,
                (
                    (
                        0,
                        0,
                    ),
                    (0, de_len - direct_events.shape[1]),
                    (0, 0),
                ),
                "constant",
                constant_values=(0,),
            )

        new_de = np.array([event.to_list() for event in de.direct_events])

        direct_events[index, : len(de.direct_events), :] = new_de
        time_data[index] = epoch_time

        # Adding data that will go into CDF file
        support_data["seq_count_in_pkts_file"][1].append(
            int(de.l0.ccsds_header.SRC_SEQ_CTR)
        )
        support_data["number_of_de_packets"][1].append(int(de.l0.LEN))
        missing_packets_sequence += str(de.missing_seq) + ","

        for key, val in data_every_second.items():
            val[1].append(de.status_data.__getattribute__(key))

    # Convert arrays and dictionaries into xarray 'DataArray' objects
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=glows_cdf_attributes.get_variable_attributes("epoch", check_schema=False),
    )

    direct_event = xr.DataArray(
        # Corresponds to DirectEvent (seconds, subseconds, impulse_length, multi_event)
        np.arange(4),
        name="direct_event_components",
        dims=["direct_event_components"],
        attrs=glows_cdf_attributes.get_variable_attributes(
            "direct_event_components_attrs", check_schema=False
        ),
    )

    within_the_second = xr.DataArray(
        np.arange(direct_events.shape[1]),
        name="within_the_second",
        dims=["within_the_second"],
        attrs=glows_cdf_attributes.get_variable_attributes(
            "within_the_second", check_schema=False
        ),
    )

    de = xr.DataArray(
        direct_events,
        name="direct_events",
        dims=["epoch", "within_the_second", "direct_event_components"],
        coords={
            "epoch": epoch_time,
            "within_the_second": within_the_second,
            "direct_event_components": direct_event,
        },
        attrs=glows_cdf_attributes.get_variable_attributes("direct_events"),
    )

    # Create an xarray dataset object, and add DataArray objects into it
    output = xr.Dataset(
        coords={"epoch": time_data},
        attrs=glows_cdf_attributes.get_global_attributes("imap_glows_l1a_de"),
    )

    output["direct_events"] = de

    for key, value in support_data.items():
        output[key] = xr.DataArray(
            np.array(value[1], dtype=value[0]),
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    for key, value in data_every_second.items():
        output[key] = xr.DataArray(
            np.array(value[1], dtype=value[0]),
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )
    output.attrs["missing_packets_sequence"] = missing_packets_sequence[:-1]
    return output


def generate_histogram_dataset(
    hist_l1a_list: list[HistogramL1A],
    glows_cdf_attributes: ImapCdfAttributes,
) -> xr.Dataset:
    """
    Generate a dataset for GLOWS L1A histogram data CDF files.

    Parameters
    ----------
    hist_l1a_list : list[HistogramL1A]
        List of HistogramL1A objects for a given day.
    glows_cdf_attributes : ImapCdfAttributes
        Object containing l1a CDF attributes for instrument glows.

    Returns
    -------
    output : xarray.Dataset
        Dataset containing the GLOWS L1A histogram CDF output.
    """
    # Store timestamps for each HistogramL1A object.
    time_data = np.zeros(len(hist_l1a_list), dtype=np.int64)
    # TODO Add daily average of histogram counts
    # Data in lists, for each of the 25 time varying datapoints in HistogramL1A

    hist_data = np.zeros((len(hist_l1a_list), 3600), dtype=np.uint16)

    # First variable is the output data type, second is the list of values
    support_data: dict = {
        "flight_software_version": [np.uint32, []],
        "seq_count_in_pkts_file": [np.uint16, []],
        "first_spin_id": [np.uint32, []],
        "last_spin_id": [np.uint32, []],
        "flags_set_onboard": [np.uint16, []],
        "is_generated_on_ground": [np.uint8, []],
        "number_of_spins_per_block": [np.uint8, []],
        "number_of_bins_per_histogram": [np.uint16, []],
        "number_of_events": [np.uint32, []],
        "filter_temperature_average": [np.uint32, []],
        "filter_temperature_variance": [np.uint32, []],
        "hv_voltage_average": [np.uint32, []],
        "hv_voltage_variance": [np.uint32, []],
        "spin_period_average": [np.uint32, []],
        "spin_period_variance": [np.uint32, []],
        "pulse_length_average": [np.uint32, []],
        "pulse_length_variance": [np.uint32, []],
    }
    time_metadata: dict = {
        "imap_start_time": [np.float64, []],
        "imap_time_offset": [np.float64, []],
        "glows_start_time": [np.float64, []],
        "glows_time_offset": [np.float64, []],
    }

    for index, hist in enumerate(hist_l1a_list):
        epoch_time = met_to_ttj2000ns(hist.imap_start_time.to_seconds())
        hist_data[index] = hist.histogram

        support_data["flags_set_onboard"][1].append(hist.flags["flags_set_onboard"])
        support_data["is_generated_on_ground"][1].append(
            int(hist.flags["is_generated_on_ground"])
        )

        # Add support_data keys to the support_data dictionary
        for key, support_val in support_data.items():
            if key not in ["flags_set_onboard", "is_generated_on_ground"]:
                support_val[1].append(hist.__getattribute__(key))
        # For the time varying data, convert to seconds and then append
        for key, time_metadata_val in time_metadata.items():
            time_metadata_val[1].append(hist.__getattribute__(key).to_seconds())
        time_data[index] = epoch_time

    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=glows_cdf_attributes.get_variable_attributes("epoch", check_schema=False),
    )
    bin_count = 3600  # TODO: Is it always 3600 bins?

    bins = xr.DataArray(
        np.arange(bin_count),
        name="bins",
        dims=["bins"],
        attrs=glows_cdf_attributes.get_variable_attributes(
            "bins_attrs", check_schema=False
        ),
    )

    bin_label = xr.DataArray(
        bins.data.astype(str),
        name="bins_label",
        dims=["bins_label"],
        attrs=glows_cdf_attributes.get_variable_attributes(
            "bins_label", check_schema=False
        ),
    )

    hist = xr.DataArray(
        hist_data,
        name="histogram",
        dims=["epoch", "bins"],
        coords={"epoch": epoch_time, "bins": bins},
        attrs=glows_cdf_attributes.get_variable_attributes(
            "histogram"
        ),  # Used to be histogram_attrs
    )

    attrs = glows_cdf_attributes.get_global_attributes("imap_glows_l1a_hist")

    output = xr.Dataset(
        coords={"epoch": epoch_time, "bins": bins, "bins_label": bin_label},
        attrs=attrs,
    )

    output["histogram"] = hist

    for key, value in support_data.items():
        output[key] = xr.DataArray(
            np.array(value[1], dtype=value[0]),
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )
    for key, value in time_metadata.items():
        output[key] = xr.DataArray(
            np.array(value[1], dtype=value[0]),
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    return output
