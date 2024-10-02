"""Methods for GLOWS Level 1A processing and CDF writing."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import J2000_EPOCH
from imap_processing.glows.l0.decom_glows import decom_packets
from imap_processing.glows.l0.glows_l0_data import DirectEventL0
from imap_processing.glows.l1a.glows_l1a_data import DirectEventL1A, HistogramL1A
from imap_processing.spice.time import met_to_j2000ns


def create_glows_attr_obj(data_version: str) -> ImapCdfAttributes:
    """
    Load in 1la CDF attributes for GLOWS instrument.

    Parameters
    ----------
    data_version : str
        Data version for CDF filename, in the format "vXXX".

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
    glows_attrs.add_global_attribute("Data_version", data_version)
    return glows_attrs


def glows_l1a(packet_filepath: Path, data_version: str) -> list[xr.Dataset]:
    """
    Will process packets into GLOWS L1A CDF files.

    Outputs Datasets for histogram and direct event GLOWS L1A. This list can be passed
    into write_cdf to output CDF files.

    Parameters
    ----------
    packet_filepath : pathlib.Path
        Path to packet file for processing.
    data_version : str
        Data version for CDF filename, in the format "vXXX".

    Returns
    -------
    generated_files : list[xr.Dataset]
        List of the L1A datasets.
    """
    # Create ImapCdfAttributes object for cdf attributes management
    glows_attrs = create_glows_attr_obj(data_version)

    # TODO: Data version inside file as well?

    # Decompose packet file into histogram, and direct event data.
    hist_l0, de_l0 = decom_packets(packet_filepath)

    # Create dictionaries to group data by day
    de_by_day = process_de_l0(de_l0)
    hists_by_day = defaultdict(list)

    # TODO: Make this its own function?
    for hist in hist_l0:
        hist_l1a = HistogramL1A(hist)
        # Split by IMAP start time
        # TODO: Should this be MET?
        hist_day = (J2000_EPOCH + met_to_j2000ns(hist.SEC)).astype("datetime64[D]")
        hists_by_day[hist_day].append(hist_l1a)

    # Generate CDF files for each day
    output_datasets = []
    for hist_l1a_list in hists_by_day.values():
        dataset = generate_histogram_dataset(hist_l1a_list, glows_attrs)
        output_datasets.append(dataset)

    for de_l1a_list in de_by_day.values():
        dataset = generate_de_dataset(de_l1a_list, glows_attrs)
        output_datasets.append(dataset)

    return output_datasets


def process_de_l0(
    de_l0: list[DirectEventL0],
) -> dict[np.datetime64, list[DirectEventL1A]]:
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
    de_by_day : dict[np.datetime64, list[DirectEventL1A]]
        Dictionary with keys of days and values of lists of DirectEventL1A objects.
        Each day has one CDF file associated with it.
    """
    de_by_day = dict()

    for de in de_l0:
        de_day = (J2000_EPOCH + met_to_j2000ns(de.MET)).astype("datetime64[D]")
        if de_day not in de_by_day:
            de_by_day[de_day] = [DirectEventL1A(de)]
        # Putting not first data int o last direct event list.
        elif de.SEQ != 0:
            # If the direct event is part of a sequence and is not the first,
            # append it to the last direct event in the list
            de_by_day[de_day][-1].append(de)
        else:
            de_by_day[de_day].append(DirectEventL1A(de))

    return de_by_day


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
    time_data = np.zeros(len(de_l1a_list), dtype="datetime64[ns]")
    # TODO: Should each timestamp point to a list of direct events, each with a
    #  timestamp? Or should the list be split out to make the timestamps?

    # Each DirectEventL1A class covers 1 second of direct events data
    direct_events = np.zeros((len(de_l1a_list), len(de_l1a_list[0].direct_events), 4))

    support_data: dict = {
        # "flight_software_version": [], # breaks
        "seq_count_in_pkts_file": [],  # works
        "number_of_de_packets": [],  # works
        # "missing_packet_sequences": [] # breaks
    }

    data_every_second: dict = {
        "imap_sclk_last_pps": [],
        "glows_sclk_last_pps": [],
        "glows_ssclk_last_pps": [],
        "imap_sclk_next_pps": [],
        "catbed_heater_active": [],
        "spin_period_valid": [],
        "spin_phase_at_next_pps_valid": [],
        "spin_period_source": [],
        "spin_period": [],
        "spin_phase_at_next_pps": [],
        "number_of_completed_spins": [],
        "filter_temperature": [],
        "hv_voltage": [],
        "glows_time_on_pps_valid": [],
        "time_status_valid": [],
        "housekeeping_valid": [],
        "is_pps_autogenerated": [],
        "hv_test_in_progress": [],
        "pulse_test_in_progress": [],
        "memory_error_detected": [],
    }

    for index, de in enumerate(de_l1a_list):
        # Set the timestamp to the first timestamp of the direct event list
        epoch_time = met_to_j2000ns(de.l0.MET).astype("datetime64[ns]")

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
        # support_data["flight_software_version"].append(
        # str(de.l0.ccsds_header.VERSION))
        support_data["seq_count_in_pkts_file"].append(
            int(de.l0.ccsds_header.SRC_SEQ_CTR)
        )
        support_data["number_of_de_packets"].append(int(de.l0.LEN))
        # support_data["missing_packet_sequences"].append(str(de.missing_seq))

        for key in data_every_second.keys():
            data_every_second[key].append(de.status_data.__getattribute__(key))

    # Convert arrays and dictionaries into xarray 'DataArray' objects
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=glows_cdf_attributes.get_variable_attributes("epoch"),
    )

    direct_event = xr.DataArray(
        # Corresponds to DirectEvent (seconds, subseconds, impulse_length, multi_event)
        np.arange(4),
        name="direct_event_components",
        dims=["direct_event_components"],
        attrs=glows_cdf_attributes.get_variable_attributes(
            "direct_event_components_attrs"
        ),
    )

    # TODO come up with a better name
    within_the_second = xr.DataArray(
        np.arange(direct_events.shape[1]),
        name="within_the_second",
        dims=["within_the_second"],
        attrs=glows_cdf_attributes.get_variable_attributes("within_the_second"),
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

    # TODO: This is the weird global attribute.
    # Create an xarray dataset object, and add DataArray objects into it
    output = xr.Dataset(
        coords={"epoch": time_data},
        attrs=glows_cdf_attributes.get_global_attributes("imap_glows_l1a_de"),
    )

    output["direct_events"] = de

    # TODO: Do we want missing_sequences as support data or as global attrs?
    # Currently: support data, with a string

    for key, value in support_data.items():
        output[key] = xr.DataArray(
            value,
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    for key, value in data_every_second.items():
        output[key] = xr.DataArray(
            value,
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    return output


def generate_histogram_dataset(
    hist_l1a_list: list[HistogramL1A], glows_cdf_attributes: ImapCdfAttributes
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
    time_data = np.zeros(len(hist_l1a_list), dtype="datetime64[ns]")
    # TODO Add daily average of histogram counts
    # TODO compute average temperature etc
    # Data in lists, for each of the 25 time varying datapoints in HistogramL1A

    hist_data = np.zeros((len(hist_l1a_list), 3600), dtype=np.int64)

    # TODO: add missing attributes
    support_data: dict = {
        "flight_software_version": [],
        # "pkts_file_name": [],
        "seq_count_in_pkts_file": [],
        "first_spin_id": [],
        "last_spin_id": [],
        "flags_set_onboard": [],
        "is_generated_on_ground": [],
        "number_of_spins_per_block": [],
        "number_of_bins_per_histogram": [],
        "number_of_events": [],
        "filter_temperature_average": [],
        "filter_temperature_variance": [],
        "hv_voltage_average": [],
        "hv_voltage_variance": [],
        "spin_period_average": [],
        "spin_period_variance": [],
        "pulse_length_average": [],
        "pulse_length_variance": [],
    }
    time_metadata: dict = {
        "imap_start_time": [],
        "imap_time_offset": [],
        "glows_start_time": [],
        "glows_time_offset": [],
    }

    for index, hist in enumerate(hist_l1a_list):
        # TODO: Should this be MET?
        epoch_time = met_to_j2000ns(hist.imap_start_time.to_seconds())
        hist_data[index] = hist.histogram

        support_data["flags_set_onboard"].append(hist.flags["flags_set_onboard"])
        support_data["is_generated_on_ground"].append(
            int(hist.flags["is_generated_on_ground"])
        )

        # Add support_data keys to the support_data dictionary
        for key in support_data.keys():
            if key not in ["flags_set_onboard", "is_generated_on_ground"]:
                support_data[key].append(hist.__getattribute__(key))
        # For the time varying data, convert to seconds and then append
        for key in time_metadata.keys():
            time_metadata[key].append(hist.__getattribute__(key).to_seconds())
        time_data[index] = epoch_time

    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=glows_cdf_attributes.get_variable_attributes("epoch"),
    )
    bin_count = 3600  # TODO: Is it always 3600 bins?

    bins = xr.DataArray(
        np.arange(bin_count),
        name="bins",
        dims=["bins"],
        attrs=glows_cdf_attributes.get_variable_attributes("bins_attrs"),
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
        coords={"epoch": epoch_time, "bins": bins},
        attrs=attrs,
    )

    output["histogram"] = hist

    for key, value in support_data.items():
        output[key] = xr.DataArray(
            value,
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    for key, value in time_metadata.items():
        output[key] = xr.DataArray(
            value,
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=glows_cdf_attributes.get_variable_attributes(key),
        )

    return output
