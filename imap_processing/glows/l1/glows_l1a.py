"""Methods for GLOWS Level 1A processing and CDF writing."""

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.glows import glows_cdf_attrs
from imap_processing.glows.l0.decom_glows import decom_packets
from imap_processing.glows.l0.glows_l0_data import DirectEventL0
from imap_processing.glows.l1.glows_l1a_data import DirectEventL1A, HistogramL1A
from imap_processing.glows.utils.constants import GlowsConstants


def glows_l1a(packet_filepath: Path, data_version: str) -> list[Path]:
    """
    Process packets into GLOWS L1A CDF files.

    Outputs CDF files for histogram and direct event GLOWS L1A CDF files.

    Parameters
    ----------
    packet_filepath: Path
        Path to packet file for processing
    data_version: str
        Data version for CDF filename, in the format "vXXX"

    Returns
    -------
    generated_files: list[Path]
        List of the paths of the generated CDF files
    """
    # TODO: Data version inside file as well?
    # Create glows L0
    hist_l0, de_l0 = decom_packets(packet_filepath)

    de_by_day = process_de_l0(de_l0)
    hists_by_day = dict()

    # Create histogram L1A and filter into days based on start time
    for hist in hist_l0:
        hist_l1a = HistogramL1A(hist)
        # Split by IMAP start time
        # TODO: Should this be MET?
        hist_day = calc_start_time(hist.SEC).astype("datetime64[D]")
        if hist_day not in hists_by_day:
            hists_by_day[hist_day] = []
        hists_by_day[hist_day].append(hist_l1a)

    # Generate CDF files for each day
    generated_files = []
    for _, hist_l1a_list in hists_by_day.items():
        dataset = generate_histogram_dataset(hist_l1a_list, data_version)
        generated_files.append(write_cdf(dataset))

    for _, de_l1a_list in de_by_day.items():
        dataset = generate_de_dataset(de_l1a_list, data_version)
        # generated_files.append(write_cdf(dataset))

    return generated_files


def process_de_l0(
    de_l0: list[DirectEventL0],
) -> dict[np.datetime64, list[DirectEventL1A]]:
    """
    Process Direct Event packets into GLOWS L1A CDF files.

    This involves combining packets with direct event sequences that span multiple
    packets.

    Parameters
    ----------
    de_l0 : list[DirectEventL0]
        List of DirectEventL0 objects

    Returns
    -------
    de_by_day : dict[np.datetime64, list[DirectEventL1A]]
        Dictionary with keys of days and values of lists of DirectEventL1A objects.
        Each day has one CDF file associated with it.
    """
    de_by_day = dict()

    for de in de_l0:
        de_day = calc_start_time(de.MET).astype("datetime64[D]")
        if de_day not in de_by_day:
            de_by_day[de_day] = [DirectEventL1A(de)]
        elif de.SEQ != 0:
            # If the direct event is part of a sequence and is not the first,
            # append it to the last direct event in the list
            de_by_day[de_day][-1].append(de)
        else:
            de_by_day[de_day].append(DirectEventL1A(de))

    return de_by_day


def generate_de_dataset(
    de_l1a_list: list[DirectEventL1A], data_version: str
) -> xr.Dataset:
    """
    Generate a dataset for GLOWS L1A direct event data CDF files.

    Parameters
    ----------
    de_l1a_list : list[DirectEventL1A]
        List of DirectEventL1A objects for a given day
    data_version : str
        Data version for CDF filename, in the format "vXXX"

    Returns
    -------
    output : xr.Dataset
        Dataset containing the GLOWS L1A direct event CDF output
    """
    # TODO: Block header per second, or global attribute?

    time_data = np.zeros(len(de_l1a_list), dtype="datetime64[ns]")
    # TODO: Should each timestamp point to a list of direct events, each with a
    #  timestamp? Or should the list be split out to make the timestamps?

    # Each DirectEventL1A class covers 1 second of direct events data
    direct_events = np.zeros((len(de_l1a_list), len(de_l1a_list[0].direct_events), 4))

    # In header: block header, missing seqs
    # Time varying - statusdata

    # TODO Check docs to see what should be in here
    support_data = {
        "flight_software_version": [],
        "ground_software_version": [],  # TODO: should this be a global file attribute?
        "seq_count_in_pkts_file": [],
        "imap_start_time": [],
        "imap_time_offset": [],
        "glows_start_time": [],
        "glows_time_offset": [],
    }

    for index, de in enumerate(de_l1a_list):
        # Set the timestamp to the first timestamp of the direct event list
        print("=============")
        # print(de.direct_events)
        # print(de.block_header)
        # print(de.l0.DE_DATA)
        epoch_time = calc_start_time(de.l0.MET).astype("datetime64[ns]")

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
        new_de = np.array([event.to_array() for event in de.direct_events])
        print(direct_events.shape)

        direct_events[index, : len(de.direct_events), :] = new_de
        time_data[index] = epoch_time
        support_data["block_header"].append(de.block_header)

    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    direct_event = xr.DataArray(
        np.arange(4),
        name="direct_event",
        dims=["direct_event"],
        attrs=glows_cdf_attrs.direct_event_attrs.output(),
    )

    # TODO come up with a better name
    per_second = xr.DataArray(
        np.arange(direct_events.shape[1]),
        name="per_second",
        dims=["per_second"],
        attrs=glows_cdf_attrs.per_second_attrs.output(),
    )

    de = xr.DataArray(
        direct_events,
        name="direct_events",
        dims=["epoch", "per_second", "direct_event"],
        coords={
            "epoch": epoch_time,
            "per_second": per_second,
            "direct_event": direct_event,
        },
        attrs=glows_cdf_attrs.direct_event_attrs.output(),
    )

    output = xr.Dataset(
        coords={"epoch": time_data},
        attrs=glows_cdf_attrs.glows_l1a_de_attrs.output(),
    )

    output["direct_events"] = de

    return output


def generate_histogram_dataset(
    hist_l1a_list: list[HistogramL1A], data_version: str
) -> xr.Dataset:
    """
    Generate a dataset for GLOWS L1A histogram data CDF files.

    Parameters
    ----------
    hist_l1a_list : list[HistogramL1A]
        List of HistogramL1A objects for a given day
    data_version : str
        Data version for CDF filename, in the format "vXXX"

    Returns
    -------
    output : xr.Dataset
        Dataset containing the GLOWS L1A histogram CDF output
    """
    time_data = np.zeros(len(hist_l1a_list), dtype="datetime64[ns]")
    # TODO Add daily average of histogram counts
    # TODO compute average temperature etc
    # Data in lists, for each of the 25 time varying datapoints in HistogramL1A

    hist_data = np.zeros((len(hist_l1a_list), 3600))

    # TODO: add missing attributes
    support_data = {
        "flight_software_version": [],
        "ground_software_version": [],  # TODO: should this be a global file attribute?
        "pkts_file_name": [],
        "seq_count_in_pkts_file": [],
        "last_spin_id": [],
        "imap_start_time": [],
        "imap_time_offset": [],
        "glows_start_time": [],
        "glows_time_offset": [],
        "flags_set_onboard": [],
        "is_generated_on_ground": [],
    }

    for index, hist in enumerate(hist_l1a_list):
        # TODO: Should this be MET?
        epoch_time = calc_start_time(hist.imap_start_time.to_seconds())
        hist_data[index] = hist.histograms
        hist_dict = dataclasses.asdict(hist)

        for block_key in hist.block_header.keys():
            support_data[block_key].append(hist.block_header[block_key])

        support_data["flags_set_onboard"].append(hist.flags["flags_set_onboard"])
        support_data["is_generated_on_ground"].append(
            hist.flags["is_generated_on_ground"]
        )

        for key, items in hist_dict.items():
            if key not in ["histograms", "block_header", "flags"]:
                if key in [
                    "imap_start_time",
                    "imap_time_offset",
                    "glows_start_time",
                    "glows_time_offset",
                ]:
                    support_data[key].append(
                        items["seconds"]
                        + items["subseconds"] / GlowsConstants.SUBSECOND_LIMIT
                    )
                else:
                    support_data[key].append(items)

        time_data[index] = epoch_time

    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    bins = xr.DataArray(
        np.arange(3600),  # TODO: Is it always 3600 bins?
        name="bins",
        dims=["bins"],
        attrs=glows_cdf_attrs.bins_attrs.output(),
    )

    hist = xr.DataArray(
        hist_data,
        name="histograms",
        dims=["epoch", "bins"],
        coords={"epoch": epoch_time, "bins": bins},
        attrs=glows_cdf_attrs.histogram_attrs.output(),
    )

    glows_attrs = glows_cdf_attrs.glows_l1a_hist_attrs.output()
    glows_attrs["Data_version"] = data_version

    output = xr.Dataset(
        coords={"epoch": epoch_time, "bins": bins},
        attrs=glows_cdf_attrs.glows_l1a_hist_attrs.output(),
    )

    output["histograms"] = hist

    # for key, value in support_data.items():
    #     output[key] = xr.DataArray(
    #         value,
    #         name=key,
    #         dims=["epoch"],
    #         coords={"epoch": epoch_time},
    #         attrs=dataclasses.replace(
    #             glows_cdf_attrs.metadata_attrs,
    #             catdesc=glows_cdf_attrs.catdesc_fieldname_l1a[key][0],
    #             fieldname=glows_cdf_attrs.catdesc_fieldname_l1a[key][1],
    #         ).output(),
    #     )

    return output
