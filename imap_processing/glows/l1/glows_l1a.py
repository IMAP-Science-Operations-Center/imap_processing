"""Methods for GLOWS Level 1A processing and CDF writing."""

import dataclasses
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.glows import glows_cdf_attrs
from imap_processing.glows.l0.decom_glows import decom_packets
from imap_processing.glows.l1.glows_l1a_data import HistogramL1A
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

    histogram_buckets = dict()

    # Create histogram L1A and filter into days based on start time
    for hist in hist_l0:
        hist_l1a = HistogramL1A(hist)
        hist_day = calc_start_time(hist.MET).astype("datetime64[D]")
        if hist_day not in histogram_buckets:
            histogram_buckets[hist_day] = []
        histogram_buckets[hist_day].append(hist_l1a)

    # Generate CDF files for each day
    generated_files = []
    for _, hist_l1a_list in histogram_buckets.items():
        dataset = generate_histogram_dataset(hist_l1a_list, data_version)
        generated_files.append(write_cdf(dataset))

    return generated_files


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

    for key, value in support_data.items():
        output[key] = xr.DataArray(
            value,
            name=key,
            dims=["epoch"],
            coords={"epoch": epoch_time},
            attrs=dataclasses.replace(
                glows_cdf_attrs.metadata_attrs,
                catdesc=glows_cdf_attrs.catdesc_fieldname_l1a[key][0],
                fieldname=glows_cdf_attrs.catdesc_fieldname_l1a[key][1],
            ).output(),
        )

    return output
