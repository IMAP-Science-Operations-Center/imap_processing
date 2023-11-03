import collections
from enum import IntEnum

import numpy as np
import xarray as xr

from imap_processing import cdf_utils
from imap_processing.swe import swe_cdf_attrs


class SWEAPID(IntEnum):
    """Create ENUM for apid.

    Parameters
    ----------
    IntEnum : IntEnum
    """

    SWE_APP_HK = 1330
    SWE_EVTMSG = 1317
    SWE_CEM_RAW = 1334
    SWE_SCIENCE = 1344


filename_descriptors = {
    SWEAPID.SWE_APP_HK: "hk",
    SWEAPID.SWE_EVTMSG: "evtmsg",
    SWEAPID.SWE_CEM_RAW: "cemraw",
    SWEAPID.SWE_SCIENCE: "sci",
}


def add_metadata_to_array(data_packet, metadata_arrays):
    """Add metadata to the metadata_arrays.

    Parameters
    ----------
    data_packet : space_packet_parser.parser.Packet
        SWE data packet
    metadata_arrays : dict
        metadata arrays
    """
    for key, value in data_packet.header.items():
        metadata_arrays.setdefault(key, []).append(value.raw_value)

    for key, value in data_packet.data.items():
        if key == "SCIENCE_DATA":
            continue
        elif key == "APP_MODE":
            # We need to get derived value for this because it's used in
            # filename.
            metadata_arrays.setdefault(key, []).append(
                value.raw_value
                if value.derived_value is None
                else str.lower(value.derived_value)
            )
        else:
            metadata_arrays.setdefault(key, []).append(value.raw_value)

    return metadata_arrays


def create_dataset(packets):
    """Create dataset for each metadata field.

    Parameters
    ----------
    packets : list
        packet list

    Returns
    -------
    xr.dataset
        dataset with all metadata field data in xr.DataArray
    """
    metadata_arrays = collections.defaultdict(list)

    for data_packet in packets:
        add_metadata_to_array(data_packet, metadata_arrays)

    epoch_time = xr.DataArray(
        metadata_arrays["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=cdf_utils.epoch_attrs,
    )

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time},
        attrs=swe_cdf_attrs.swe_l1a_global_attrs,
    )

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        if key == "SHCOARSE":
            continue
        elif key == "APP_MODE":
            string_attrs = swe_cdf_attrs.string_attrs
            string_attrs["CATDESC"] = string_attrs["FIELDNAM"] = "APP_MODE"
            dataset[key] = xr.DataArray(
                value,
                dims=["Epoch"],
                attrs=string_attrs,
            )
        else:
            int_attrs = swe_cdf_attrs.int_attrs
            int_attrs["CATDESC"] = int_attrs["FIELDNAM"] = int_attrs["LABLAXIS"] = key
            # get int32's max since most of metadata is under 32-bits
            int_attrs["VALIDMAX"] = np.iinfo(np.int32).max
            dataset[key] = xr.DataArray(
                value,
                dims=["Epoch"],
                attrs=int_attrs,
            )
    return dataset
