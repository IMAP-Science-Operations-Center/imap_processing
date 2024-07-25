"""IMAP-HI L1B processing module."""

import logging

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import HIAPID
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)
ATTR_MGR = ImapCdfAttributes()
ATTR_MGR.add_instrument_global_attrs("hi")
ATTR_MGR.load_variable_attributes("imap_hi_variable_attrs.yaml")


def hi_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    High level IMAP-HI L1B processing function.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        L1A dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        Processed xarray dataset.
    """
    logger.info(
        f"Running Hi L1B processing on dataset: {l1a_dataset.attrs['Logical_source']}"
    )
    logical_source_parts = l1a_dataset.attrs["Logical_source"].split("_")
    # TODO: apid is not currently stored in all L1A data but should be.
    #    Use apid to determine what L1B processing function to call

    # Housekeeping processing
    if logical_source_parts[-1].endswith("hk"):
        # if packet_enum in (HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK):
        packet_enum = HIAPID(l1a_dataset["pkt_apid"].data[0])
        conversion_table_path = str(
            imap_module_directory / "hi" / "l1b" / "hi_eng_unit_convert_table.csv"
        )
        l1b_dataset = convert_raw_to_eu(
            l1a_dataset,
            conversion_table_path=conversion_table_path,
            packet_name=packet_enum.name,
            comment="#",  # type: ignore[arg-type]
            # Todo error, Argument "comment" to "convert_raw_to_eu" has incompatible
            # type "str"; expected "dict[Any, Any]"
            converters={"mnemonic": str.lower},
        )

        l1b_dataset.attrs.update(ATTR_MGR.get_global_attributes("imap_hi_l1b_hk_attrs"))
    elif logical_source_parts[-1].endswith("de"):
        l1b_dataset = annotate_direct_events(l1a_dataset)
    else:
        raise NotImplementedError(
            f"No Hi L1B processing defined for file type: "
            f"{l1a_dataset.attrs['Logical_source']}"
        )
    # Update global attributes
    # TODO: write a function that extracts the sensor from Logical_source
    #    some functionality can be found in imap_data_access.file_validation but
    #    only works on full file names
    sensor_str = logical_source_parts[-1].split("-")[0]
    l1b_dataset.attrs["Logical_source"] = l1b_dataset.attrs["Logical_source"].format(
        sensor=sensor_str
    )
    # TODO: revisit this
    l1b_dataset.attrs["Data_version"] = data_version
    return l1b_dataset


def annotate_direct_events(l1a_dataset: xr.Dataset) -> xr.Dataset:
    """
    Perform Hi L1B processing on direct event data.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        L1A direct event data.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        L1B direct event data.
    """
    n_epoch = l1a_dataset["epoch"].size
    new_data_vars = dict()
    for var in [
        "coincidence_type",
        "esa_step",
        "delta_t_ab",
        "delta_t_ac1",
        "delta_t_bc1",
        "delta_t_c1c2",
        "spin_phase",
        "hae_latitude",
        "hae_longitude",
        "quality_flag",
        "nominal_bin",
    ]:
        attrs = ATTR_MGR.get_variable_attributes(
            f"hi_de_{var}", check_schema=False
        ).copy()
        dtype = attrs.pop("dtype")
        if attrs["FILLVAL"] == "NaN":
            attrs["FILLVAL"] = np.nan
        new_data_vars[var] = xr.DataArray(
            data=np.full(n_epoch, attrs["FILLVAL"], dtype=np.dtype(dtype)),
            dims=["epoch"],
            attrs=attrs,
        )
    l1b_dataset = l1a_dataset.assign(new_data_vars)
    l1b_dataset = l1b_dataset.drop_vars(
        ["tof_1", "tof_2", "tof_3", "de_tag", "ccsds_met", "meta_event_met"]
    )

    de_global_attrs = ATTR_MGR.get_global_attributes("imap_hi_l1b_de_attrs")
    l1b_dataset.attrs.update(**de_global_attrs)
    return l1b_dataset
