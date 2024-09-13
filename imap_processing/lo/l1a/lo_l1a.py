"""IMAP-Lo L1A Data Processing."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.lo_science import parse_histogram
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lo_l1a(dependency: Path, data_version: str) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L0 data into L1A CDF data products.

    Parameters
    ----------
    dependency : Path
        Dependency file needed for data product creation.
        Should always be only one for L1A.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    xtce_file = imap_module_directory / "lo/packet_definitions/lo_xtce.xml"

    logger.info("\nDecommutating packets and converting to dataset")
    datasets_by_apid = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
        use_derived_value=False,
    )

    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    logger.info(
        f"\nProcessing {LoAPID(LoAPID.ILO_SCI_CNT).name} "
        f"packet (APID: {LoAPID.ILO_SCI_CNT.value})"
    )
    if LoAPID.ILO_SCI_CNT in datasets_by_apid:
        logical_source = "imap_lo_l1a_histogram"
        datasets_by_apid[LoAPID.ILO_SCI_CNT] = parse_histogram(
            datasets_by_apid[LoAPID.ILO_SCI_CNT], attr_mgr
        )
        datasets_by_apid[LoAPID.ILO_SCI_CNT] = add_dataset_attrs(
            datasets_by_apid[LoAPID.ILO_SCI_CNT], attr_mgr, logical_source
        )

    good_apids = [LoAPID.ILO_SCI_CNT]
    logger.info(f"\nReturning datasets: {[LoAPID(apid) for apid in good_apids]}")
    return [datasets_by_apid[good_apid] for good_apid in good_apids]


def add_dataset_attrs(
    dataset: xr.Dataset, attr_mgr: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Add Attributes to the dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Lo dataset from packets_to_dataset function.
    attr_mgr : ImapCdfAttributes
        CDF attribute manager for Lo L1A.
    logical_source : str
        Logical source for the data.

    Returns
    -------
    dataset : xr.Dataset
        Data with attributes added.
    """
    # TODO: may want up split up these if statements into their
    # own functions
    if logical_source == "imap_lo_l1a_histogram":
        azimuth_60 = xr.DataArray(
            data=np.arange(0, 6, dtype=np.uint8),
            name="azimuth_60",
            dims=["azimuth_60"],
            attrs=attr_mgr.get_variable_attributes("azimuth_60"),
        )
        azimuth_60_label = xr.DataArray(
            data=azimuth_60.values.astype(str),
            name="azimuth_60_label",
            dims=["azimuth_60_label"],
            attrs=attr_mgr.get_variable_attributes("azimuth_60_label"),
        )
        azimuth_6 = xr.DataArray(
            data=np.arange(0, 60, dtype=np.uint8),
            name="azimuth_6",
            dims=["azimuth_6"],
            attrs=attr_mgr.get_variable_attributes("azimuth_6"),
        )
        azimuth_6_label = xr.DataArray(
            data=azimuth_6.values.astype(str),
            name="azimuth_6_label",
            dims=["azimuth_6_label"],
            attrs=attr_mgr.get_variable_attributes("azimuth_6_label"),
        )

        esa_step = xr.DataArray(
            data=np.arange(1, 8, dtype=np.uint8),
            name="esa_step",
            dims=["esa_step"],
            attrs=attr_mgr.get_variable_attributes("esa_step"),
        )
        esa_step_label = xr.DataArray(
            esa_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )

        dataset.shcoarse.attrs.update(attr_mgr.get_variable_attributes("shcoarse"))
        dataset.epoch.attrs.update(attr_mgr.get_variable_attributes("epoch"))

        dataset = dataset.assign_coords(
            azimuth_60=azimuth_60,
            azimuth_60_label=azimuth_60_label,
            azimuth_6=azimuth_6,
            azimuth_6_label=azimuth_6_label,
            esa_step=esa_step,
            esa_step_label=esa_step_label,
        )
        dataset.attrs.update(attr_mgr.get_global_attributes(logical_source))
        # remove the binary field and CCSDS header from the dataset
        dataset = dataset.drop_vars(
            [
                "sci_cnt",
                "chksum",
                "version",
                "type",
                "sec_hdr_flg",
                "pkt_apid",
                "seq_flgs",
                "src_seq_ctr",
                "pkt_len",
            ]
        )

    return dataset
