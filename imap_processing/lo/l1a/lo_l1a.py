"""IMAP-Lo L1A Data Processing."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.lo_science import (
    combine_segmented_packets,
    organize_spin_data,
    parse_events,
    parse_histogram,
)
from imap_processing.lo.l0.lo_star_sensor import process_star_sensor
from imap_processing.utils import convert_to_binary_string, packet_file_to_datasets

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def lo_l1a(dependency: Path) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L0 data into L1A CDF data products.

    Parameters
    ----------
    dependency : Path
        Dependency file needed for data product creation.
        Should always be only one for L1A.

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

    datasets_to_return = []

    if LoAPID.ILO_SPIN in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_SPIN).name} "
            f"packet (APID: {LoAPID.ILO_SPIN.value})"
        )
        logical_source = "imap_lo_l1a_spin"
        ds = datasets_by_apid[LoAPID.ILO_SPIN]
        ds = organize_spin_data(ds, attr_mgr)
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if LoAPID.ILO_SCI_CNT in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_SCI_CNT).name} "
            f"packet (APID: {LoAPID.ILO_SCI_CNT.value})"
        )
        logical_source = "imap_lo_l1a_histogram"
        ds = datasets_by_apid[LoAPID.ILO_SCI_CNT]
        ds = parse_histogram(ds, attr_mgr)
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if LoAPID.ILO_SCI_DE in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_SCI_DE).name} "
            f"packet (APID: {LoAPID.ILO_SCI_DE.value})"
        )
        logical_source = "imap_lo_l1a_de"
        ds = datasets_by_apid[LoAPID.ILO_SCI_DE]
        # Process the "data" array into a string
        ds["data"] = xr.DataArray(
            [convert_to_binary_string(data) for data in ds["data"].values],
            dims=ds["data"].dims,
            attrs=ds["data"].attrs,
        )

        ds = combine_segmented_packets(ds)
        ds = parse_events(ds, attr_mgr)
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if LoAPID.ILO_STAR in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_STAR).name} "
            f"packet (APID: {LoAPID.ILO_STAR.value})"
        )
        logical_source = "imap_lo_l1a_star"
        ds = datasets_by_apid[LoAPID.ILO_STAR]
        ds = process_star_sensor(ds)
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)

    logger.info(f"Returning [{len(datasets_to_return)}] datasets")
    return datasets_to_return


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
    #  own functions
    # Get global attributes
    dataset.attrs.update(attr_mgr.get_global_attributes(logical_source))
    # Get attributes for shcoarse and epoch
    dataset.shcoarse.attrs.update(attr_mgr.get_variable_attributes("shcoarse"))
    dataset.epoch.attrs.update(attr_mgr.get_variable_attributes("epoch"))

    if logical_source == "imap_lo_l1a_spin":
        spin = xr.DataArray(
            data=np.arange(0, 28, dtype=np.uint8),
            name="spin",
            dims=["spin"],
            attrs=attr_mgr.get_variable_attributes("spin"),
        )
        spin_label = xr.DataArray(
            data=spin.values.astype(str),
            name="spin_label",
            dims=["spin_label"],
            attrs=attr_mgr.get_variable_attributes("spin_label"),
        )

        dataset = dataset.assign_coords(spin=spin, spin_label=spin_label)
        dataset.num_completed.attrs.update(
            attr_mgr.get_variable_attributes("num_completed")
        )
        dataset.acq_start_sec.attrs.update(
            attr_mgr.get_variable_attributes("acq_start_sec")
        )
        dataset.acq_start_subsec.attrs.update(
            attr_mgr.get_variable_attributes("acq_start_subsec")
        )
        dataset.acq_end_sec.attrs.update(
            attr_mgr.get_variable_attributes("acq_end_sec")
        )
        dataset.acq_end_subsec.attrs.update(
            attr_mgr.get_variable_attributes("acq_end_subsec")
        )

        dataset = dataset.drop_vars(
            [
                "version",
                "type",
                "sec_hdr_flg",
                "pkt_apid",
                "seq_flgs",
                "src_seq_ctr",
                "pkt_len",
                "chksum",
            ]
        )
        # An empty DEPEND_0 is being added to support_data
        # variables that should only have DEPEND_1
        # Removing Depend_0 here.
        # TODO: Should look for a fix to this issue
        del dataset["spin"].attrs["DEPEND_0"]

    elif logical_source == "imap_lo_l1a_histogram":
        # Create coordinates for the dataset
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
            attrs=attr_mgr.get_variable_attributes("esa_step_coord"),
        )
        esa_step_label = xr.DataArray(
            esa_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )

        dataset = dataset.assign_coords(
            azimuth_60=azimuth_60,
            azimuth_60_label=azimuth_60_label,
            azimuth_6=azimuth_6,
            azimuth_6_label=azimuth_6_label,
            esa_step=esa_step,
            esa_step_label=esa_step_label,
        )
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
        # An empty DEPEND_0 is being added to support_data
        # variables that should only have DEPEND_1
        # Removing Depend_0 here.
        # TODO: Should look for a fix to this issue
        del dataset["azimuth_60"].attrs["DEPEND_0"]
        del dataset["azimuth_6"].attrs["DEPEND_0"]
        del dataset["esa_step"].attrs["DEPEND_0"]

    elif logical_source == "imap_lo_l1a_de":
        # Create the coordinates for the dataset
        direct_events = xr.DataArray(
            data=np.arange(sum(dataset["de_count"].values), dtype=np.uint16),
            name="direct_events",
            dims=["direct_events"],
            attrs=attr_mgr.get_variable_attributes("direct_events"),
        )

        direct_events_label = xr.DataArray(
            direct_events.values.astype(str),
            name="direct_events_label",
            dims=["direct_events_label"],
            attrs=attr_mgr.get_variable_attributes("direct_events_label"),
        )

        dataset = dataset.assign_coords(
            direct_events=direct_events,
            direct_events_label=direct_events_label,
        )
        # add the epoch and global attributes
        dataset = dataset.drop_vars(
            [
                "version",
                "type",
                "sec_hdr_flg",
                "pkt_apid",
                "seq_flgs",
                "src_seq_ctr",
                "pkt_len",
                "data",
                "events",
            ]
        )
        # An empty DEPEND_0 is being added to support_data
        # variables that should only have DEPEND_1
        # Removing Depend_0 here.
        # TODO: Should look for a fix to this issue
        for var in [
            "direct_events",
            "coincidence_type",
            "de_time",
            "mode",
            "esa_step",
            "tof0",
            "tof1",
            "tof2",
            "tof3",
            "pos",
            "cksm",
        ]:
            dataset[var].attrs.pop("DEPEND_0")

    return dataset
