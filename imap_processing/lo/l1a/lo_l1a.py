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
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


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
    datasets_by_apid_derived = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
        use_derived_value=True,
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

        # For segmented packets, combine raw bytes directly
        # Always call combine_segmented_packets to set MET field
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
    if LoAPID.ILO_DIAG_PCC in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_DIAG_PCC).name} "
            f"packet (APID: {LoAPID.ILO_DIAG_PCC.value})"
        )
        logical_source = "imap_lo_l1a_pcc"
        ds = datasets_by_apid[LoAPID.ILO_DIAG_PCC]
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if LoAPID.ILO_APP_NHK in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_APP_NHK).name} "
            f"packet (APID: {LoAPID.ILO_APP_NHK.value})"
        )
        logical_source = "imap_lo_l1a_nhk"
        ds = datasets_by_apid[LoAPID.ILO_APP_NHK]
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)

        # Engineering units conversion
        logical_source = "imap_lo_l1b_nhk"
        ds = datasets_by_apid_derived[LoAPID.ILO_APP_NHK]
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if LoAPID.ILO_APP_SHK in datasets_by_apid:
        logger.info(
            f"\nProcessing {LoAPID(LoAPID.ILO_APP_SHK).name} "
            f"packet (APID: {LoAPID.ILO_APP_SHK.value})"
        )
        logical_source = "imap_lo_l1a_shk"
        ds = datasets_by_apid[LoAPID.ILO_APP_SHK]
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)

        # Engineering units conversion
        logical_source = "imap_lo_l1b_shk"
        ds = datasets_by_apid_derived[LoAPID.ILO_APP_SHK]
        ds = add_dataset_attrs(ds, attr_mgr, logical_source)
        datasets_to_return.append(ds)
    if any(
        apid in datasets_by_apid
        for apid in [LoAPID.ILO_APP_NHK, LoAPID.ILO_APP_SHK, LoAPID.ILO_DIAG_PCC]
    ):
        logger.info("\nCreating instrument state vector")
        logical_source = "imap_lo_l1b_instrument-status-summary"
        combined_ds = instrument_status_summary(datasets_by_apid_derived)
        combined_ds = add_dataset_attrs(combined_ds, attr_mgr, logical_source)
        datasets_to_return.append(combined_ds)

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

        # For DEs, shcoarse applies to each ASC group and is not per individual epoch
        # so we can't depend on epoch in the attributes
        dataset["shcoarse"].attrs.pop("DEPEND_0")
        dataset["shcoarse"].attrs["DEPEND_1"] = "shcoarse"

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


def instrument_status_summary(datasets_by_apid_derived: dict) -> xr.Dataset:
    """
    Combine relevant datasets to create instrument status summary.

    This is from section 9.1.1 of the IMAP-Lo Algorithm Document. We
    combine the NHK, SHK, and PCC datasets to create a single data
    product containing the instrument status summary. These are not
    all aligned in time, so we simply merge them and fill NaNs where
    the times are not aligned across data products.

    Parameters
    ----------
    datasets_by_apid_derived : dict
        Dictionary of datasets keyed by APID with derived values.

    Returns
    -------
    xr.Dataset
        Combined dataset containing the instrument state vector.
    """
    datasets_to_merge = [
        ds
        for apid, ds in datasets_by_apid_derived.items()
        if apid in [LoAPID.ILO_APP_NHK, LoAPID.ILO_APP_SHK, LoAPID.ILO_DIAG_PCC]
    ]
    combined_ds = xr.merge(datasets_to_merge, compat="override")
    # Only keep the desired keys for the state vector
    status_summary_keys = [
        "shcoarse",
        "op_mode",
        "pac_vset",
        "mcp_vset",
        "tof_mcp_v",
        "bhv_def_neg_dac",
        "bhv_def_pos_dac",
        "bhv_pmt_dac",
        "fsw_version_str",
        "eng_lut_version_str",
        "sci_lut_version_str",
        "an_a_thr",
        "an_b0_thr",
        "an_b3_thr",
        "an_c_thr",
        "tof3_thr",
        "tof2_thr",
        "tof1_thr",
        "tof0_thr",
        "ifb_hot_spot_t",
        "coarse_pot_pri",
        "fine_pot_pri",
    ]
    vars_present = set(status_summary_keys).intersection(
        set(combined_ds.data_vars.keys())
    )
    combined_ds = combined_ds[list(vars_present)]
    # Add variables that are missing but expected to be in the state vector
    for key in status_summary_keys:
        if key not in combined_ds:
            if key in ["fsw_version_str", "eng_lut_version_str", "sci_lut_version_str"]:
                combined_ds[key] = xr.DataArray(
                    data=np.full(combined_ds["epoch"].shape, "", dtype=str),
                    dims=["epoch"],
                )
            else:
                combined_ds[key] = xr.DataArray(
                    data=np.full(combined_ds["epoch"].shape, np.nan, dtype=float),
                    dims=["epoch"],
                )
    return combined_ds
