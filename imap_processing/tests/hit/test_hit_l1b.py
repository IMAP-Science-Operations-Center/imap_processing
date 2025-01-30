import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a import hit_l1a
from imap_processing.hit.l1b import hit_l1b

# TODO: Packet files are per apid at the moment so the tests currently
#  reflect this. Eventually, HIT will provide a packet file with all apids
#  and the tests will need to be updated.


@pytest.fixture(scope="module")
def packet_filepath():
    """Set path to test data file"""
    # TODO: Update this path when HIT provides a packet file with all apids.
    #  Current test file only has the housekeeping apid is available.
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"


@pytest.fixture()
def dependencies(packet_filepath):
    """Get dependencies for L1B processing"""
    # Create dictionary of dependencies and add CCSDS packet file
    data_dict = {"imap_hit_l0_raw": packet_filepath}
    # Add L1A datasets
    l1a_datasets = hit_l1a.hit_l1a(packet_filepath, "001")
    l1a_datasets.extend(
        hit_l1a.hit_l1a(
            imap_module_directory / "tests/hit/test_data/sci_sample.ccsds", "001"
        )
    )
    for dataset in l1a_datasets:
        data_dict[dataset.attrs["Logical_source"]] = dataset
    return data_dict


@pytest.fixture()
def l1b_hk_dataset(dependencies):
    """Get the housekeeping dataset"""
    datasets = hit_l1b.hit_l1b(dependencies, "001")
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_hk":
            return dataset


@pytest.fixture()
def l1b_standard_rates_dataset(dependencies):
    """Get the standard rates dataset"""
    # TODO: use this fixture in future unit test to validate the standard rates dataset
    datasets = hit_l1b.hit_l1b(dependencies, "001")
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1b_standard-rates":
            return dataset


@pytest.fixture()
def l1a_counts_dataset(sci_packet_filepath):
    """Get L1A counts dataset to test l1b processing functions"""
    l1a_datasets = hit_l1a.hit_l1a(sci_packet_filepath, "001")
    for dataset in l1a_datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1a_count-rates":
            return dataset


def test_process_standard_rates_data(l1a_counts_dataset):
    """Test function for processing standard rates data"""
    l1b_standard_rates_dataset = hit_l1b.process_standard_rates_data(l1a_counts_dataset)

    # Check that a xarray dataset with the correct logical source is returned
    assert isinstance(l1b_standard_rates_dataset, xr.Dataset)

    # Define the data variables that should be present in the dataset
    valid_data_vars = {
        "sngrates",
        "coinrates",
        "pbufrates",
        "l2fgrates",
        "l2bgrates",
        "l3fgrates",
        "l3bgrates",
        "penfgrates",
        "penbgrates",
        "ialirtrates",
        "l4fgrates",
        "l4bgrates",
        "sngrates_delta_plus",
        "coinrates_delta_plus",
        "pbufrates_delta_plus",
        "l2fgrates_delta_plus",
        "l2bgrates_delta_plus",
        "l3fgrates_delta_plus",
        "l3bgrates_delta_plus",
        "penfgrates_delta_plus",
        "penbgrates_delta_plus",
        "ialirtrates_delta_plus",
        "l4fgrates_delta_plus",
        "l4bgrates_delta_plus",
        "sngrates_delta_minus",
        "coinrates_delta_minus",
        "pbufrates_delta_minus",
        "l2fgrates_delta_minus",
        "l2bgrates_delta_minus",
        "l3fgrates_delta_minus",
        "l3bgrates_delta_minus",
        "penfgrates_delta_minus",
        "penbgrates_delta_minus",
        "ialirtrates_delta_minus",
        "l4fgrates_delta_minus",
        "l4bgrates_delta_minus",
        "dynamic_threshold_state",
    }

    valid_coords = [
        "epoch",
        "gain",
        "sngrates_index",
        "coinrates_index",
        "pbufrates_index",
        "l2fgrates_index",
        "l2bgrates_index",
        "l3fgrates_index",
        "l3bgrates_index",
        "penfgrates_index",
        "penbgrates_index",
        "ialirtrates_index",
        "l4fgrates_index",
        "l4bgrates_index",
    ]

    # Check that the dataset has the correct variables
    assert valid_data_vars == set(
        l1b_standard_rates_dataset.data_vars.keys()
    ), "Data variables mismatch"
    assert valid_coords == list(
        l1b_standard_rates_dataset.coords
    ), "Coordinates mismatch"


def test_hit_l1b_hk_dataset_variables(l1b_hk_dataset):
    """Test the variables in the housekeeping dataset"""
    # Define the keys that should have dropped from the housekeeping dataset
    dropped_keys = {
        "pkt_apid",
        "sc_tick",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }
    # Define the keys that should be present in the housekeeping dataset
    valid_keys = {
        "heater_on",
        "fsw_version_b",
        "ebox_m12va",
        "phasic_stat",
        "ebox_3d4vd",
        "ebox_p2d0vd",
        "temp1",
        "last_bad_seq_num",
        "ebox_m5d7va",
        "ebox_p12va",
        "table_status",
        "enable_50khz",
        "mram_disabled",
        "temp3",
        "preamp_l1a",
        "l2ab_bias",
        "l34b_bias",
        "fsw_version_c",
        "num_evnt_last_hk",
        "dac1_enable",
        "preamp_l234b",
        "analog_temp",
        "fee_running",
        "fsw_version_a",
        "num_errors",
        "test_pulser_on",
        "dac0_enable",
        "preamp_l1b",
        "l1ab_bias",
        "l34a_bias",
        "leak_i",
        "last_good_cmd",
        "lvps_temp",
        "idpu_temp",
        "temp2",
        "preamp_l234a",
        "last_good_seq_num",
        "num_good_cmds",
        "heater_control",
        "hvps_temp",
        "ebox_p5d7va",
        "spin_period_long",
        "enable_hvps",
        "temp0",
        "spin_period_short",
        "dyn_thresh_lvl",
        "num_bad_cmds",
        "adc_mode",
        "ebox_5d1vd",
        "active_heater",
        "last_error_num",
        "last_bad_cmd",
        "ref_p5v",
        "code_checksum",
        "mode",
    }
    # Check that the dataset has the correct variables
    assert valid_keys == set(l1b_hk_dataset.data_vars.keys())
    assert set(dropped_keys).isdisjoint(set(l1b_hk_dataset.data_vars.keys()))

    # Define the coordinates and dimensions. Both have equivalent values
    dataset_coords_dims = {"epoch", "adc_channels", "adc_channels_label"}

    # Check that the dataset has the correct coordinates, and dimensions
    assert l1b_hk_dataset.coords.keys() == dataset_coords_dims


def test_validate_l1b_hk_data(l1b_hk_dataset):
    """Validate the housekeeping dataset created by the L1B processing.

    Parameters
    ----------
    hk_dataset : xr.Dataset
        Housekeeping dataset created by the L1B processing.
    """
    # TODO: finish test. HIT will provide an updated validation file to fix issues:
    #  - some fields have strings as values but in the processed data they're integers
    #  - Some columns have blank cells where there should be data

    # Load the validation data
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/hskp_sample_eu.csv"
    )
    validation_data = pd.read_csv(validation_file)
    validation_data.columns = validation_data.columns.str.lower().str.strip()

    # Get a list of leak columns in ascending order
    # (LEAK_I_00, LEAK_I_01, ..., LEAK_I_63)
    # and group values into a single column
    leak_columns = [
        col for col in validation_data.columns if col.startswith("leak_i_")
    ][::-1]
    validation_data["leak_i"] = validation_data[leak_columns].apply(
        lambda row: row.values, axis=1
    )
    validation_data.drop(columns=leak_columns, inplace=True)

    # Define the keys that should have dropped from the housekeeping dataset
    dropped_fields = {
        "pkt_apid",
        "sc_tick",
        "version",
        "type",
        "sec_hdr_flg",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "hskp_spare1",
        "hskp_spare2",
        "hskp_spare3",
        "hskp_spare4",
        "hskp_spare5",
    }

    # Check that dropped variables are not in the dataset
    assert set(dropped_fields).isdisjoint(set(l1b_hk_dataset.data_vars.keys()))

    # TODO: uncomment block after new validation data is provided
    # Define the keys that should be ignored in the validation
    # like ccsds headers
    # ignore_validation_fields = {
    #     "ccsds_version",
    #     "ccsds_type",
    #     "ccsds_sec_hdr_flag",
    #     "ccsds_appid",
    #     "ccsds_grp_flag",
    #     "ccsds_seq_cnt",
    #     "ccsds_length",
    #     "sc_tick",
    # }

    # # Compare the housekeeping dataset with the expected validation data
    # for field in validation_data.columns:
    #     if field not in ignore_validation_fields:
    #         print(field)
    #         assert field in hk_dataset.data_vars.keys()
    #         for pkt in range(validation_data.shape[0]):
    #             assert np.array_equal(
    #                 hk_dataset[field][pkt].data, validation_data[field][pkt]
    #             )


def test_hit_l1b(dependencies):
    """Test creating L1B CDF files

    Creates a list of xarray datasets for each L1B product

    Parameters
    ----------
    dependencies : dict
        Dictionary of L1A datasets and CCSDS packet file path
    """
    # TODO: update assertions after science data processing is completed
    datasets = hit_l1b.hit_l1b(dependencies, "001")

    assert len(datasets) == 2
    for dataset in datasets:
        assert isinstance(dataset, xr.Dataset)
    assert datasets[0].attrs["Logical_source"] == "imap_hit_l1b_hk"
    assert datasets[1].attrs["Logical_source"] == "imap_hit_l1b_standard-rates"
