from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_datasets_by_apid,
)
from imap_processing.hit.l1a.hit_l1a import (
    add_cdf_attributes,
    calculate_uncertainties,
    decom_hit,
    filter_dataset_to_processing_day,
    find_complete_mod10_sets,
    hit_l1a,
    subcom_sectorates,
    subset_livetime,
    subset_sectored_counts,
    update_livetime_coord,
)
from imap_processing.tests.hit.helpers.l1_validation import (
    compare_data,
    prepare_counts_validation_data,
)

# TODO: Packet files are per apid at the moment so the tests currently
#  reflect this. Eventually, HIT will provide a packet file with all apids
#  and the tests will need to be updated.


@pytest.fixture(scope="module")
def hk_packet_filepath():
    """Set path to test data file"""
    return Path(imap_module_directory / "tests/hit/test_data/hskp_sample.ccsds")


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return Path(imap_module_directory / "tests/hit/test_data/sci_sample.ccsds")


@pytest.fixture(scope="module")
def validation_data():
    """Load validation data from CSV file."""
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/sci_sample_raw.csv"
    )
    validation_data = pd.read_csv(validation_file)
    return validation_data


# <=== TESTS ===>


def test_subcom_sectorates(sci_packet_filepath):
    """Test the subcom_sectorates function.

    This function organizes the sectored rates data
    by species and adds the data as new variables
    to the dataset.
    """

    # Unpack and decompress ccsds file to xarray datasets
    sci_dataset = get_datasets_by_apid(sci_packet_filepath)[HitAPID.HIT_SCIENCE]
    sci_dataset = decom_hit(sci_dataset)

    # Call the function to be tested
    sci_dataset = subcom_sectorates(sci_dataset)

    # Number of science frames in the dataset
    frames = sci_dataset["epoch"].shape[0]

    # Shape of the new data variables
    expected_shapes = {
        "h": (3, 15, 8),
        "he4": (2, 15, 8),
        "cno": (2, 15, 8),
        "nemgsi": (2, 15, 8),
        "fe": (1, 15, 8),
    }

    for species, shape in expected_shapes.items():
        # Check if the dataset has the new  data variables
        assert f"{species}_sectored_counts" in sci_dataset
        assert f"{species}_energy_mean" in sci_dataset.coords
        assert f"{species}_energy_delta_minus" in sci_dataset
        assert f"{species}_energy_delta_plus" in sci_dataset
        # Check the shape of the new data variables
        assert sci_dataset[f"{species}_sectored_counts"].shape == (frames, *shape)
        assert sci_dataset[f"{species}_energy_mean"].shape == (shape[0],)
        assert sci_dataset[f"{species}_energy_delta_minus"].shape == (shape[0],)
        assert sci_dataset[f"{species}_energy_delta_plus"].shape == (shape[0],)


def test_update_livetime_coord():
    """Test the update_livetime_coord function."""

    # Create a mock dataset with livetime_counter and epoch
    epoch_values = np.array([1, 2, 3, 4, 5])
    livetime_values = np.array([10, 20, 30, 40, 50])

    sectored_dataset = xr.Dataset(
        {
            "livetime_counter": ("epoch", livetime_values),
        },
        coords={
            "epoch": epoch_values,
        },
    )

    # Call the function
    updated_dataset = update_livetime_coord(sectored_dataset)

    # Assert the new coordinate 'epoch_livetime' exists
    assert "epoch_livetime" in updated_dataset.coords

    # Assert the values of 'epoch_livetime' match the original epoch values
    np.testing.assert_array_equal(
        updated_dataset["epoch_livetime"].values, epoch_values
    )

    # Assert the dimension of 'livetime_counter' is swapped to 'epoch_livetime'
    assert updated_dataset["livetime_counter"].dims == ("epoch_livetime",)

    # Assert the values of 'livetime_counter' remain unchanged
    np.testing.assert_array_equal(
        updated_dataset["livetime_counter"].values, livetime_values
    )

    # Assert the original 'epoch' coordinate is still present
    assert "epoch" in updated_dataset.coords


def test_subset_livetime():
    """Test the subset_livetime function."""

    # Test case 1: Normal case with valid epoch and epoch_livetime sizes

    # epoch_livetime goes from 0 to 59 (simulates 60 minutes)
    epoch_livetime = np.arange(60)

    # epoch is 30 minute subset of epoch_livetime
    epoch = epoch_livetime[10:40]

    dataset = xr.Dataset(
        {
            "livetime_counter": xr.DataArray(
                np.arange(60),
                dims=["epoch_livetime"],
                coords={"epoch_livetime": epoch_livetime},
            )
        },
        coords={"epoch": epoch, "epoch_livetime": epoch_livetime},
    )

    # Call the function
    trimmed = subset_livetime(dataset)

    # The trimmed livetime should match the slice from index 0 to 29 (inclusive)
    expected_indices = slice(0, 30)
    expected_livetime = dataset.isel(epoch_livetime=expected_indices)

    # Validate the result
    np.testing.assert_array_equal(
        trimmed["epoch_livetime"].values, expected_livetime["epoch_livetime"].values
    )
    np.testing.assert_array_equal(
        trimmed["livetime_counter"].values, expected_livetime.livetime_counter.values
    )

    # Test case 2: epoch is empty, the function should raise a ValueError
    dataset = xr.Dataset(
        {
            "livetime_counter": ("epoch_livetime", np.array([10, 20, 30])),
        },
        coords={
            "epoch": np.array([]),
            "epoch_livetime": np.array([0, 1, 2]),
        },
    )
    with pytest.raises(
        ValueError,
        match="Epoch values are empty. Cannot proceed with livetime subsetting.",
    ):
        subset_livetime(dataset)

    # Test case 3: Not enough livetime values, the function should raise a ValueError
    dataset = xr.Dataset(
        {
            "livetime_counter": ("epoch_livetime", np.array([10, 20, 30, 40, 50])),
        },
        coords={
            "epoch": np.array([1, 2, 3]),
            "epoch_livetime": np.array([0, 1, 2, 3, 4]),
        },
    )
    with pytest.raises(
        ValueError,
        match="Start index for livetime is less than 10. This indicates that the "
        "dataset is too small to shift livetime correctly.",
    ):
        subset_livetime(dataset)


def test_find_complete_mod10_sets():
    """Test the find_complete_mod10_sets function."""

    # Test case 1: Valid pattern exists throughout
    mod_vals = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    expected_indices = np.array([0, 10])
    result = find_complete_mod10_sets(mod_vals)
    assert np.array_equal(result, expected_indices), (
        f"Expected {expected_indices}, got {result}"
    )

    # Test case 2: No valid pattern
    mod_vals = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6])
    expected_indices = np.array([])
    result = find_complete_mod10_sets(mod_vals)
    assert np.array_equal(result, expected_indices), (
        f"Expected {expected_indices}, got {result}"
    )

    # Test case 3: Pattern in the middle
    mod_vals = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    expected_indices = np.array([5])
    result = find_complete_mod10_sets(mod_vals)
    assert np.array_equal(result, expected_indices), (
        f"Expected {expected_indices}, got {result}"
    )

    # Test case 4: Empty input
    mod_vals = np.array([])
    expected_indices = np.array([])
    result = find_complete_mod10_sets(mod_vals)
    assert np.array_equal(result, expected_indices), (
        f"Expected {expected_indices}, got {result}"
    )


def test_calculate_uncertainties():
    """Test the calculate_uncertainties function.

    This function calculates the uncertainties for the counts data.
    """

    # Create a sample dataset
    data = {
        "counts": (("epoch", "index"), np.array([[10, 20], [0, 1]])),
        "version": (("epoch",), np.array([1, 1])),
    }
    dataset = xr.Dataset(data)

    # Calculate uncertainties
    result = calculate_uncertainties(dataset)

    # Expected uncertainties
    #   DELTA_PLUS = sqrt(counts + 1) + 1
    #   DELTA_MINUS = sqrt(counts)
    expected_delta_plus = np.array(
        [[np.sqrt(11) + 1, np.sqrt(21) + 1], [np.sqrt(1) + 1, np.sqrt(2) + 1]]
    )
    expected_delta_minus = np.array(
        [[np.sqrt(10), np.sqrt(20)], [np.sqrt(0), np.sqrt(1)]]
    )

    # Assertions
    np.testing.assert_array_almost_equal(
        result["counts_stat_uncert_plus"].values, expected_delta_plus
    )
    np.testing.assert_array_almost_equal(
        result["counts_stat_uncert_minus"].values, expected_delta_minus
    )
    assert "version_stat_uncert_plus" not in result
    assert "version_stat_uncert_minus" not in result


def test_add_cdf_attributes():
    """Test the add_cdf_attributes function."""
    # Create a dataset with multiple variable name patterns
    dataset = xr.Dataset(
        {
            "var": (["dim1", "dim2"], np.ones((2, 2))),
            "uncert_var": (["dim1", "dim2"], np.ones((2, 2))),
            "energy_var": (["dim1"], np.ones(2)),
        },
        coords={"dim1": [10, 20], "dim2": [1, 2]},
    )

    # Logical source to test macropixel logic
    logical_source = "test_logical_source"

    # Create a mock attribute manager
    attr_mgr = Mock()
    attr_mgr.get_global_attributes.return_value = {"Global_attr": "Test Dataset"}

    def fake_get_variable_attributes(name, check_schema=True):
        return {f"{name}_attr": "value", "check_schema": check_schema}

    attr_mgr.get_variable_attributes.side_effect = fake_get_variable_attributes

    # Run the function
    result = add_cdf_attributes(dataset, logical_source, attr_mgr)

    # 1. Global attributes
    assert result.attrs["Global_attr"] == "Test Dataset"

    # 2. Variable attributes
    assert "var_attr" in result["var"].attrs
    assert "uncert_var_attr" in result["uncert_var"].attrs
    assert "energy_var_attr" in result["energy_var"].attrs

    # 3. Dimension attributes and labels
    for dim in ["dim1", "dim2"]:
        assert f"{dim}_attr" in result[dim].attrs
        assert f"{dim}_label" in result.coords
        assert f"{f'{dim}_label'}_attr" in result[f"{dim}_label"].attrs
        assert list(result[f"{dim}_label"].dims) == [f"{dim}"]


def test_filter_dataset_to_processing_day():
    """Test the filter_dataset_to_processing_day function."""

    # Create a mock dataset
    epoch_values = np.array(
        [
            316008024684000000,  # 2010-01-05T23:59:18.500
            316008084684000000,  # 2010-01-06T00:10:18.500
            316094424684000000,  # 2010-01-06T23:59:18.500
            316094484684000000,  # 2010-01-07T00:10:18.500
        ]
    )

    sc_tick_values = np.array(
        [
            431999,  # 2010-01-05T23:59:59.000000000
            432002,  # 2010-01-06T00:00:02.000000000
            518399,  # 2010-01-06T23:59:59.000000000
            518402,  # 2010-01-07T00:00:02.000000000
        ]
    )

    dataset = xr.Dataset(
        {
            "var1": (["epoch"], np.arange(len(epoch_values))),
            "var2": (["sc_tick"], np.arange(len(sc_tick_values)) * 2),
        },
        coords={
            "epoch": epoch_values,
            "sc_tick": sc_tick_values,
        },
    )

    # Define the packet date
    packet_date = "20100106"

    # Call the function
    filtered_dataset = filter_dataset_to_processing_day(
        dataset, packet_date, epoch_vals=epoch_values, sc_tick=True
    )

    # Assert the filtered dataset contains only data within the processing day
    expected_epochs = np.array(
        [
            316008084684000000,  # 2010-01-06T00:10:18.500
            316094424684000000,  # 2010-01-06T23:59:18.500
        ]
    )
    assert np.array_equal(filtered_dataset["epoch"].values, expected_epochs)

    # Assert the sc_tick values are filtered correctly
    expected_sc_ticks = np.array([432002, 518399])
    assert np.array_equal(filtered_dataset["sc_tick"].values, expected_sc_ticks)


def test_subset_sectored_counts():
    """Test the subset_sectored_counts function."""

    def create_l1a_counts_dataset(hdr_minute_cnt_values):
        """Helper to create L1A counts dataset."""
        return xr.Dataset(
            {
                "hdr_minute_cnt": ("epoch", hdr_minute_cnt_values),
                "h_sectored_counts": ("epoch", np.arange(len(hdr_minute_cnt_values))),
                "he4_sectored_counts": ("epoch", np.arange(len(hdr_minute_cnt_values))),
                "livetime_counter": ("epoch", np.arange(len(hdr_minute_cnt_values))),
            },
            coords={
                "epoch": np.arange(
                    316008084684000000, 316008084684000000 + len(hdr_minute_cnt_values)
                ),
            },
        )

    def validate_subset(l1a_counts_dataset):
        """Helper to validate the subset results."""
        # Define the packet date
        packet_date = "20100106"
        subset_dataset = subset_sectored_counts(l1a_counts_dataset, packet_date)
        assert subset_dataset.sizes["epoch"] == 10
        assert subset_dataset.sizes["epoch_livetime"] == 10
        assert np.all(subset_dataset["hdr_minute_cnt"].values % 10 == np.arange(10))
        assert np.all(
            subset_dataset["epoch_livetime"].values
            == subset_dataset["epoch"].values - 10
        ), "epoch_livetime values are not shifted by 10 from epoch values"

    # Test with partial data at the start and end of the dataset
    l1a_counts_dataset = create_l1a_counts_dataset(np.arange(105, 135))
    validate_subset(l1a_counts_dataset)

    # Test with partial data in the middle of the dataset
    l1a_counts_dataset = create_l1a_counts_dataset(
        [
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            120,
            121,
            122,
            123,
            124,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
        ]
    )
    validate_subset(l1a_counts_dataset)

    # Test with partial data at the start, middle, and end of the dataset
    l1a_counts_dataset = create_l1a_counts_dataset(
        [
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
        ]
    )
    validate_subset(l1a_counts_dataset)

    # Test with only partial data in the dataset
    l1a_counts_dataset = create_l1a_counts_dataset(np.arange(100, 160, 2))
    with pytest.raises(
        ValueError,
        match="No data to process - valid start indices not found for "
        "complete sectored counts.",
    ):
        subset_sectored_counts(l1a_counts_dataset, packet_date="20100106")


def test_validate_l1a_housekeeping_data(hk_packet_filepath):
    """Validate the housekeeping dataset created by the L1A processing.

    Compares the processed housekeeping data with expected values from
    a validation csv file.

    Parameters
    ----------
    hk_packet_filepath : str
        File path to housekeeping ccsds file
    """
    datasets = hit_l1a(hk_packet_filepath, "20100105")
    hk_dataset = None
    for dataset in datasets:
        if dataset.attrs["Logical_source"] == "imap_hit_l1a_hk":
            hk_dataset = dataset

    # Load the validation data
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/hskp_sample_raw.csv"
    )
    validation_data = pd.read_csv(validation_file)
    validation_data.columns = validation_data.columns.str.lower()
    validation_data.columns = validation_data.columns.str.strip()

    # Get a list of leak columns in ascending order
    # (LEAK_I_00, LEAK_I_01, ..., LEAK_I_63)
    # and group values into a single column
    leak_columns = [col for col in validation_data.columns if col.startswith("leak")][
        ::-1
    ]
    validation_data["leak_i"] = validation_data[leak_columns].apply(
        lambda row: row.values, axis=1
    )
    validation_data.drop(columns=leak_columns, inplace=True)

    # Define the keys that should have dropped from the housekeeping dataset
    dropped_fields = {
        "pkt_apid",
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

    # Define the keys that should be ignored in the validation
    # like ccsds headers
    ignore_validation_fields = {
        "ccsds_version",
        "ccsds_type",
        "ccsds_sec_hdr_flag",
        "ccsds_appid",
        "ccsds_grp_flag",
        "ccsds_seq_cnt",
        "ccsds_length",
    }

    # Check that dropped variables are not in the dataset
    assert set(dropped_fields).isdisjoint(set(hk_dataset.data_vars.keys()))

    # Compare the housekeeping dataset with the expected validation data
    for field in validation_data.columns:
        if field not in ignore_validation_fields:
            assert field in hk_dataset.data_vars.keys()
            for pkt in range(validation_data.shape[0]):
                assert np.array_equal(
                    hk_dataset[field][pkt].data, validation_data[field][pkt]
                )


def test_validate_l1a_counts_data(sci_packet_filepath, validation_data):
    """Compare the output of the L1A processing to the validation data.

    This test compares the counts data products with the validation data.
    The PHA data product is not validated since it's not being decommutated.

    Since the validation data is structured differently than the processed data,
    This test prepares the validation data for comparison by calling helper
    functions to consolidate the data into arrays and rename columns to match
    the processed data.

    Additionally, since standard counts, sectored counts, and the livetime values
    related to the sectored counts all have different time ranges, validation data
    is further split into three parts:
        - Standard counts validation
        - Sectored counts validation
        - Livetime validation

    Parameters
    ----------
    sci_packet_filepath : str
        Path to ccsds file for science data
    validation_data : pd.DataFrame
        Preloaded validation data
    """
    # TODO: consider parameterization to test both the instrument
    #  file and fake data file

    # Prepare validation data for comparison with processed data
    validation_data = prepare_counts_validation_data(validation_data)

    # Copy the validation data for sectored data validation.
    # The first complete set of sectored data with sufficient livetime
    # data available is from index 17 to -4. Slice the validation data.
    sectored_validation_data = validation_data.iloc[17:-4].copy().reset_index(drop=True)

    # The corresponding livetime values for the sectored data is from index 7 to -14
    # (i.e. 10 minutes before the first complete set of sectored data).
    livetime_validation_data = (
        validation_data[["livetime_counter"]].iloc[7:-14].copy().reset_index(drop=True)
    )

    # NOTE: slicing indices are specific to the sci_sample_raw.csv validation file

    # Process the sample data into datasets to be validated
    processed_datasets = hit_l1a(sci_packet_filepath, packet_date="20100105")
    standard_counts_data = processed_datasets[0]
    sectored_counts_data = processed_datasets[1]

    # The validation data contains all science data variables for both
    # standard and sectored datasets. When comparing each dataset to the
    # validation data, a list of variables to skip in the comparison is
    # provided. This is to avoid comparing variables that are not present
    # in the dataset or are not relevant for the comparison. Variables to
    # skip in comparison also includes CCSDS headers the datasets contain
    # data per packet, but the validation data contains one value per
    # science frame (20 packets)

    skip_vars = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "energy_bin",
    ]

    skip_standard_vars = [
        "hdr_unit_num",
        "hdr_frame_version",
        "hdr_leak_conv",
        "hdr_heater_duty_cycle",
        "hdr_code_ok",
        "livetime_counter",
        "num_trig",
        "num_reject",
        "num_acc_w_pha",
        "num_acc_no_pha",
        "num_haz_trig",
        "num_haz_reject",
        "num_haz_acc_w_pha",
        "num_haz_acc_no_pha",
        "sngrates",
        "nread",
        "nhazard",
        "nadcstim",
        "nodd",
        "noddfix",
        "nmulti",
        "nmultifix",
        "nbadtraj",
        "nl2",
        "nl3",
        "nl4",
        "npen",
        "nformat",
        "naside",
        "nbside",
        "nerror",
        "nbadtags",
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
        "nbside_stat_uncert_plus",
        "nbside_stat_uncert_minus",
        "l2fgrates_stat_uncert_plus",
        "l2fgrates_stat_uncert_minus",
        "noddfix_stat_uncert_plus",
        "noddfix_stat_uncert_minus",
        "num_haz_acc_no_pha_stat_uncert_plus",
        "num_haz_acc_no_pha_stat_uncert_minus",
        "ialirtrates_stat_uncert_plus",
        "ialirtrates_stat_uncert_minus",
        "num_acc_no_pha_stat_uncert_plus",
        "num_acc_no_pha_stat_uncert_minus",
        "coinrates_stat_uncert_plus",
        "coinrates_stat_uncert_minus",
        "penbgrates_stat_uncert_plus",
        "penbgrates_stat_uncert_minus",
        "nmulti_stat_uncert_plus",
        "nmulti_stat_uncert_minus",
        "nmultifix_stat_uncert_plus",
        "nmultifix_stat_uncert_minus",
        "nodd_stat_uncert_plus",
        "nodd_stat_uncert_minus",
        "num_reject_stat_uncert_plus",
        "num_reject_stat_uncert_minus",
        "nhazard_stat_uncert_plus",
        "nhazard_stat_uncert_minus",
        "nl4_stat_uncert_plus",
        "nl4_stat_uncert_minus",
        "npen_stat_uncert_plus",
        "npen_stat_uncert_minus",
        "sngrates_stat_uncert_plus",
        "sngrates_stat_uncert_minus",
        "num_haz_reject_stat_uncert_plus",
        "num_haz_reject_stat_uncert_minus",
        "nformat_stat_uncert_plus",
        "nformat_stat_uncert_minus",
        "l4fgrates_stat_uncert_plus",
        "l4fgrates_stat_uncert_minus",
        "num_trig_stat_uncert_plus",
        "num_trig_stat_uncert_minus",
        "nl2_stat_uncert_plus",
        "nl2_stat_uncert_minus",
        "l2bgrates_stat_uncert_plus",
        "l2bgrates_stat_uncert_minus",
        "penfgrates_stat_uncert_plus",
        "penfgrates_stat_uncert_minus",
        "nl3_stat_uncert_plus",
        "nl3_stat_uncert_minus",
        "nerror_stat_uncert_plus",
        "nerror_stat_uncert_minus",
        "l3fgrates_stat_uncert_plus",
        "l3fgrates_stat_uncert_minus",
        "num_acc_w_pha_stat_uncert_plus",
        "num_acc_w_pha_stat_uncert_minus",
        "naside_stat_uncert_plus",
        "naside_stat_uncert_minus",
        "l3bgrates_stat_uncert_plus",
        "l3bgrates_stat_uncert_minus",
        "num_haz_acc_w_pha_stat_uncert_plus",
        "num_haz_acc_w_pha_stat_uncert_minus",
        "nread_stat_uncert_plus",
        "nread_stat_uncert_minus",
        "pbufrates_stat_uncert_plus",
        "pbufrates_stat_uncert_minus",
        "nbadtags_stat_uncert_plus",
        "nbadtags_stat_uncert_minus",
        "l4bgrates_stat_uncert_plus",
        "l4bgrates_stat_uncert_minus",
        "nadcstim_stat_uncert_plus",
        "nadcstim_stat_uncert_minus",
        "nbadtraj_stat_uncert_plus",
        "nbadtraj_stat_uncert_minus",
        "num_haz_trig_stat_uncert_plus",
        "num_haz_trig_stat_uncert_minus",
        "sc_tick_by_frame",
    ]

    skip_sectored_vars = [
        "sectorates",
        "sectorates_stat_uncert_plus",
        "sectorates_stat_uncert_minus",
        "h_sectored_counts",
        "h_energy_delta_minus",
        "h_energy_delta_plus",
        "he4_sectored_counts",
        "he4_energy_delta_minus",
        "he4_energy_delta_plus",
        "cno_sectored_counts",
        "cno_energy_delta_minus",
        "cno_energy_delta_plus",
        "nemgsi_sectored_counts",
        "nemgsi_energy_delta_minus",
        "nemgsi_energy_delta_plus",
        "fe_sectored_counts",
        "fe_energy_delta_minus",
        "fe_energy_delta_plus",
        "he4_sectored_counts_stat_uncert_plus",
        "he4_sectored_counts_stat_uncert_minus",
        "nemgsi_sectored_counts_stat_uncert_plus",
        "nemgsi_sectored_counts_stat_uncert_minus",
        "fe_sectored_counts_stat_uncert_plus",
        "fe_sectored_counts_stat_uncert_minus",
        "cno_sectored_counts_stat_uncert_plus",
        "cno_sectored_counts_stat_uncert_minus",
        "h_sectored_counts_stat_uncert_plus",
        "h_sectored_counts_stat_uncert_minus",
        "species",
    ]

    # Compare processed standard data to validation data skipping
    # ccsds headers and sectored data vars
    compare_data(
        expected_data=validation_data,
        actual_data=standard_counts_data,
        skip=[*skip_vars, *skip_sectored_vars],
    )

    # Compare processed sectored data to validation data skipping
    # ccsds headers and standard data vars
    compare_data(
        expected_data=sectored_validation_data,
        actual_data=sectored_counts_data,
        skip=[*skip_vars, *skip_standard_vars],
    )

    # Compare processed livetime data for sectored data to validation data
    compare_data(
        expected_data=livetime_validation_data,
        actual_data=sectored_counts_data,
        skip=[],
    )


def test_hit_l1a(hk_packet_filepath, sci_packet_filepath):
    """Create L1A datasets from packet files.

    Parameters
    ----------
    hk_packet_filepath : Path
        File path to ccsds file for housekeeping data
    sci_packet_filepath : Path
        File path to ccsds file for science data
    """
    for packet_filepath in [hk_packet_filepath, sci_packet_filepath]:
        processed_datasets = hit_l1a(packet_filepath, packet_date="20100105")
        assert isinstance(processed_datasets, list)
        assert all(isinstance(ds, xr.Dataset) for ds in processed_datasets)
        if packet_filepath == hk_packet_filepath:
            assert len(processed_datasets) == 1
            assert processed_datasets[0].attrs["Logical_source"] == "imap_hit_l1a_hk"
        else:
            assert len(processed_datasets) == 3
            assert (
                processed_datasets[0].attrs["Logical_source"]
                == "imap_hit_l1a_counts-standard"
            )
            assert (
                processed_datasets[1].attrs["Logical_source"]
                == "imap_hit_l1a_counts-sectored"
            )
            assert (
                processed_datasets[2].attrs["Logical_source"]
                == "imap_hit_l1a_direct-events"
            )

    # Assert that ValueError is raised when packet_date is None
    with pytest.raises(
        ValueError, match="Packet date is required for processing L1A data."
    ):
        hit_l1a(packet_filepath, "")
