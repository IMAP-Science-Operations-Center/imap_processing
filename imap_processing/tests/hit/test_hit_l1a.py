import re

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
    calculate_uncertainties,
    decom_hit,
    hit_l1a,
    subcom_sectorates,
)

# TODO: Packet files are per apid at the moment so the tests currently
#  reflect this. Eventually, HIT will provide a packet file with all apids
#  and the tests will need to be updated.


@pytest.fixture(scope="module")
def hk_packet_filepath():
    """Set path to test data file"""
    return (
        imap_module_directory / "tests/hit/test_data/imap_hit_l0_raw_20100105_v001.pkts"
    )


@pytest.fixture(scope="module")
def sci_packet_filepath():
    """Set path to test data file"""
    return imap_module_directory / "tests/hit/test_data/sci_sample.ccsds"


@pytest.fixture(scope="module")
def validation_data():
    """Load validation data from CSV file."""
    validation_file = (
        imap_module_directory / "tests/hit/validation_data/sci_sample_raw.csv"
    )
    validation_data = pd.read_csv(validation_file)
    return validation_data


# <=== HELPER FUNCTIONS FOR VALIDATION ===>


def prepare_validation_data(validation_data):
    """Prepare validation data for comparison with processed data.

    The validation data is organized by columns with each value
    in a separate column. This function consolidates related data
    into arrays to match the processed data. It also renames columns
    to match the processed data.

    Parameters
    ----------
    validation_data : pd.DataFrame
        Validation data extracted from a csv file

    Returns
    -------
    pd.DataFrame
        Validation data formatted for comparison with processed data
    """
    rate_columns = {
        "coinrates": "COINRATES_",
        "pbufrates": "BUFRATES_",
        "l2fgrates": "L2FGRATES_",
        "l2bgrates": "L2BGRATES_",
        "l3fgrates": "L3FGRATES_",
        "l3bgrates": "L3BGRATES_",
        "penfgrates": "PENFGRATES_",
        "penbgrates": "PENBGRATES_",
        "sectorates": "SECTORATES_",
        "l4fgrates": "L4FGRATES_",
        "l4bgrates": "L4BGRATES_",
        "ialirtrates": "IALIRTRATES_",
        "sngrates_hg": "SNGRATES_HG_",
        "sngrates_lg": "SNGRATES_LG_",
    }

    rename_columns = {
        "CCSDS_VERSION": "version",
        "CCSDS_TYPE": "type",
        "CCSDS_SEC_HDR_FLAG": "sec_hdr_flg",
        "CCSDS_APPID": "pkt_apid",
        "CCSDS_GRP_FLAG": "seq_flgs",
        "CCSDS_SEQ_CNT": "src_seq_ctr",
        "CCSDS_LENGTH": "pkt_len",
        "CODE_OK": "hdr_code_ok",
        "HEATER_DUTY_CYCLE": "hdr_heater_duty_cycle",
        "LEAK_CONV": "hdr_leak_conv",
        "DY_TH_STATE": "hdr_dynamic_threshold_state",
        "LIVE_TIME": "livetime_counter",
    }

    validation_data.columns = validation_data.columns.str.strip()
    validation_data.rename(columns=rename_columns, inplace=True)
    validation_data = consolidate_rate_columns(validation_data, rate_columns)
    validation_data = process_single_rates(validation_data)
    validation_data = add_species_energy(validation_data)
    validation_data.columns = validation_data.columns.str.lower()
    return validation_data


def consolidate_rate_columns(data, rate_columns):
    """Consolidate related data into arrays to match processed data.

    The validation data has each value in a separate column. This
    function aggregates related data into arrays to match processed
    data. Each rate column has a corresponding delta plus and delta
    minus column for uncertainty values.

    Sector rates are more complex. This function distinguishes
    between sectorate columns with three digits and those with four
    digits in their names.

    Columns with three digits (e.g., SECTORATE_000) contain sectorate
    values for the science frame, with 120 such columns in the validation
    data. These will be organized into an array named "sectorates".

    Columns with four digits (e.g., SECTORATES_000_0) include the sectorate
    values with a mod 10 value appended (e.g., 0). The mod 10 value determines
    the species and energy range the sector rates represent in the science frame.
    There are 10 possible species and energy ranges, but only one with data per
    science frame. The validation data has 10 columns per 120 sector rates,
    totaling 1200 columns per science frame. Each set of 10 columns will have
    only one value, resulting in an array that looks like this:

    [nan, nan, nan, 0, nan, nan, nan, nan, nan, nan...nan, nan, nan, 0, nan...]

    These will be consolidated into a "sectorates_by_mod_val" column in the
    validation data.

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    rate_columns : dict
        Dictionary of rate columns and their prefixes in the
        validation data

    Returns
    -------
    pd.DataFrame
        Validation data with rate columns consolidated into arrays
    """

    for new_col, prefix in rate_columns.items():
        # Aggregate columns using regex patterns
        pattern_rates = re.compile(rf"^{prefix}\d+$")
        pattern_delta_plus = re.compile(rf"^{prefix}\d+_DELTA_PLUS$")
        pattern_delta_minus = re.compile(rf"^{prefix}\d+_DELTA_MINUS$")
        data[f"{new_col}"] = data.filter(regex=pattern_rates).apply(
            lambda row: row.values, axis=1
        )
        data[f"{new_col}_delta_plus"] = data.filter(regex=pattern_delta_plus).apply(
            lambda row: row.values, axis=1
        )
        data[f"{new_col}_delta_minus"] = data.filter(regex=pattern_delta_minus).apply(
            lambda row: row.values, axis=1
        )
        if new_col == "sectorates":
            # Get columns that match the pattern for sectorates with three digits
            sectorates_three_digits = data.filter(regex=r"^SECTORATES_\d{3}$").columns

            sectorates_delta_plus_three_digits = data.filter(
                regex=r"^SECTORATES_\d{3}_DELTA_PLUS$"
            ).columns

            sectorates_delta_minus_three_digits = data.filter(
                regex=r"^SECTORATES_\d{3}_DELTA_MINUS$"
            ).columns

            # Add the sectorates data as 2D arrays to the data frame
            data["sectorates"] = data[sectorates_three_digits].apply(
                lambda row: row.values.reshape(8, 15), axis=1
            )
            data["sectorates_delta_plus"] = data[
                sectorates_delta_plus_three_digits
            ].apply(lambda row: row.values.reshape(8, 15), axis=1)
            data["sectorates_delta_minus"] = data[
                sectorates_delta_minus_three_digits
            ].apply(lambda row: row.values.reshape(8, 15), axis=1)

            # Consolidate the fields that include the mod value in the
            # column name. The mod value will be extracted later to
            # determine the species and energy range the sector rates
            # data correspond to.
            sectorates_four_digits = data.filter(
                regex=r"^SECTORATES_\d{3}_\d{1}$"
            ).columns

            data["sectorates_by_mod_val"] = data[sectorates_four_digits].apply(
                lambda row: row.values, axis=1
            )
            data.drop(
                columns=data.filter(regex=r"^SECTORATES_\d{3}_\d{1}.*$").columns,
                inplace=True,
            )
        # Drop the original columns
        data.drop(columns=data.filter(regex=pattern_rates).columns, inplace=True)
        data.drop(columns=data.filter(regex=pattern_delta_plus).columns, inplace=True)
        data.drop(columns=data.filter(regex=pattern_delta_minus).columns, inplace=True)

    return data


def process_single_rates(data):
    """Combine the high and low gain single rates into 2D arrays

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    Returns
    -------
    pd.DataFrame
        Validation data with single rates combined into 2D arrays
    """

    data["sngrates"] = data.apply(
        lambda row: np.array([row["sngrates_hg"], row["sngrates_lg"]]), axis=1
    )
    data["sngrates_delta_plus"] = data.apply(
        lambda row: np.array(
            [row["sngrates_hg_delta_plus"], row["sngrates_lg_delta_plus"]]
        ),
        axis=1,
    )
    data["sngrates_delta_minus"] = data.apply(
        lambda row: np.array(
            [row["sngrates_hg_delta_minus"], row["sngrates_lg_delta_minus"]]
        ),
        axis=1,
    )
    data.drop(
        columns=[
            "sngrates_hg",
            "sngrates_lg",
            "sngrates_hg_delta_plus",
            "sngrates_lg_delta_plus",
            "sngrates_hg_delta_minus",
            "sngrates_lg_delta_minus",
        ],
        inplace=True,
    )
    return data


def add_species_energy(data):
    """Add species and energy index to the validation data.

    The sectorates data is organized by species and energy index
    in the processed data so this function adds this information
    to each row (i.e. science frame) in the validation data.

    Parameters
    ----------
    data : pd.DataFrame
        Validation data

    Returns
    -------
    pd.DataFrame
        Validation data with species and energy index added
    """

    # Find the mod value for each science frame which equals the
    # first index in the sectorates_by_mod_val array that has a value
    # instead of a nan or empty string.
    data["mod_10"] = data["sectorates_by_mod_val"].apply(
        lambda row: next(
            (i for i, value in enumerate(row) if pd.notna(value) and value != " "),
            None,
        )
    )

    mod_value_to_species_energy_map = {
        0: {"species": "H", "energy_idx": 0},
        1: {"species": "H", "energy_idx": 1},
        2: {"species": "H", "energy_idx": 2},
        3: {"species": "He4", "energy_idx": 0},
        4: {"species": "He4", "energy_idx": 1},
        5: {"species": "CNO", "energy_idx": 0},
        6: {"species": "CNO", "energy_idx": 1},
        7: {"species": "NeMgSi", "energy_idx": 0},
        8: {"species": "NeMgSi", "energy_idx": 1},
        9: {"species": "Fe", "energy_idx": 0},
    }
    # Use the mod 10 value to determine the species and energy index
    # for each science frame and add this information to the data frame
    data["species"] = data["mod_10"].apply(
        lambda row: mod_value_to_species_energy_map[row]["species"].lower()
        if row is not None
        else None
    )
    data["energy_idx"] = data["mod_10"].apply(
        lambda row: mod_value_to_species_energy_map[row]["energy_idx"]
        if row is not None
        else None
    )

    data.drop(columns=["sectorates_by_mod_val", "mod_10"], inplace=True)
    return data


def compare_data(expected_data, actual_data, skip):
    """Compare the processed data to the validation data.

    Parameters
    ----------
    expected_data : pd.DataFrame
        Validation data extracted from a csv file
        and reformatted for comparison
    actual_data : xr.Dataset
        Processed counts data from l1a processing
    skip : list
        Fields to skip in comparison
    """
    for field in expected_data.columns:
        if field not in [
            "sc_tick",
            "species",
            "energy_idx",
        ]:
            assert (
                field in actual_data.data_vars.keys()
            ), f"Field {field} not found in actual data variables"
        if field not in skip:
            for frame in range(expected_data.shape[0]):
                if field == "species":
                    # Compare sector rates data.
                    # Use species and energy index for this comparison
                    species = expected_data[field][frame]
                    energy_idx = expected_data["energy_idx"][frame]

                    # Sector rate uncertainties (float values)
                    if "sectorates_delta_plus" in expected_data.columns:
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored_delta_plus"][frame][
                                energy_idx
                            ].data,
                            expected_data["sectorates_delta_plus"][frame],
                            rtol=1e-7,  # relative tolerance
                            atol=1e-8,  # absolute tolerance
                            err_msg=f"Mismatch in {species}_counts_sectored_delta_"
                            f"plus at frame {frame}, energy_idx {energy_idx}",
                        )

                    if "sectorates_delta_minus" in expected_data.columns:
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored_delta_minus"][
                                frame
                            ][energy_idx].data,
                            expected_data["sectorates_delta_minus"][frame],
                            rtol=1e-7,
                            atol=1e-8,
                            err_msg=f"Mismatch in {species}_counts_sectored_delta_"
                            f"minus at frame {frame}, energy_idx {energy_idx}",
                        )
                    else:
                        # Sector rate data (integer values)
                        np.testing.assert_allclose(
                            actual_data[f"{species}_counts_sectored"][frame][
                                energy_idx
                            ].data,
                            expected_data["sectorates"][frame],
                            rtol=1e-7,
                            atol=1e-8,
                            err_msg=f"Mismatch in {species}_counts_sectored at"
                            f"frame {frame}, energy_idx {energy_idx}",
                        )

                else:
                    # Compare other fields
                    np.testing.assert_allclose(
                        actual_data[field][frame].data,
                        expected_data[field][frame],
                        rtol=1e-7,
                        atol=1e-8,
                        err_msg=f"Mismatch in {field} at frame {frame}",
                    )


# <=== TESTS ===>


def test_validate_l1a_housekeeping_data(hk_packet_filepath):
    """Validate the housekeeping dataset created by the L1A processing.

    Compares the processed housekeeping data with expected values from
    a validation csv file.

    Parameters
    ----------
    hk_packet_filepath : str
        File path to housekeeping ccsds file
    """
    datasets = hit_l1a(hk_packet_filepath, "001")
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
        "shcoarse",
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


def test_subcom_sectorates(sci_packet_filepath):
    """Test the subcom_sectorates function.

    This function organizes the sector rates data
    by species and adds the data as new variables
    to the dataset.
    """

    # Unpack and decompress ccsds file to xarray datasets
    sci_dataset = get_datasets_by_apid(sci_packet_filepath)[HitAPID.HIT_SCIENCE]
    sci_dataset = decom_hit(sci_dataset)

    # Call the function to be tested
    subcom_sectorates(sci_dataset)

    # Number of science frames in the dataset
    frames = sci_dataset["epoch"].shape[0]

    # Check if the dataset has the expected new variables
    for species in ["h", "he4", "cno", "nemgsi", "fe"]:
        assert f"{species}_counts_sectored" in sci_dataset
        assert f"{species}_energy_min" in sci_dataset
        assert f"{species}_energy_max" in sci_dataset

        # Check the shape of the new data variables
        if species == "h":
            assert sci_dataset[f"{species}_counts_sectored"].shape == (frames, 3, 8, 15)
            assert sci_dataset[f"{species}_energy_min"].shape == (3,)
        elif species in ("4he", "cno", "nemgsi"):
            assert sci_dataset[f"{species}_counts_sectored"].shape == (frames, 2, 8, 15)
            assert sci_dataset[f"{species}_energy_min"].shape == (2,)
        elif species == "fe":
            assert sci_dataset[f"{species}_counts_sectored"].shape == (frames, 1, 8, 15)
            assert sci_dataset[f"{species}_energy_min"].shape == (1,)
        assert (
            sci_dataset[f"{species}_energy_max"].shape
            == sci_dataset[f"{species}_energy_min"].shape
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
        result["counts_delta_plus"].values, expected_delta_plus
    )
    np.testing.assert_array_almost_equal(
        result["counts_delta_minus"].values, expected_delta_minus
    )
    assert "version_delta_plus" not in result
    assert "version_delta_minus" not in result


def test_validate_l1a_counts_data(sci_packet_filepath, validation_data):
    """Compare the output of the L1A processing to the validation data.

    This test compares the counts data product with the validation data.
    The PHA data product is not validated since it's not being decommutated.

    Since the validation data is structured differently than the processed data,
    This test prepares the validation data for comparison by calling helper
    functions to consolidate the data into arrays and rename columns to match
    the processed data.

    Parameters
    ----------
    sci_packet_filepath : str
        Path to ccsds file for science data
    validation_data : pd.DataFrame
        Preloaded validation data
    """

    # Process the sample data
    processed_datasets = hit_l1a(sci_packet_filepath, "001")
    l1a_counts_data = processed_datasets[0]

    # Prepare validation data for comparison with processed data
    validation_data = prepare_validation_data(validation_data)

    # Fields to skip in comparison. CCSDS headers plus a few others.
    # The CCSDS header fields contain data per packet in the dataset, but the
    # validation data has one value per science frame.
    skip_fields = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "sc_tick",
        "energy_idx",
    ]

    # Compare processed data to validation data
    compare_data(
        expected_data=validation_data, actual_data=l1a_counts_data, skip=skip_fields
    )

    # TODO: add validation for SC_TICK field. currently validation data only has
    #  one value per frame (from first packet in the frame) and the processed data
    #  has one value per packet.


def test_hit_l1a(hk_packet_filepath, sci_packet_filepath):
    """Create L1A datasets from packet files.

    Parameters
    ----------
    hk_packet_filepath : str
        Path to ccsds file for housekeeping data
    sci_packet_filepath : str
        Path to ccsds file for science data
    """
    for packet_filepath in [hk_packet_filepath, sci_packet_filepath]:
        processed_datasets = hit_l1a(packet_filepath, "001")
        assert isinstance(processed_datasets, list)
        assert all(isinstance(ds, xr.Dataset) for ds in processed_datasets)
        if packet_filepath == hk_packet_filepath:
            assert len(processed_datasets) == 1
            assert processed_datasets[0].attrs["Logical_source"] == "imap_hit_l1a_hk"
        else:
            assert len(processed_datasets) == 2
            assert (
                processed_datasets[0].attrs["Logical_source"]
                == "imap_hit_l1a_count-rates"
            )
            assert (
                processed_datasets[1].attrs["Logical_source"]
                == "imap_hit_l1a_pulse-height-events"
            )
