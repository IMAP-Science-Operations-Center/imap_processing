import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import (
    HitAPID,
    get_datasets_by_apid,
)
from imap_processing.hit.l1a.hit_l1a import decom_hit, hit_l1a, subcom_sectorates

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
    return imap_module_directory / "tests/hit/test_data/sci_sample1.ccsds"


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


def test_validate_l1a_counts_data(sci_packet_filepath):
    """Compare the output of the L1A processing to the validation data.

    This test compares the counts data product with the validation data.
    The PHA data product is not validated since it's not being decommutated.

    Parameters
    ----------
    sci_packet_filepath : str
        Path to ccsds file for science data
    """
    # Process the sample data
    processed_datasets = hit_l1a(sci_packet_filepath, "001")
    l1a_counts_data = processed_datasets[0]

    # Read in the validation data
    validation_data = pd.read_csv(
        imap_module_directory / "tests/hit/validation_data/sci_sample_raw1.csv"
    )

    # Helper functions for this test
    def consolidate_rate_columns(data, rate_columns):
        # The validation data isn't organized by arrays.
        # Each value is in a separate column.
        # Aggregate related data into arrays.
        for new_col, prefix in rate_columns.items():
            columns = [col for col in data.columns if prefix in col]
            data[new_col] = data[columns].apply(lambda row: row.values, axis=1)
            if new_col == "sectorates":
                # Differentiate between the sectorate columns with three and
                # five digits in the name. Those with three digits contain the
                # sectorate value for the science frame and those with five digits
                # are the sectorate values with the mod value appended to the end.
                # The mod value determines the species and energy range for that
                # science frame
                sectorates_three_digits = data.filter(
                    regex=r"^SECTORATES_\d{3}$"
                ).columns
                sectorates_five_digits = data.filter(
                    regex=r"^SECTORATES_\d{3}_\d{1}$"
                ).columns
                data["sectorates"] = data[sectorates_three_digits].apply(
                    lambda row: row.values.reshape(8, 15), axis=1
                )
                data["sectorates_by_mod_val"] = data[sectorates_five_digits].apply(
                    lambda row: row.values, axis=1
                )
            data.drop(columns=columns, inplace=True)
        return data

    def process_single_rates(data):
        # Combine the single rates for high and low gain into a 2D array
        data["sngrates"] = data.apply(
            lambda row: np.array([row["sngrates_hg"], row["sngrates_lg"]]), axis=1
        )
        data.drop(columns=["sngrates_hg", "sngrates_lg"], inplace=True)
        return data

    def process_sectorates(data):
        # Add species and energy index to the data frame for each science frame
        # First find the mod value for each science frame which equals the first index
        # in the sectorates_by_mod_val array that has a value instead of a blank space
        data["mod_10"] = data["sectorates_by_mod_val"].apply(
            lambda row: next((i for i, value in enumerate(row) if value != " "), None)
        )
        # Mapping of mod value to species and energy index
        species_energy = {
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
            lambda row: species_energy[row]["species"].lower()
            if row is not None
            else None
        )
        data["energy_idx"] = data["mod_10"].apply(
            lambda row: species_energy[row]["energy_idx"] if row is not None else None
        )
        data.drop(columns=["sectorates_by_mod_val", "mod_10"], inplace=True)
        return data

    def compare_data(expected_data, actual_data, skip):
        # Compare the processed data to the validation data
        for field in expected_data.columns:
            if field not in [
                "sc_tick",
                "hdr_status_bits",
                "species",
                "energy_idx",
            ]:
                assert field in l1a_counts_data.data_vars.keys()
            if field not in ignore:
                for frame in range(expected_data.shape[0]):
                    if field == "species":
                        species = expected_data[field][frame]
                        energy_idx = expected_data["energy_idx"][frame]
                        assert np.array_equal(
                            actual_data[f"{species}_counts_sectored"][frame][
                                energy_idx
                            ].data,
                            expected_data["sectorates"][frame],
                        )
                    else:
                        assert np.array_equal(
                            actual_data[field][frame].data, expected_data[field][frame]
                        )

    rate_columns = {
        "coinrates": "COINRATES_",
        "bufrates": "BUFRATES_",
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

    # Prepare validation data for comparison with processed data
    validation_data.columns = validation_data.columns.str.strip()
    validation_data = consolidate_rate_columns(validation_data, rate_columns)
    validation_data = process_single_rates(validation_data)
    validation_data = process_sectorates(validation_data)

    # Fields to skip in comparison. CCSDS headers plus a few others that are not
    # relevant to the comparison.
    # The CCSDS header fields contain data per packet in the dataset, but the
    # validation data has a value per science frame so skipping comparison for now
    ignore = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "sc_tick",
        "hdr_status_bits",
        "energy_idx",
    ]

    # Compare processed data to validation data
    validation_data.columns = validation_data.columns.str.lower()
    compare_data(validation_data, l1a_counts_data, ignore)

    # TODO: add validation for hdr_status_bits once validation data has been updated
    #  to include this field broken out into its subfields

    # TODO: add validation for CCSDS fields? currently validation data only has
    #  one value per frame and the processed data has one value per packet.


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
