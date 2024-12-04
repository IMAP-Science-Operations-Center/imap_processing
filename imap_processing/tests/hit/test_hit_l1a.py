import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.l1a.hit_l1a import hit_l1a

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


def test_compare_validation_data(sci_packet_filepath):
    """Compare the output of the L1A processing to the validation data.

    Parameters
    ----------
    sci_packet_filepath : str
        Path to ccsds file for science data
    """
    processed_datasets = hit_l1a(sci_packet_filepath, "001")
    validation_data = pd.read_csv(
        imap_module_directory / "tests/hit/test_data/sci_sample_raw1.csv"
    )
    # Prepare validation data for comparison with L1A processed data

    # Consolidate data from these columns into a new column as arrays and drop the
    # original columns. The key is the new column name and the value is the prefix
    # of the columns to consolidate.
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
    # Strip leading and trailing spaces from column names
    validation_data.columns = validation_data.columns.str.strip()
    for new_col, prefix in rate_columns.items():
        columns = [col for col in validation_data.columns if prefix in col]
        validation_data[new_col] = validation_data[columns].apply(
            lambda row: row.values, axis=1
        )

        if new_col == "sectorates":
            # Identify sectorates columns with three digits and five digits using regex
            sectorates_three_digits = validation_data.filter(
                regex=r"^SECTORATES_\d{3}$"
            ).columns
            sectorates_five_digits = validation_data.filter(
                regex=r"^SECTORATES_\d{3}_\d{1}$"
            ).columns

            # Consolidate data from these columns into new columns as arrays
            validation_data["sectorates"] = validation_data[
                sectorates_three_digits
            ].apply(lambda row: row.values.reshape(8, 15), axis=1)
            validation_data["sectorates_by_mod_val"] = validation_data[
                sectorates_five_digits
            ].apply(lambda row: row.values, axis=1)

            # Function to find the first index where a value exists
            def first_non_empty_index(row):
                return next((i for i, value in enumerate(row) if value != " "), None)

            # The index where the first value exists in the sectorates_by_mod_val
            # columns equals the mod 10 value that identifies the species detected
            validation_data["mod_10"] = validation_data["sectorates_by_mod_val"].apply(
                first_non_empty_index
            )
        # Drop the original rates columns
        validation_data.drop(columns=columns, inplace=True)

    # Process single rates into 2D arrays for both low and high gains
    validation_data["sngrates"] = validation_data.apply(
        lambda row: np.array([row["sngrates_hg"], row["sngrates_lg"]]), axis=1
    )
    validation_data.drop(columns=["sngrates_hg", "sngrates_lg"], inplace=True)

    # Columns to skip in the comparison because the processed data has values for every
    # packet and the validation data has one value per science frame. Also includes
    # columns that are not in the processed data.
    skip = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "pkt_sec_hdr",
        "sc_tick",
        "hdr_status_bits",
        "hdr_frame_version",
        "hdr_unit_num",
        "hdr_minute_cnt",
        "livetime",
        "num_trig",
        "num_reject",
        "num_acc_w_pha",
        "num_acc_no_pha",
        "num_haz_trig",
        "num_haz_reject",
        "num_haz_acc_w_pha",
        "num_haz_acc_no_pha",
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
        "sectorates_by_mod_val",
        "mod_10",
    ]

    # compare values
    validation_data.columns = validation_data.columns.str.lower()
    print(f"Validation data: {[i for i in list(validation_data.columns)]}")
    print(f"Dataset {list(processed_datasets[0].data_vars.keys())}")
    for field in list(validation_data.columns):
        # Check data fields
        if field not in [
            "sc_tick",
            "hdr_status_bits",
            "sectorates_by_mod_val",
            "mod_10",
        ]:
            assert field in processed_datasets[0].data_vars.keys()
        # Check that data values match per science frame
        if field not in skip:
            print(field)
            for frame in range(validation_data.shape[0]):
                print(frame)
                print(f"VALIDATION: {validation_data[field][frame]}")
                print(f"PROCESSED: {processed_datasets[0][field][frame].data}")
                assert np.array_equal(
                    validation_data[field][frame],
                    processed_datasets[0][field][frame].data,
                )

        # TODO: convert dataframe to xarray and compare with processed data?
        # TODO: consolidate data into arrays for all frames? to avoid looping
        #  through each frame?
        # TODO: assert correct species and energy range detected
        # TODO Add column for species and energy range detected?

        # TODO: add validation for hdr_status_bits once validation data has been updated
        #  to include this field broken out into its subfields
        # TODO: add validation to non rate fields. currently validation data only has
        #  one value per frame and the processed data has one value per packet. Which
        #  value should be used for the science frame from the sample?


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
