"""Tests the L1a processing for decommutated CoDICE data"""

import logging
import re

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice import constants
from imap_processing.codice.codice_l1a import process_codice_l1a
from imap_processing.tests.conftest import _download_external_data, _test_data_paths

from .conftest import TEST_L0_FILE, VALIDATION_DATA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pytestmark = pytest.mark.external_test_data

DESCRIPTORS = [
    "hi-ialirt",
    "lo-ialirt",
    "hskp",
    "lo-counters-aggregated",
    "lo-counters-singles",
    "lo-sw-priority",
    "lo-nsw-priority",
    "lo-sw-species",
    "lo-nsw-species",
    "lo-sw-angular",
    "lo-nsw-angular",
    "hi-counters-aggregated",
    "hi-counters-singles",
    "hi-omni",
    "hi-sectored",
    "hi-priority",
    "lo-pha",
    "hi-pha",
]

EXPECTED_ARRAY_SHAPES = [
    (),  # hi-ialirt  # TODO: Need to implement
    (),  # lo-ialirt  # TODO: Need to implement
    (31778,),  # hskp
    (77, 6, 128),  # lo-counters-aggregated
    (77, 24, 6, 128),  # lo-counters-singles
    (77, 12, 128),  # lo-sw-priority
    (77, 12, 128),  # lo-nsw-priority
    (77, 1, 128),  # lo-sw-species
    (77, 1, 128),  # lo-nsw-species
    (77, 5, 12, 128),  # lo-sw-angular
    (77, 19, 12, 128),  # lo-nsw-angular
    (77,),  # hi-counters-aggregated
    (77, 12),  # hi-counters-singles
    (77, 15, 4),  # hi-omni
    (77, 8, 12, 12),  # hi-sectored
    (77,),  # hi-priority
    (77, 10000),  # lo-pha
    (),  # hi-pha  # TODO: Need to implement
]

EXPECTED_NUM_VARIABLES = [
    0,  # hi-ialirt  # TODO: Need to implement
    0,  # lo-ialirt  # TODO: Need to implement
    139,  # hskp
    8 + len(constants.LO_COUNTERS_AGGREGATED_VARIABLE_NAMES),  # lo-counters-aggregated
    9,  # lo-counters-singles
    13,  # lo-sw-priority
    10,  # lo-nsw-priority
    24,  # lo-sw-species
    16,  # lo-nsw-species
    12,  # lo-sw-angular
    9,  # lo-nsw-angular
    2 + len(constants.HI_COUNTERS_AGGREGATED_VARIABLE_NAMES),  # hi-counters-aggregated
    5,  # hi-counters-singles
    10,  # hi-omni
    6,  # hi-sectored
    8,  # hi-priority
    80,  # lo-pha
    0,  # hi-pha  # TODO: Need to implement
]

# CoDICE-Hi products that have support variables to test
CODICE_HI_PRODUCTS = [
    "hi-counters-aggregated",
    "hi-counters-singles",
    "hi-priority",
    "hi-sectored",
]
# TODO: Add hi-omni here once I sort out the array shape discrepancy with the
#       validation data

# CoDICE-Lo products that have support variables to test
CODICE_LO_PRODUCTS = [
    "lo-counters-aggregated",
    "lo-counters-singles",
    "lo-sw-priority",
    "lo-nsw-priority",
    "lo-sw-species",
    "lo-nsw-species",
    "lo-sw-angular",
    "lo-nsw-angular",
]


@pytest.fixture(scope="session")
def test_l1a_data() -> xr.Dataset:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    processed_datasets : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    """
    # Make sure we have the data available here. This test collection gets
    # skipped at the module level if the mark isn't present. We can't decorate
    # a fixture, so add the needed call directly here instead.
    _download_external_data(_test_data_paths())
    processed_datasets = process_codice_l1a(file_path=TEST_L0_FILE)

    return processed_datasets


@pytest.mark.parametrize("index", range(len(EXPECTED_ARRAY_SHAPES)))
def test_l1a_data_array_shape(test_l1a_data, index):
    """Tests that the data arrays in the generated CDFs have the expected shape.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]
    expected_shape = EXPECTED_ARRAY_SHAPES[index]

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 17]:
        pytest.xfail("Data product is currently unsupported")

    # There are exceptions for some variables
    for variable in processed_dataset:
        # For variables with energy dimensions
        if variable in ["energy_table", "acquisition_time_per_step"]:
            assert processed_dataset[variable].data.shape == (128,)
        # For "support" variables with epoch dimensions
        elif variable in [
            "rgfo_half_spin",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
            "data_quality",
            "spin_period",
        ]:
            assert processed_dataset[variable].data.shape == (
                len(processed_dataset["epoch"].data),
            )
        # For some direct event variables:
        elif re.match(r"P[0-7]_(NumEvents|DataQuality)", variable):
            assert processed_dataset[variable].data.shape == (77,)
        # For nominal variables
        else:
            assert processed_dataset[variable].data.shape == expected_shape


@pytest.mark.parametrize("index", range(len(DESCRIPTORS)))
def test_l1a_logical_sources(test_l1a_data, index):
    """Tests that the Logical source of the dataset is what is expected.

    Since the logical source gets set by ``write_cdf``, this also tests that
    the dataset can be written to a file.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]
    expected_logical_source = f"imap_codice_l1a_{DESCRIPTORS[index]}"

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 17]:
        pytest.xfail("Data product is currently unsupported")

    # Write the dataset to a file to set the logical source attribute
    _ = write_cdf(processed_dataset)

    assert processed_dataset.attrs["Logical_source"] == expected_logical_source


@pytest.mark.parametrize("index", range(len(EXPECTED_NUM_VARIABLES)))
def test_l1a_num_data_variables(test_l1a_data, index):
    """Tests that the generated CDFs have the expected number of data variables.

    These data variables include counter data (e.g. hplus, heplus, etc.) as well
    as any "support" variables (e.g. data_quality, spin_period, etc.).

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    processed_dataset = test_l1a_data[index]

    # Mark currently broken/unsupported datasets as expected to fail
    # TODO: Remove these once they are supported
    if index in [0, 1, 17]:
        pytest.xfail("Data product is currently unsupported")

    assert len(processed_dataset) == EXPECTED_NUM_VARIABLES[index]


@pytest.mark.parametrize("index", range(len(VALIDATION_DATA)))
def test_l1a_validate_data_arrays(test_l1a_data: xr.Dataset, index):
    """Tests that the generated L1a CDF data array contents are valid.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    descriptor = DESCRIPTORS[index]

    if descriptor == "hskp":
        pytest.skip("Housekeeping data is validated in a separate test")

    # TODO: Currently only the following products can be validated, expand this
    #       to other data products as I can validate them.
    able_to_be_validated = [
        "hi-counters-aggregated",
        "hi-counters-singles",
        "hi-priority",
        "hi-sectored",
        "lo-counters-aggregated",
        "lo-counters-singles",
        "lo-sw-angular",
        "lo-nsw-angular",
        "lo-sw-priority",
        "lo-nsw-priority",
        "lo-sw-species",
        "lo-nsw-species",
        "lo_pha",
    ]

    if descriptor in able_to_be_validated:
        counters = getattr(
            constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
        )
        processed_dataset = test_l1a_data[index]
        validation_dataset = load_cdf(VALIDATION_DATA[index])

        for counter in counters:
            # Ensure the data arrays are equal
            np.testing.assert_equal(
                processed_dataset[counter].data, validation_dataset[counter].data
            )

    else:
        pytest.xfail(f"Still need to implement validation for {descriptor}")


def test_l1a_validate_hskp_data(test_l1a_data):
    """Tests that the L1a housekeeping data is valid"""

    # Housekeeping data is the 2nd element in the list of test products
    hskp_data = test_l1a_data[2]
    validation_hskp_filepath = VALIDATION_DATA[2]

    # Load the validation housekeeping data
    validation_hskp_data = load_cdf(validation_hskp_filepath)

    # These variables are not present in the validation dataset
    exclude_variables = [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
    ]

    for variable in hskp_data:
        if variable not in exclude_variables:
            np.testing.assert_array_equal(
                hskp_data[variable], validation_hskp_data[variable.upper()]
            )


@pytest.mark.parametrize("index", range(len(DESCRIPTORS)))
def test_l1a_validate_support_variables(test_l1a_data, index):
    """Tests that the support variables for the generated products match the
    validation data

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    # Hopefully I can remove this someday if Joey gives me validation data
    # with updated naming conventions
    variable_name_mapping = {
        "data_quality": "DataQuality",
        "nso_half_spin": "NSOHalfSpin",
        "rgfo_half_spin": "RGFOHalfSpin",
        "spin_period": "SpinPeriod",
        "st_bias_gain_mode": "STBiasGainMode",
        "sw_bias_gain_mode": "SWBiasGainMode",
    }

    descriptor = DESCRIPTORS[index]
    dataset = test_l1a_data[index]
    validation_dataset = load_cdf(VALIDATION_DATA[index])

    if descriptor in CODICE_LO_PRODUCTS:
        # Note that for the energy table and acquisition time, the validation
        # data only carries three decimal places whereas the SDC-generated CDFs
        # carry more significant figures

        # Ensure the energy table values are (nearly) equal
        np.testing.assert_almost_equal(
            dataset.energy_table.data, validation_dataset.EnergyTable.data, decimal=3
        )

        # Ensure that the acquisition times are (nearly) equal
        np.testing.assert_almost_equal(
            dataset.acquisition_time_per_step.data,
            validation_dataset.AcquisitionTimePerStep.data,
            decimal=3,
        )

        # Ensure that the support variables derived from packet data are equal
        for variable in variable_name_mapping:
            np.testing.assert_equal(
                dataset[variable].data,
                validation_dataset[variable_name_mapping[variable]].data,
            )

    elif descriptor in CODICE_HI_PRODUCTS:
        for variable in ["spin_period", "data_quality"]:
            np.testing.assert_equal(
                dataset[variable].data,
                validation_dataset[variable_name_mapping[variable]].data,
            )


def test_l1a_multiple_packets():
    """Tests that an input L0 file containing multiple APIDs can be processed."""

    processed_datasets = process_codice_l1a(file_path=TEST_L0_FILE)

    # TODO: Could add some more checks here?
    assert len(processed_datasets) == 18
