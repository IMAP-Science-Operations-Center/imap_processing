"""Tests the L1a processing for decommutated CoDICE data"""

import logging
import re

import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.codice import constants
from imap_processing.codice.codice_l1a import process_codice_l1a

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
    "lo-direct-events",
    "hi-direct-events",
]


EXPECTED_ARRAY_SHAPES = [
    (304, 15),  # hi-ialirt
    (76, 128, 1),  # lo-ialirt
    (31778,),  # hskp
    (77, 128, 6),  # lo-counters-aggregated
    (77, 128, 24, 6),  # lo-counters-singles
    (77, 128, 24),  # lo-sw-priority
    (77, 128, 24),  # lo-nsw-priority
    (77, 128, 1),  # lo-sw-species
    (77, 128, 1),  # lo-nsw-species
    (77, 128, 5, 24),  # lo-sw-angular
    (77, 128, 19, 24),  # lo-nsw-angular
    (77,),  # hi-counters-aggregated
    (77, 12),  # hi-counters-singles
    (),  # hi-omni, shapes are specific to species
    (77, 8, 12, 12),  # hi-sectored
    (77,),  # hi-priorities
    (77, 10000),  # lo-direct-events
    (77, 10000),  # hi-direct-events
]

EXPECTED_HI_OMNI_ARRAY_SHAPES = {
    "h": (308, 15),
    "he3": (308, 15),
    "he4": (308, 15),
    "c": (308, 18),
    "o": (308, 18),
    "ne_mg_si": (308, 15),
    "fe": (308, 18),
    "uh": (308, 5),
    "junk": (308, 1),
}

EXPECTED_NUM_VARIABLES = [
    3,  # hi-ialirt
    18,  # lo-ialirt
    139,  # hskp
    9 + len(constants.LO_COUNTERS_AGGREGATED_VARIABLE_NAMES),  # lo-counters-aggregated
    10,  # lo-counters-singles
    14,  # lo-sw-priority
    11,  # lo-nsw-priority
    25,  # lo-sw-species
    17,  # lo-nsw-species
    13,  # lo-sw-angular
    10,  # lo-nsw-angular
    2 + len(constants.HI_COUNTERS_AGGREGATED_VARIABLE_NAMES),  # hi-counters-aggregated
    5,  # hi-counters-singles
    11,  # hi-omni
    6,  # hi-sectored
    8,  # hi-priority
    80,  # lo-direct-events
    60,  # hi-direct-events
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
# TODO: Investigate why lo-ialirt is failing some tests
CODICE_LO_PRODUCTS = [
    "lo-counters-aggregated",
    "lo-counters-singles",
    "lo-nsw-angular",
    "lo-nsw-priority",
    "lo-nsw-species",
    "lo-sw-angular",
    "lo-sw-priority",
    "lo-sw-species",
]


@pytest.fixture(scope="session")
def test_l1a_data() -> list[xr.Dataset]:
    """Return a ``xarray`` dataset containing test data.

    Returns
    -------
    processed_datasets : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    """
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

    descriptor = DESCRIPTORS[index]
    processed_dataset = test_l1a_data[index]
    expected_shape = EXPECTED_ARRAY_SHAPES[index]

    # hi-omni data array shapes depend on the species
    if descriptor == "hi-omni":
        for variable in constants.HI_OMNI_VARIABLE_NAMES:
            assert (
                processed_dataset[variable].data.shape
                == EXPECTED_HI_OMNI_ARRAY_SHAPES[variable]
            )

    else:
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
            elif re.match(r"p[0-7]_(num_events|data_quality)", variable):
                assert processed_dataset[variable].data.shape == (77,)
            # For the k-factor
            elif variable == "k_factor":
                assert processed_dataset[variable].data.shape == (1,)
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
    assert len(processed_dataset) == EXPECTED_NUM_VARIABLES[index]


@pytest.mark.parametrize("index", range(len(VALIDATION_DATA)))
@pytest.mark.xfail(reason="Validation test turned off; awaiting fixes")
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

    # Mark currently broken/unsupported datasets as expected to fail
    if descriptor == "hskp":
        pytest.skip("Housekeeping data is validated in a separate test")
    # TODO: Remove this next condition once hi-ialirt is validated
    if descriptor == "hi-ialirt":
        pytest.xfail("Awaiting validation fixes")

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


@pytest.mark.parametrize("index", range(len(DESCRIPTORS)))
def test_l1a_validate_dimensions(test_l1a_data, index):
    """Tests that the dimensions of the data are in the expected order.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    descriptor = DESCRIPTORS[index]
    dataset = test_l1a_data[index]

    # This is the expected order of dimensions. Not all of these appear in every
    # data product, but for those that do appear, they should be in this order.
    expected_dims_order = [
        "epoch",
        "esa_step",
        "inst_az",
        "spin_sector",
        "spin_sector_pairs",
        "ssd_index",
    ]

    # We don't need to check hskp, direct events, or binned datasets since they
    # are not multidimensional
    if descriptor not in [
        "hskp",
        "lo-direct-events",
        "hi-direct-events",
        "hi-omni",
        "hi-ialirt",
        "hi-sectored",
    ]:
        # Get the variables that have dimensions that need to be checked
        counters = getattr(
            constants, f"{descriptor.upper().replace('-', '_')}_VARIABLE_NAMES"
        )

        # Ensure that, of the dimensions in the particular variable, they occur
        # in the expected order.
        for counter in counters:
            positions = [
                expected_dims_order.index(dim) for dim in dataset[counter].dims
            ]
            assert positions == sorted(positions)


@pytest.mark.parametrize("index", range(len(DESCRIPTORS)))
def test_l1a_validate_epoch_values(test_l1a_data, index):
    """Tests that the epoch values in the generated data products match the
    validation data.

    Parameters
    ----------
    test_l1a_data : list[xarray.Dataset]
        A list of ``xarray`` datasets containing the test data
    index : int
        The index of the list to test
    """

    descriptor = DESCRIPTORS[index]
    dataset = test_l1a_data[index]
    validation_dataset = load_cdf(VALIDATION_DATA[index])

    if descriptor in ["hi-ialirt", "lo-ialirt"]:
        pytest.xfail(
            f"Awaiting implementation of proper epoch calculation for {descriptor}"
        )

    # TODO: Add checks for epoch_delta_minus
    # TODO: Revisit this at some point to see if we can do an exact comparison
    np.testing.assert_allclose(
        dataset.epoch.data, validation_dataset.epoch.data, rtol=1e-6, atol=0
    )


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

    support_variables = [
        "data_quality",
        "nso_half_spin",
        "rgfo_half_spin",
        "spin_period",
        "st_bias_gain_mode",
        "sw_bias_gain_mode",
        "k_factor",
    ]

    descriptor = DESCRIPTORS[index]
    dataset = test_l1a_data[index]
    validation_dataset = load_cdf(VALIDATION_DATA[index])

    if descriptor in CODICE_LO_PRODUCTS:
        # Note that for the energy table and acquisition time, the validation
        # data only carries three decimal places whereas the SDC-generated CDFs
        # carry more significant figures

        # Ensure the energy table values are (nearly) equal
        np.testing.assert_almost_equal(
            dataset.energy_table.data, validation_dataset.voltage_table.data, decimal=3
        )

        # Ensure that the acquisition times are (nearly) equal
        # TODO: Turn this back on when Joey supplies updated validation data with
        #       updated acquisition times
        # np.testing.assert_almost_equal(
        #     dataset.acquisition_time_per_step.data,
        #     validation_dataset.acquisition_time_per_step.data,
        #     decimal=3,
        # )

        # Ensure that the support variables derived from packet data are equal
        for variable in support_variables:
            np.testing.assert_equal(
                dataset[variable].data,
                validation_dataset[variable].data,
            )

    elif descriptor in CODICE_HI_PRODUCTS:
        for variable in ["spin_period", "data_quality"]:
            np.testing.assert_equal(
                dataset[variable].data,
                validation_dataset[variable].data,
            )


def test_l1a_multiple_packets():
    """Tests that an input L0 file containing multiple APIDs can be processed."""

    processed_datasets = process_codice_l1a(file_path=TEST_L0_FILE)

    assert len(processed_datasets) == 18
