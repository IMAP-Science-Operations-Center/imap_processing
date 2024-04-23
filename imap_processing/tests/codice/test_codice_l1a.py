"""Tests the L1a processing for decommutated CoDICE data"""

from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1a import CoDICEL1aPipeline, process_codice_l1a

TEST_DATA = [
    (
        Path(
            f"{imap_module_directory}/tests/codice/data/raw_ccsds_20230822_122700Z_idle.bin"
        ),
        "imap_codice_l1a_hskp_20100101_v001.cdf",
    ),
    (
        Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_5_ccsds.bin"),
        "imap_codice_l1a_lo-sw-species-counts_20240319_v001.cdf",
    ),
    (
        Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_6_ccsds.bin"),
        "imap_codice_l1a_lo-nsw-species-counts_20240319_v001.cdf",
    ),
]


@pytest.fixture(scope="session")
def pipeline() -> CoDICEL1aPipeline:
    """Return an instance of the pipeline to test with

    Returns
    -------
    pipeline_instance : CoDICEL1aPipeline
        An instance of the pipeline to run tests with
    """

    pipeline_instance = CoDICEL1aPipeline(1, 1, 1, 1)

    return pipeline_instance


def test_codicel1apipeline(pipeline):
    """

    Parameters
    ----------
    pipeline

    Returns
    -------

    """

    packets = decom_packets(
        Path(f"{imap_module_directory}/tests/codice/data/lo_fsw_view_5_ccsds.bin")
    )

    pipeline.get_esa_sweep_values()
    pipeline.get_acquisition_times()
    pipeline.get_lo_data_products()
    pipeline.unpack_science_data(packets)
    dataset = pipeline.create_science_dataset(packets)

    assert hasattr(pipeline, "esa_sweep_values")
    assert hasattr(pipeline, "acquisition_times")
    assert hasattr(pipeline, "data")

    print("\n\n\n\n\n")
    print(dataset)


@pytest.mark.parametrize(("test_file", "expected_filename"), TEST_DATA)
def test_process_codice_l1a(test_file: Path, expected_filename: str):
    """Tests the ``process_codice_l1a`` function and ensure that a proper CDF
    files are created.

    Parameters
    ----------
    test_file : Path
        The file containing test data
    expected_filename : str
        The filename of the generated CDF file
    """

    packets = decom_packets(test_file)
    cdf_filename = process_codice_l1a(packets)
    assert cdf_filename.name == expected_filename
