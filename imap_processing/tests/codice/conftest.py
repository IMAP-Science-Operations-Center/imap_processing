from pathlib import Path

import numpy as np
import pytest

from imap_processing import imap_module_directory

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
TEST_DATA_L0_PATH = TEST_DATA_PATH / "l0_data"
TEST_L0_FILE = TEST_DATA_L0_PATH / "imap_codice_l0_raw_20241110_v001.pkts"

VALIDATION_FILE_DATE = "20260204"
VALIDATION_FILE_VERSION = "v026"

IALIRT_VALIDATION_FILE_DATE = "20250814"
IALIRT_VALIDATION_FILE_VERSION = "v015"


def assert_allclose_fillaware(actual, expected_da, rtol=1e-5, err_msg=""):
    """Compare processed data against a validation DataArray, treating the
    validation's CDF FILLVAL sentinel as equivalent to NaN.

    Some of our in-memory processed data represents "no data" as float NaN
    (e.g. after masking science counts), but integer-typed CDF variables (e.g.
    uint32 raw counts) can't hold NaN, so the validation CDF (loaded straight
    from disk) stores the CDF FILLVAL sentinel instead. Comparing those
    directly always reports a mismatch even though both sides mean the same
    thing, so normalize the validation side to NaN wherever it equals FILLVAL
    before comparing -- but only when our own data is actually float (and so
    could contain real NaN). Variables that stay integer end-to-end on both
    sides (e.g. half_spin_per_esa_step, which uses its own FILLVAL literally
    rather than NaN) should be compared as-is.
    """
    actual = np.asarray(actual)
    expected = expected_da.values
    fillval = expected_da.attrs.get("FILLVAL")
    if (
        fillval is not None
        and np.issubdtype(expected.dtype, np.integer)
        and np.issubdtype(actual.dtype, np.floating)
    ):
        expected = expected.astype(np.float64)
        expected[expected == fillval] = np.nan
    np.testing.assert_allclose(actual, expected, rtol=rtol, err_msg=err_msg)


@pytest.fixture(scope="session")
def codice_lut_path():
    """Return a callable side-effect that returns LUT paths based on descriptor.

    This fixture is intended to be used as the `side_effect` for
    `ProcessingInputCollection.get_file_paths` in tests, e.g.:

        mock_get_file_paths.side_effect = codice_lut_path

    The returned function accepts a single argument `descriptor` and returns
    a list of Paths.
    """

    def _side_effect(descriptor: str = None, data_type: str = None) -> list[Path]:  # noqa: RUF013, PLR0911, PLR0912
        # Science data could need to be distinguished by data_type since
        # there are both L0 and L1A science files for same descriptor.
        if descriptor == "lo-sw-species" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-sw-species_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "hi-sectored" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_hi-sectored_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "hi-omni" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_hi-omni_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "lo-direct-events" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-direct-events_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "hi-direct-events" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_hi-direct-events_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "lo-nsw-priority" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-nsw-priority_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "lo-sw-priority" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-sw-priority_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "hi-priorities" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_hi-priority_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "hi-counters-singles" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hi-counters-singles_20260204_v001.pkts"
            ]
        elif descriptor == "hi-counters-aggregated" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hi-counters-aggregated_20260204_v001.pkts"
            ]
        elif descriptor == "lo-counters-singles" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-counters-singles_20260204_v001.pkts"
            ]
        elif descriptor == "lo-counters-aggregated" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-counters-aggregated_20260204_v001.pkts"
            ]
        elif descriptor == "hskp" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hskp_20260204_v001.pkts"
            ]
        elif descriptor == "lo-sw-species" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1b_validation"
                / (
                    f"imap_codice_l1b_lo-sw-species_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
            ]
        elif descriptor == "hi-sectored" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests/codice/data/l1b_validation"
                / (
                    f"imap_codice_l1b_hi-sectored_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
            ]
        elif descriptor == "hi-omni" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests/codice/data/l1b_validation"
                / (
                    f"imap_codice_l1b_hi-omni_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
            ]
        elif descriptor == "l1a-sci-lut":
            return [
                TEST_DATA_PATH
                / "l1a_lut"
                / "imap_codice_l1a-sci-lut_20260129_v002.json"
            ]
        elif descriptor == "l1a-sci-lut-oct":
            return [
                TEST_DATA_PATH
                / "l1a_lut"
                / "imap_codice_l1a-sci-lut_20251007_v005.json"
            ]
        elif descriptor == "l1a-sci-lut-jan":
            return [
                TEST_DATA_PATH
                / "l1a_lut"
                / "imap_codice_l1a-sci-lut_20260129_v002.json"
            ]
        elif descriptor == "l2-hi-omni-efficiency":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-hi-omni-efficiency_20251212_v003.csv"
            ]
        elif descriptor == "l2-hi-sectored-efficiency":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-hi-sectored-efficiency_20251212_v003.csv"
            ]
        elif descriptor == "l2-lo-efficiency":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-lo-efficiency_20251008_v003.csv"
            ]
        elif descriptor == "l2-lo-gfactor":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-lo-gfactor_20251212_v003.csv"
            ]
        elif descriptor == "l2-lo-onboard-mpq-cal":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-lo-onboard-mpq-cal_20250101_v001.csv"
            ]
        elif descriptor == "l2-lo-onboard-energy-bins":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-lo-onboard-energy-bins_20250101_v001.csv"
            ]
        elif descriptor == "l2-lo-onboard-energy-table":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-lo-onboard-energy-table_20250101_v001.csv"
            ]
        elif descriptor == "l2-hi-energy-table":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-hi-energy-table_20250101_v002.csv"
            ]
        elif descriptor == "l2-hi-tof-table":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-hi-tof-table_20250101_v001.csv"
            ]
        elif descriptor == "fsw-changes":
            return [TEST_DATA_PATH / "l1a_input/imap_codice_l0_raw_20260130_v001.pkts"]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    return _side_effect
