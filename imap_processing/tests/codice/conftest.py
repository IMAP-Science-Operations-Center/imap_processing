from pathlib import Path

import pytest

from imap_processing import imap_module_directory

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
TEST_DATA_L0_PATH = TEST_DATA_PATH / "l0_data"
TEST_L0_FILE = TEST_DATA_L0_PATH / "imap_codice_l0_raw_20241110_v001.pkts"

VALIDATION_FILE_DATE = "20250814"
VALIDATION_FILE_VERSION = "v015"


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
        elif descriptor == "lo-nsw-species" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-nsw-species_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "lo-sw-angular" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-sw-angular_{VALIDATION_FILE_DATE}_v001.pkts"
            ]
        elif descriptor == "lo-nsw-angular" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / f"imap_codice_l0_lo-nsw-angular_{VALIDATION_FILE_DATE}_v001.pkts"
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
                / "imap_codice_l0_hi-counters-singles_20250814_v001.pkts"
            ]
        elif descriptor == "hi-counters-aggregated" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hi-counters-aggregated_20250814_v001.pkts"
            ]
        elif descriptor == "lo-counters-singles" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-counters-singles_20250814_v001.pkts"
            ]
        elif descriptor == "lo-counters-aggregated" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-counters-aggregated_20250814_v001.pkts"
            ]
        elif descriptor == "hskp" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_hskp_20250814_v001.pkts"
            ]
        if descriptor == "lo-nsw-species" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1b_validation"
                / (
                    f"imap_codice_l1b_lo-nsw-species_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
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
        elif descriptor == "lo-nsw-angular" and data_type == "l1b":
            return [
                TEST_DATA_PATH
                / "l1b_validation"
                / (
                    f"imap_codice_l1b_lo-nsw-angular_{VALIDATION_FILE_DATE}"
                    f"_{VALIDATION_FILE_VERSION}.cdf"
                )
            ]
        elif descriptor == "lo-sw-angular" and data_type == "l1b":
            return [
                TEST_DATA_PATH
                / "l1b_validation"
                / (
                    f"imap_codice_l1b_lo-sw-angular_{VALIDATION_FILE_DATE}"
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
                / "imap_codice_l1a-sci-lut_20251007_v004.json"
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
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-lo-efficiency_20251212_v003.csv"
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
                / "l2_lut/imap_codice_l2-hi-energy-table_20250101_v001.csv"
            ]
        elif descriptor == "l2-hi-tof-table":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-hi-tof-table_20250101_v001.csv"
            ]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    return _side_effect
