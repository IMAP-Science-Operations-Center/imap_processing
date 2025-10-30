from pathlib import Path

import pytest

from imap_processing import imap_module_directory

TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
TEST_DATA_L0_PATH = TEST_DATA_PATH / "l0_data"
TEST_L0_FILE = TEST_DATA_L0_PATH / "imap_codice_l0_raw_20241110_v001.pkts"


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
                / "imap_codice_l0_lo-sw-species_20250814_v001.pkts"
            ]
        elif descriptor == "lo-nsw-species" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-nsw-species_20250814_v001.pkts"
            ]
        elif descriptor == "lo-sw-angular" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-sw-angular_20250814_v001.pkts"
            ]
        elif descriptor == "lo-nsw-angular" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_lo-nsw-angular_20250814_v001.pkts"
            ]
        elif descriptor == "hi-sectored" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hi-sectored_20250814_v001.pkts"
            ]
        elif descriptor == "hi-omni" and data_type == "l0":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1a_input"
                / "imap_codice_l0_hi-omni_20250814_v001.pkts"
            ]
        if descriptor == "lo-nsw-species" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1b_validation"
                / "imap_codice_l1b_lo-nsw-species_20250814_v007.cdf"
            ]
        elif descriptor == "lo-sw-species" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests"
                / "codice"
                / "data"
                / "l1b_validation"
                / "imap_codice_l1b_lo-sw-species_20250814_v007.cdf"
            ]
        elif descriptor == "lo-nsw-angular" and data_type == "l1b":
            return [
                TEST_DATA_PATH
                / "l1b_validation"
                / "imap_codice_l1b_lo-nsw-angular_20250814_v007.cdf"
            ]
        elif descriptor == "lo-sw-angular" and data_type == "l1b":
            return [
                TEST_DATA_PATH
                / "l1b_validation"
                / "imap_codice_l1b_lo-sw-angular_20250814_v007.cdf"
            ]
        elif descriptor == "hi-sectored" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests/codice/data/l1b_validation"
                / "imap_codice_l1b_hi-sectored_20250814_v007.cdf"
            ]
        elif descriptor == "hi-omni" and data_type == "l1b":
            return [
                imap_module_directory
                / "tests/codice/data/l1b_validation"
                / "imap_codice_l1b_hi-omni_20250814_v007.cdf"
            ]
        elif descriptor == "l1a-sci-lut":
            return [
                TEST_DATA_PATH
                / "l1a_lut"
                / "imap_codice_l1a-sci-lut_20251007_v001.json"
            ]
        elif descriptor == "l2-hi-omni-efficiency":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-hi-omni-efficiency_20251008_v001.csv"
            ]
        elif descriptor == "l2-hi-sectored-efficiency":
            return [
                TEST_DATA_PATH
                / "l2_lut/imap_codice_l2-hi-sectored-efficiency_20251008_v001.csv"
            ]
        elif descriptor == "l2-lo-efficiency":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-lo-efficiency_20251008_v001.csv"
            ]
        elif descriptor == "l2-lo-gfactor":
            return [
                TEST_DATA_PATH / "l2_lut/imap_codice_l2-lo-gfactor_20251008_v001.csv"
            ]
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    return _side_effect
