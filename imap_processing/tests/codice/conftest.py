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

    def _side_effect(descriptor):
        if descriptor == "l2-hi-omni-efficiency":
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
