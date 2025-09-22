"""Global pytest configuration for the package."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import cdflib
import imap_data_access
import numpy as np
import pandas as pd
import pytest
import requests
import spiceypy

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.spice import config as spice_config
from imap_processing.spice.time import TTJ2000_EPOCH, met_to_ttj2000ns
from imap_processing.tests.external_test_data_config import EXTERNAL_TEST_DATA


@pytest.fixture(autouse=True)
def _set_global_config(monkeypatch, tmp_path):
    """Set the global data directory to a temporary directory."""
    monkeypatch.setitem(imap_data_access.config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        imap_data_access.config, "DATA_ACCESS_URL", "https://api.test.com"
    )


@pytest.fixture(scope="session")
def imap_tests_path():
    return imap_module_directory / "tests"


@pytest.fixture(autouse=True)
def clear_spin_and_repoint_paths(monkeypatch):
    """Clear the spin and repoint paths to avoid having test side effects."""
    monkeypatch.setattr(spice_config, "_spin_table_paths", [])
    monkeypatch.setattr(spice_config, "_repoint_table_path", None)


@pytest.fixture(scope="session")
def _download_kernels(spice_test_data_path):
    _download_external_kernels(spice_test_data_path)


def _download_external_kernels(spice_test_data_path):
    """This fixture downloads externally-located kernels into the tests/spice/test_data
    directory if they do not already exist there. The fixture is not intended to be
    used directly. It is automatically added to tests marked with "external_kernel"
    in the hook below."""
    logger = logging.getLogger(__name__)
    kernel_urls = [
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
    ]

    for kernel_url in kernel_urls:
        kernel_name = kernel_url.split("/")[-1]
        local_filepath = spice_test_data_path / kernel_name

        if local_filepath.exists():
            continue
        allowed_attempts = 3
        for attempt_number in range(allowed_attempts):
            try:
                with requests.get(kernel_url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(local_filepath, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info("Cached kernel file to %s", local_filepath)
                continue
            except requests.exceptions.RequestException as error:
                logger.info(f"Request failed. {error}")
                if attempt_number < allowed_attempts:
                    logger.info(
                        f"Trying again, retries left "
                        f"{allowed_attempts - attempt_number}, "
                        f"Exception: {error}"
                    )
                    time.sleep(1)
                else:
                    logger.error(
                        f"Failed to download file {kernel_name} after "
                        f"{allowed_attempts} attempts, Final Error: {error}"
                    )
                    raise


@pytest.fixture(scope="session")
def _download_test_data():
    _download_external_data()


def _download_external_data():
    """This fixture downloads externally-located test data files into a specific
    location. The list of files and their storage locations are specified in
    the `test_data_paths` parameter, which is a list of tuples; the zeroth
    element being the source of the test file in the AWS S3 bucket, and the
    first element being the location in which to store the downloaded file."""

    logger = logging.getLogger(__name__)

    api_path = "https://api.dev.imap-mission.com/download/test_data/"
    for source_filename, destination_path in EXTERNAL_TEST_DATA:
        source = api_path + source_filename
        destination = (
            Path(f"{imap_module_directory}/tests") / destination_path / source_filename
        )

        # Create parent directories if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Download the test data if necessary and write it to the appropriate
        # directory
        if not destination.exists():
            response = requests.get(source, timeout=60)
            if response.status_code == 200:
                with open(destination, "wb") as file:
                    file.write(response.content)
                logger.info(f"Downloaded file: {source}")
            else:
                logger.error(
                    f"Failed to download file: {source} "
                    f"with response: {response.status_code}"
                )
        else:
            logger.info(f"File already exists: {destination}")


def pytest_collection_modifyitems(items):
    """
    The use of this hook allows modification of test `Items` after tests have
    been collected. In this case, it automatically adds fixtures based on the
    following table:

    +---------------------+----------------------------+
    | pytest mark         | fixture added              |
    +=====================+============================+
    | external_kernel     | _download_kernels          |
    | external_test_data  | _download_test_data        |
    +---------------------+----------------------------+

    Notes
    -----
    See the following link for details about this function, also known as a
    pytest hook:
    https://docs.pytest.org/en/stable/reference/reference.html#
    pytest.hookspec.pytest_collection_modifyitems
    """
    markers_to_fixtures = {
        "external_kernel": "_download_kernels",
        "external_test_data": "_download_test_data",
    }

    for item in items:
        for marker, fixture in markers_to_fixtures.items():
            if item.get_closest_marker(marker) is not None:
                item.fixturenames.append(fixture)


@pytest.fixture(scope="session")
def spice_test_data_path(imap_tests_path):
    return imap_tests_path / "spice/test_data"


@pytest.fixture(autouse=True, scope="session")
def furnish_time_kernels(spice_test_data_path):
    """Furnishes (temporarily) the testing LSK and SCLK"""
    spiceypy.kclear()
    test_lsk = spice_test_data_path / "naif0012.tls"
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spiceypy.furnsh(str(test_lsk))
    spiceypy.furnsh(str(test_sclk))
    yield test_lsk, test_sclk
    spiceypy.kclear()


@pytest.fixture
def furnish_kernels(spice_test_data_path):
    """
    Return a function to use as a context manager to furnish a list of kernels.

    Kernel files are assumed to exist in the tests/spice/test_data directory.

    Examples
    --------
    >>> def test_spicey_function(furnish_kernels):
    >>>     kernels_to_furnish = [
    >>>         "naif0012.tls",
    >>>         "kernel_0.tm",
    >>>         "kernel_1.bsp",
    >>>     ]
    >>>     with furnish_kernels(kernels_to_furnish):
    >>>         result = spicey_function()
    """

    @contextmanager
    def furnish_kernels(kernels: list[Path]):
        with spiceypy.KernelPool(
            [str(spice_test_data_path / k) for k in kernels]
        ) as pool:
            yield pool

    return furnish_kernels


@pytest.fixture
def use_test_spin_data_csv(monkeypatch):
    """Monkeypatches `spin._spin_table_paths` to the input Path."""

    def wrapped_set_spin_data_filepath(paths: list[Path]):
        monkeypatch.setattr(spice_config, "_spin_table_paths", paths)

    return wrapped_set_spin_data_filepath


@pytest.fixture
def use_fake_spin_data_for_time(
    request,
    use_test_spin_data_csv,
    tmp_path,
    generate_spin_data,
    spin_period=15.0,
):
    """
    Generate and use fake spin data for testing.

    Returns
    -------
    callable
        Returns a callable function that takes start_met and optionally end_met
        as inputs, generates fake spin data, writes the data to a csv file,
        and sets the SPIN_DATA_FILEPATH environment variable to point to the
        fake spin data file.
    """

    def wrapped_set_spin_data_filepath(
        start_met: float,
        end_met: int | None = None,
        spin_period: float | None = 15.0,
    ) -> pd.DataFrame:
        """
        Generate and use fake spin data for testing.
        Parameters
        ----------
        start_met : int
            Provides the start time in Mission Elapsed Time (MET).
        end_met : int
            Provides the end time in MET. If not provided, default to one day
            from start time.
        spin_period : float, optional
            Provides the spin period in seconds. Default is 15.0 seconds.
        """
        spin_df = generate_spin_data(
            start_met, end_met=end_met, spin_period=spin_period
        )
        spin_csv_file_path = tmp_path / "spin_data.spin.csv"
        spin_df.to_csv(spin_csv_file_path, index=False)
        use_test_spin_data_csv([spin_csv_file_path])

    return wrapped_set_spin_data_filepath


@pytest.fixture
def generate_spin_data():
    def make_data(
        start_met: float,
        end_met: float | None = None,
        spin_period: float | None = None,
    ) -> pd.DataFrame:
        """
        Generate a spin table CSV covering one or more days.
        Spin table contains the following fields:
            (
            spin_number,
            spin_start_sec_sclk,
            spin_start_subsec_sclk,
            spin_start_utc,
            spin_period_sec,
            spin_period_valid,
            spin_phase_valid,
            spin_period_source,
            thruster_firing
            )
        This function creates spin data using start MET and end MET time.
        Each spin start data uses the nominal 15-second spin period. The spins that
        occur from 00:00(Mid-night) to 00:10 UTC are marked with flags for
        thruster firing, invalid spin period, and invalid spin phase.
        Parameters
        ----------
        start_met : float
            Provides the start time in Mission Elapsed Time (MET).
        end_met : float
            Provides the end time in MET. If not provided, default to one day
            from start time.
        spin_period : float, optional
            Provides the spin period in seconds. Default is 15.0 seconds.
        Returns
        -------
        spin_df : pd.DataFrame
            Spin data.
        """
        if end_met is None:
            # end_time is one day after start_time
            end_met = start_met + 86400

        # Create spin start second data of 15 seconds increment
        spin_start_met = np.arange(start_met, end_met + 0.001, spin_period)
        spin_start_sec = np.floor(spin_start_met).astype(int)
        spin_start_subsec = ((spin_start_met - spin_start_sec) * 1e6).astype(int)

        # Calculate UTC times without spice (accepting ~5 second inaccuracy)
        spin_start_dt64 = TTJ2000_EPOCH + (spin_start_met * 1e9).astype(
            "timedelta64[ns]"
        )

        nspins = len(spin_start_sec)

        spin_df = pd.DataFrame.from_dict(
            {
                "spin_number": np.arange(nspins, dtype=np.uint32),
                "spin_start_sec_sclk": spin_start_sec,
                "spin_start_subsec_sclk": np.full(
                    nspins, spin_start_subsec, dtype=np.uint32
                ),
                "spin_start_utc": np.datetime_as_string(spin_start_dt64, unit="us"),
                "spin_period_sec": np.full(nspins, spin_period, dtype=np.float32),
                "spin_period_valid": np.ones(nspins, dtype=np.uint8),
                "spin_phase_valid": np.ones(nspins, dtype=np.uint8),
                "spin_period_source": np.zeros(nspins, dtype=np.uint8),
                "thruster_firing": np.zeros(nspins, dtype=np.uint8),
            }
        )

        # Convert spin_start_sec to datetime to set repointing times flags
        spin_start_dates = met_to_ttj2000ns(spin_start_sec + spin_start_subsec / 1e6)
        spin_start_dates = cdflib.cdfepoch.to_datetime(spin_start_dates)

        # Convert DatetimeIndex to Series for using .dt accessor
        spin_start_dates_series = pd.Series(spin_start_dates)

        # Find index of all timestamps that fall within 10 minutes after midnight
        repointing_times = spin_start_dates_series[
            (spin_start_dates_series.dt.time >= pd.Timestamp("00:00:00").time())
            & (spin_start_dates_series.dt.time < pd.Timestamp("00:10:00").time())
        ]

        repointing_times_index = repointing_times.index

        # Use the repointing times to set thruster firing flag and spin period valid
        spin_df.loc[repointing_times_index.values, "thruster_firing"] = 1
        spin_df.loc[repointing_times_index.values, "spin_period_valid"] = 0
        spin_df.loc[repointing_times_index.values, "spin_phase_valid"] = 0

        return spin_df

    return make_data


@pytest.fixture
def use_test_repoint_data_csv(monkeypatch):
    """Monkeypatches repoint._repoint_table_path to point to the input path."""

    def wrapped_set_repoint_data_filepath(path: Path):
        monkeypatch.setattr(spice_config, "_repoint_table_path", path)

    return wrapped_set_repoint_data_filepath


def generate_repoint_data(
    repoint_start_met: float | np.ndarray,
    repoint_end_met: float | np.ndarray | None = None,
    repoint_id_start: int | None = 0,
) -> pd.DataFrame:
    """
    Generate a repoint dataframe for the star/end times provided.

    Parameters
    ----------
    repoint_start_met : float, np.ndarray
            Provides the repoint start time(s) in Mission Elapsed Time (MET).
    repoint_end_met : float, np.ndarray, optional
        Provides the repoint end time(s) in MET. If not provided, end times
        will be 15 minutes after start times.
    repoint_id_start : int, optional
        Provides the starting repoint id number of the first repoint in the
        generated data.

    Returns
    -------
    repoint_df : pd.DataFrame
        Repoint dataframe with start and end repoint times provided and incrementing
        repoint_ids starting at 1.
    """
    repoint_start_times = np.atleast_1d(repoint_start_met)
    if repoint_end_met is None:
        repoint_end_met = repoint_start_times + 15 * 60

    # Calculate UTC times without spice (accepting ~5 second inaccuracy)
    repoint_start_dt64 = TTJ2000_EPOCH + (repoint_start_times * 1e9).astype(
        "timedelta64[ns]"
    )
    repoint_end_dt64 = TTJ2000_EPOCH + (repoint_end_met * 1e9).astype("timedelta64[ns]")
    repoint_df = pd.DataFrame.from_dict(
        {
            "repoint_start_sec_sclk": repoint_start_times.astype(int),
            "repoint_start_subsec_sclk": ((repoint_start_times % 1.0) * 1e6).astype(
                int
            ),
            "repoint_start_utc": np.datetime_as_string(repoint_start_dt64, unit="us"),
            "repoint_end_sec_sclk": repoint_end_met.astype(int),
            "repoint_end_subsec_sclk": ((repoint_end_met % 1.0) * 1e6).astype(int),
            "repoint_end_utc": np.datetime_as_string(repoint_end_dt64, unit="us"),
            "repoint_id": np.arange(repoint_start_times.size, dtype=int)
            + repoint_id_start,
        }
    )
    return repoint_df


@pytest.fixture
def use_fake_repoint_data_for_time(use_test_repoint_data_csv, tmp_path):
    """
    Generate and use fake spin data for testing.

    Returns
    -------
    callable
        Returns a callable function that takes start_met and optionally n_repoints
        as inputs, generates fake repoint data, writes the data to a csv file,
        and sets the REPOINT_DATA_FILEPATH environment variable to point to the
        fake repoint data file.
    """

    def wrapped_repoint_data_filepath(
        repoint_start_met: float | np.ndarray,
        repoint_end_met: float | np.ndarray | None = None,
        repoint_id_start: int | None = 0,
    ) -> pd.DataFrame:
        """
        Generate and use fake repoint data for testing.
        Parameters
        ----------
        repoint_start_met : float, np.ndarray
            Provides the repoint start time(s) in Mission Elapsed Time (MET).
        repoint_end_met : float, np.ndarray
            Provides the repoint end time(s) in MET. If not provided, end times
            will be 15 minutes after start times.
        repoint_id_start : int, optional
            Provides the starting repoint id number of the first repoint in the
            generated data.
        """
        repoint_df = generate_repoint_data(
            repoint_start_met,
            repoint_end_met=repoint_end_met,
            repoint_id_start=repoint_id_start,
        )
        repoint_csv_file_path = tmp_path / "repoint_data.repointing.csv"
        repoint_df.to_csv(repoint_csv_file_path, index=False)
        use_test_repoint_data_csv(repoint_csv_file_path)

    return wrapped_repoint_data_filepath


@pytest.fixture
def imap_ena_sim_metakernel(furnish_kernels, _download_kernels):
    kernels = [
        "imap_sclk_0000.tsc",
        "naif0012.tls",
        "imap_spk_demo.bsp",
        "sim_1yr_imap_attitude.bc",
        "imap_100.tf",
        "de440s.bsp",
        "imap_science_100.tf",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels) as k:
        yield k


@pytest.fixture
def imap_ialirt_sim_metakernel(furnish_kernels):
    kernels = ["imap_100.tf"]
    with furnish_kernels(kernels) as k:
        yield k


@pytest.fixture
def imap_simple_sim_metakernel(furnish_kernels):
    kernels = ["imap_sclk_0000.tsc", "naif0012.tls", "imap_spk_demo.bsp"]
    with furnish_kernels(kernels) as k:
        yield k


# Shared with i-alirt and mag tests
@pytest.fixture
def mag_test_l1b_calibration_data():
    imap_dir = Path(__file__).parent
    cal_file = (
        imap_dir
        / "mag"
        / "validation"
        / "calibration"
        / "imap_mag_l1b-calibration_20240229_v001.cdf"
    )
    calibration_data = load_cdf(cal_file)
    matrix_mago = calibration_data["MFOTOURFO"]
    time_shift_mago = calibration_data["OTS"]
    matrix_magi = calibration_data["MFITOURFI"]
    time_shift_magi = calibration_data["ITS"]

    return matrix_mago, time_shift_mago, matrix_magi, time_shift_magi


if __name__ == "__main__":
    # This is to enable downloading files easier by letting us
    # run this file directly
    _download_external_data()
    _download_external_kernels(imap_module_directory / "tests" / "spice" / "test_data")
