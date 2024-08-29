"""Global pytest configuration for the package."""

import re

import imap_data_access
import numpy as np
import pytest
import spiceypy as spice

from imap_processing import imap_module_directory


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


# Furnishing fixtures for testing kernels
# ---------------------------------------
@pytest.fixture(autouse=True)
def _autoclear_spice():
    """Automatically clears out all SPICE remnants after every single test to
    prevent the kernel pool from interfering with future tests. Option autouse
    ensures this is run after every test."""
    yield
    spice.kclear()


@pytest.fixture(scope="session")
def spice_test_data_path(imap_tests_path):
    return imap_tests_path / "spice/test_data"


@pytest.fixture()
def furnish_test_lsk(spice_test_data_path):
    """Furnishes (temporarily) the testing LSK"""
    test_lsk = spice_test_data_path / "naif0012.tls"
    spice.furnsh(test_lsk)
    yield test_lsk
    spice.kclear()


@pytest.fixture()
def furnish_sclk(spice_test_data_path):
    """Furnishes (temporarily) the SCLK for JPSS stored in the package data directory"""
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spice.furnsh(test_sclk)
    yield test_sclk
    spice.kclear()


@pytest.fixture()
def use_test_metakernel(monkeypatch, spice_test_data_path):
    """For the whole test session, set the SPICE_METAKERNEL environment variable
    Prime the test metakernel by creating it from the template metakernel
    (allows using absolute paths on any dev system)"""

    def make_metakernel_from_kernels(metakernel, kernels):
        """Helper function that writes a test metakernel from a list of filenames"""
        with open(metakernel, "w") as mk:
            mk.writelines(
                [
                    "\n",
                    "\\begintext\n",
                    "\n",
                    "This is a temporary metakernel for imap_processing"
                    " unit and integration testing.\n",
                    "\n",
                    "\\begindata\n",
                    "\n",
                    "KERNELS_TO_LOAD = (\n",
                ]
            )
            # Put single quotes around every kernel name
            kernels_with_quotes = ["    '" + kern + "'" for kern in kernels]
            # Add a comma and EOL to the end of each kernel path except the last.
            formated_kernels = [kern + ",\n" for kern in kernels_with_quotes[0:-1]]
            # Add ')' to the last kernel
            formated_kernels.append(kernels_with_quotes[-1] + "\n)\n\n")
            mk.writelines(formated_kernels)

    def get_test_kernels_to_load():
        """
        Helper function for grabbing a list of kernel filenames from the test
        metakernel template. This is necessary in order to get absolute paths on
        any system. Formats the absolute paths using the test data path fixture
        value.
        """
        test_metakernel = spice_test_data_path / "imap_test_metakernel.template"
        kernels_to_load = []
        max_line_length = 80
        with open(test_metakernel) as mk:
            for k in mk:
                kernel = k.rstrip("\n").format(
                    **{"SPICE_TEST_DATA_PATH": str(spice_test_data_path.absolute())}
                )
                while len(kernel) > 0:
                    if len(kernel) <= max_line_length:
                        kernels_to_load.append(kernel)
                        break
                    else:
                        slash_positions = np.array(
                            [m.start() for m in re.finditer("/", kernel)]
                        )
                        stop_idx = (
                            slash_positions[slash_positions < max_line_length - 1].max()
                            + 1
                        )
                        kernels_to_load.append(kernel[0:stop_idx] + "+")
                        kernel = kernel[stop_idx:]
        return kernels_to_load

    metakernel_path = imap_data_access.config["DATA_DIR"] / "imap_2024_v001.tm"
    kernels_to_load = get_test_kernels_to_load()
    make_metakernel_from_kernels(metakernel_path, kernels_to_load)
    monkeypatch.setenv("SPICE_METAKERNEL", str(metakernel_path))
    yield str(metakernel_path)
    spice.kclear()
