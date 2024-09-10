"""Global pytest configuration for the package."""

import logging
import os
import re
import time
from typing import Optional

import cdflib
import imap_data_access
import numpy as np
import pandas as pd
import pytest
import requests
import spiceypy as spice

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import met_to_j2000ns


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
def _download_de440s(spice_test_data_path):
    """This fixture downloads the de440s.bsp kernel into the
    tests/spice/test_data directory if it does not already exist there. The
    fixture is not intended to be used directly. It is automatically added to
    tests marked with "external_kernel" in the hook below."""
    logger = logging.getLogger(__name__)
    kernel_url = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
    )
    kernel_name = kernel_url.split("/")[-1]
    local_filepath = spice_test_data_path / kernel_name

    if local_filepath.exists():
        return
    allowed_attempts = 3
    for attempt_number in range(allowed_attempts):
        try:
            with requests.get(kernel_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(local_filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info("Cached kernel file to %s", local_filepath)
            break
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
                    f"Failed to download file after {allowed_attempts} "
                    f"attempts, Final Error: {error}"
                )
                raise


def pytest_collection_modifyitems(items):
    """
    The use of this hook allows modification of test `Items` after tests have
    been collected. In this case, it automatically adds the _download_de440s
    fixture to any test marked with the `external_kernel`.
    https://docs.pytest.org/en/stable/reference/reference.html#pytest.hookspec.pytest_collection_modifyitems
    """
    for item in items:
        if item.get_closest_marker("external_kernel") is not None:
            item.fixturenames.append("_download_de440s")


@pytest.fixture(scope="session")
def spice_test_data_path(imap_tests_path):
    return imap_tests_path / "spice/test_data"


@pytest.fixture()
def furnish_time_kernels(spice_test_data_path):
    """Furnishes (temporarily) the testing LSK and SCLK"""
    spice.kclear()
    test_lsk = spice_test_data_path / "naif0012.tls"
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spice.furnsh(str(test_lsk))
    spice.furnsh(str(test_sclk))
    yield test_lsk, test_sclk
    spice.kclear()


@pytest.fixture()
def furnish_sclk(spice_test_data_path):
    """Furnishes (temporarily) the SCLK for JPSS stored in the package data directory"""
    test_sclk = spice_test_data_path / "imap_sclk_0000.tsc"
    spice.furnsh(str(test_sclk))
    yield test_sclk
    spice.kclear()


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch

    m = MonkeyPatch()
    yield m
    m.undo()


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
        formatted_kernels = [kern + ",\n" for kern in kernels_with_quotes[0:-1]]
        # Add ')' to the last kernel
        formatted_kernels.append(kernels_with_quotes[-1] + "\n)\n\n")
        mk.writelines(formatted_kernels)


def get_test_kernels_to_load(template_path, kernel_dir_path):
    """
    Helper function for grabbing a list of kernel filenames from the test
    metakernel template. This is necessary in order to get absolute paths on
    any system. Formats the absolute paths using the test data path fixture
    value.
    """
    kernels_to_load = []
    max_line_length = 80
    with open(template_path) as mk:
        for k in mk:
            kernel = k.rstrip("\n").format(
                **{"SPICE_TEST_DATA_PATH": str(kernel_dir_path.absolute())}
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
                        slash_positions[slash_positions < max_line_length - 1].max() + 1
                    )
                    kernels_to_load.append(kernel[0:stop_idx] + "+")
                    kernel = kernel[stop_idx:]
    return kernels_to_load


@pytest.fixture(scope="session", autouse=True)
def session_test_metakernel(monkeypatch_session, tmpdir_factory, spice_test_data_path):
    """Generate a metakernel from the template metakernel by injecting the local
    path into the metakernel and set the SPICE_METAKERNEL environment variable.

    Notes
    -----
    - This fixture needs to `scope=session` so that the SPICE_METAKERNEL
    environment variable is available for other fixtures that require time
    conversions using spice.
    - No furnishing of kernels occur as part of this fixture. This allows other
    fixtures with lesser scope or individual tests to override the environment
    variable as needed. Use the `metakernel_path_not_set` fixture in tests that
    need to override the environment variable.
    """
    template_path = spice_test_data_path / "imap_simple_metakernel.template"
    kernels_to_load = get_test_kernels_to_load(template_path, spice_test_data_path)
    metakernel_path = tmpdir_factory.mktemp("spice") / "imap_2024_v001.tm"
    make_metakernel_from_kernels(metakernel_path, kernels_to_load)
    monkeypatch_session.setenv("SPICE_METAKERNEL", str(metakernel_path))
    yield str(metakernel_path)
    spice.kclear()


@pytest.fixture()
def use_test_metakernel(
    request, monkeypatch, spice_test_data_path, session_test_metakernel
):
    """
    Generate a metakernel and set SPICE_METAKERNEL environment variable.

    This fixture generates a metakernel in the directory pointed to by
    `imap_data_access.config["DATA_DIR"]` and sets the SPICE_METAKERNEL
    environment variable to point to it for use by the `@ensure_spice` decorator.
    The default metekernel is named "imap_simple_metakernel.template". Other
    metakerels can be specified by marking the test with metakernel. See
    examples below.

    Parameters
    ----------
    request : fixture
    monkeypatch : fixture
    spice_test_data_path : fixture
    session_test_metakernel : fixture

    Yields
    ------
    metakernel_path : Path

    Examples
    --------
    1. Use the default metakernel template
        >>> def test_my_spicey_func(use_test_metakernel):
        ...     pass

    2. Specify a different metakernel template
        >>> @pytest.mark.metakernel("other_template_mk.template")
        ... def test_my_spicey_func(use_test_metakernel):
        ...     pass
    """
    marker = request.node.get_closest_marker("metakernel")
    if marker is None:
        yield session_test_metakernel
    else:
        template_name = marker.args[0]
        template_path = spice_test_data_path / template_name
        metakernel_path = imap_data_access.config["DATA_DIR"] / "imap_2024_v001.tm"
        kernels_to_load = get_test_kernels_to_load(template_path, spice_test_data_path)
        make_metakernel_from_kernels(metakernel_path, kernels_to_load)
        monkeypatch.setenv("SPICE_METAKERNEL", str(metakernel_path))
        yield str(metakernel_path)
    spice.kclear()


@pytest.fixture()
def _unset_metakernel_path(monkeypatch):
    """Temporarily unsets the SPICE_METAKERNEL environment variable"""
    if os.getenv("SPICE_METAKERNEL", None) is not None:
        monkeypatch.delenv("SPICE_METAKERNEL")
