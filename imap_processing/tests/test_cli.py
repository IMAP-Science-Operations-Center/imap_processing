"""Tests coverage for imap_processing.cli."""

import json
import logging
import shutil
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, sentinel

import imap_data_access.io
import numpy as np
import pytest
import spiceypy
import xarray as xr
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
    SPICEInput,
    SpinInput,
)

from imap_processing.cli import (
    Codice,
    Glows,
    Hi,
    Hit,
    Idex,
    Lo,
    Spacecraft,
    Swe,
    Ultra,
    _parse_args,
    _validate_args,
    main,
)
from imap_processing.spice import config as spice_config


@pytest.fixture(autouse=True)
def clear_spice_kernels():
    """Fixture to clear SPICE kernels before each test."""
    with spiceypy.KernelPool([]):
        yield


@pytest.fixture
def mock_instrument_dependencies():
    with (
        mock.patch("imap_processing.cli.imap_data_access.query") as mock_query,
        mock.patch("imap_processing.cli.imap_data_access.download") as mock_download,
        mock.patch("imap_processing.cli.imap_data_access.upload") as mock_upload,
        mock.patch("imap_processing.cli.write_cdf") as mock_write_cdf,
        mock.patch("imap_processing.cli.load_cdf") as mock_load_cdf,
        mock.patch(
            "imap_processing.cli.ProcessInstrument.pre_processing"
        ) as mock_pre_processing,
    ):
        mocks = {
            "mock_query": mock_query,
            "mock_download": mock_download,
            "mock_upload": mock_upload,
            "mock_write_cdf": mock_write_cdf,
            "mock_load_cdf": mock_load_cdf,
            "mock_pre_processing": mock_pre_processing,
        }
        yield mocks


@mock.patch("imap_processing.cli.Mag")
def test_main(mock_instrument):
    """Test imap_processing.cli.main()"""
    test_args = [
        "imap_cli",
        "--instrument",
        "mag",
        "--dependency",
        (
            "["
            "{"
            '"type": "ancillary",'
            '"files": ['
            '"imap_mag_l1b-cal_20250101_v001.cdf",'
            '"imap_mag_l1b-cal_20250103-20250104_v002.cdf"'
            "]"
            "},"
            "{"
            '"type": "science",'
            '"files": ['
            '"imap_mag_l0_raw_20240430_v001.cdf",'
            "]"
            "}"
            "]"
        ),
        "--data-level",
        "l1a",
        "--start-date",
        "20240430",
        "--repointing",
        "repoint12345",
        "--version",
        "v001",
        "--upload-to-sdc",
    ]
    with mock.patch.object(sys, "argv", test_args):
        # Running without raising an exception is a pass.
        # No asserts needed.
        main()


def test_parse_args_dependency_json_file(caplog, tmp_path):
    # Set caplog to capture all log levels
    caplog.set_level(logging.DEBUG)
    """Test imap_processing.cli.main() with --dependency as a JSON file path."""
    test_json_content = [
        {
            "type": "ancillary",
            "files": [
                "imap_mag_l1b-cal_20250101_v001.cdf",
                "imap_mag_l1b-cal_20250103_20250104_v002.cdf",
            ],
        },
        {
            "type": "science",
            "files": [
                "imap_idex_l2_sci_20240312_v000.cdf",
                "imap_idex_l2_sci_20240312_v001.cdf",
            ],
        },
    ]
    test_json_filename = "imap_ultra_l2_test-dependency-json_20250520_v999.json"
    test_json_dir = tmp_path / "imap/dependency/ultra/l2/2025/05/"
    test_json_dir.mkdir(parents=True, exist_ok=True)
    test_json_dst = test_json_dir / test_json_filename

    with open(test_json_dst, "w") as f:
        f.write(json.dumps(test_json_content))

    test_args = [
        "imap_cli",
        "--instrument",
        "mag",
        "--dependency",
        str(test_json_dst),
        "--data-level",
        "l1a",
        "--start-date",
        "20240430",
        "--repointing",
        "repoint12345",
        "--version",
        "v001",
        "--upload-to-sdc",
    ]
    with mock.patch.object(sys, "argv", test_args):
        _parse_args()
        # Check that the dependency JSON file was read correctly
        assert "Interpreting dependency argument as a JSON file" in caplog.text, (
            "Dependency JSON file was not read correctly"
        )


@pytest.mark.parametrize(
    "instrument, data_level, start_date, repointing, raises_value_error",
    [
        ("mag", "l1a", "20250101", None, ""),
        ("foo", "l1a", None, None, "foo is not in the supported .*"),
        ("codice", "l1z", None, None, "l1z is not a supported .*"),
        ("glows", "l1a", None, "repoint12345", ""),
        ("glows", "l1a", None, "12345", ".* not a valid repointing.*"),
        ("glows", "l1a", "2000001", None, ".* not a valid date.*"),
    ],
)
def test_validate_args(
    instrument, data_level, start_date, repointing, raises_value_error
):
    """Test coverage for imap_processing.cli._validate_args()"""
    args = mock.Mock
    args.instrument = instrument
    args.data_level = data_level
    args.start_date = start_date
    args.repointing = repointing

    if raises_value_error:
        with pytest.raises(ValueError, match=raises_value_error):
            _validate_args(args)
    else:
        _validate_args(args)


@mock.patch("imap_processing.cli.codice_l1a.process_l1a")
def test_codice(mock_process_l1a, mock_instrument_dependencies):
    """Test coverage for cli.CoDICE class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_codice_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_process_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_codice_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Codice("l1a", "hskp", dependency_str, "20230822", None, "v001", False)

    instrument.process()
    assert mock_process_l1a.call_count == 1
    # Assert that write_cdf was called with the expected arguments
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


def test_repointing_file_creation(mock_instrument_dependencies):
    test_datasets = [xr.Dataset({}, attrs={"cdf_filename": "file0"})]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_glows_l0_raw_20230822-repoint00001_v001.pkts")
    )
    dependency_str = (
        '[{"type": "science","files": '
        '["imap_glows_l0_raw_20230822-repoint00001_v001.pkts"]}]'
    )
    instrument = Glows(
        "l1a", "hist", dependency_str, None, "repoint00002", "v001", False
    )

    mock_instrument_dependencies["mock_write_cdf"].side_effect = ["/path/to/file0"]

    # Call the method that uses write_cdf
    instrument.post_processing(test_datasets, input_collection)

    # Assert that write_cdf was called with the expected arguments
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1
    assert (
        mock_instrument_dependencies["mock_write_cdf"]
        .call_args[0][0]
        .attrs.get("Repointing", None)
        == "repoint00002"
    )


def test_post_processing_returns_path_to_written_cdf(mock_instrument_dependencies):
    test_datasets = [xr.Dataset({}, attrs={"cdf_filename": "file0"})]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_glows_l0_raw_20230822-repoint00001_v001.pkts")
    )
    dependency_str = (
        '[{"type": "science","files": '
        '["imap_glows_l0_raw_20230822-repoint00001_v001.pkts"]}]'
    )
    instrument = Glows(
        "l1a", "hist", dependency_str, None, "repoint00002", "v001", False
    )

    expected_path = "/path/to/file0"
    mock_instrument_dependencies["mock_write_cdf"].side_effect = [expected_path]

    # Call the method that uses write_cdf
    returned_path = instrument.post_processing(test_datasets, input_collection)

    # Assert that post_processing returned the path to the CDF written in write_cdf
    assert returned_path == [expected_path]


def test_post_processing_returns_empty_list_if_invoked_with_no_data(
    mock_instrument_dependencies,
):
    test_datasets = []
    input_collection = ProcessingInputCollection()
    instrument = Glows("l1a", "hist", "", None, "repoint00002", "v001", False)

    # Call the method that uses write_cdf
    returned_products = instrument.post_processing(test_datasets, input_collection)

    # Assert that post_processing returned the path to the CDF written in write_cdf
    assert returned_products == []


@pytest.mark.parametrize(
    "data_level, data_descriptor, function_name, science_input, anc_input, n_prods",
    [
        ("l1a", "sci", "hi_l1a", ["imap_hi_l0_raw_20231212_v001.pkts"], [], 2),
        (
            "l1b",
            "90sensor-de",
            "annotate_direct_events",
            [
                "imap_hi_l1a_90sensor-de_20241105_v001.cdf",
                "imap_hi_l1b_90sensor-hk_20241105_v001.cdf",
            ],
            ["imap_hi_90sensor-esa-energies_20240101_v001.csv"],
            1,
        ),
        ("l1b", "sci", "housekeeping", ["imap_hi_l0_raw_20231212_v001.pkts"], [], 2),
        (
            "l1c",
            "45sensor-pset",
            "hi_l1c",
            [
                "imap_hi_l1b_45sensor-de_20250415_v001.cdf",
                "imap_hi_l1b_45sensor-goodtimes_20250415_v001.cdf",
            ],
            ["imap_hi_calibration-prod-config_20240101_v001.csv"],
            1,
        ),
        (
            "l2",
            "h90-ena-h-sf-nsp-full-hae-4deg-3mo",
            "hi_l2",
            [
                "imap_hi_l1c_90sensor-pset_20250415_v001.cdf",
                "imap_hi_l1c_90sensor-pset_20250416_v001.cdf",
            ],
            [
                "imap_hi_calibration-prod-config_20240101_v001.csv",
                "imap_hi_90sensor-esa-energies_20240101_v001.csv",
                "imap_hi_90sensor-esa-eta-fit-factors_20240101_v001.csv",
            ],
            1,
        ),
    ],
)
def test_hi(
    mock_instrument_dependencies,
    data_level,
    data_descriptor,
    function_name,
    science_input,
    anc_input,
    n_prods,
):
    """Test coverage for cli.Hi class"""
    mocks = mock_instrument_dependencies
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"] * n_prods
    mocks["mock_load_cdf"].return_value = xr.Dataset()
    input_collection = ProcessingInputCollection(
        *[ScienceInput(file) for file in science_input],
        *[AncillaryInput(file) for file in anc_input],
    )
    mocks["mock_pre_processing"].return_value = input_collection

    # patch autospec=True makes this test confirm that the function call in cli.py
    # matches the mocked function signature.
    with mock.patch(
        f"imap_processing.cli.hi_{data_level}.{function_name}", autospec=True
    ) as mock_hi:
        mock_hi.return_value = [xr.Dataset()] * n_prods
        dependency_str = (
            '[{"type": "science","files": ["imap_hi_l0_raw_20231212_v001.pkts"]}]'
        )
        instrument = Hi(
            data_level,
            data_descriptor,
            dependency_str,
            "20231212",
            "repoint00001",
            "v005",
            False,
        )

        instrument.process()
        assert mock_hi.call_count == 1
        assert mock_instrument_dependencies["mock_write_cdf"].call_count == n_prods


@mock.patch("imap_processing.cli.hi_goodtimes.hi_goodtimes", autospec=True)
def test_hi_l1b_goodtimes(mock_hi_goodtimes, mock_instrument_dependencies):
    """Test coverage for cli.Hi class with l1b goodtimes descriptor"""
    mocks = mock_instrument_dependencies
    # goodtimes now returns xr.Dataset for CDF writing
    mock_goodtimes_ds = xr.Dataset()
    mock_hi_goodtimes.return_value = [mock_goodtimes_ds]
    mocks["mock_write_cdf"].return_value = Path("/path/to/goodtimes_output.cdf")

    # set load_cdf to return empty datasets
    mocks["mock_load_cdf"].return_value = xr.Dataset()
    # Set up the input collection with required dependencies
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00001_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00002_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00003_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00004_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00005_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00006_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-de_20250415-repoint00007_v001.cdf"),
        ScienceInput("imap_hi_l1b_45sensor-hk_20250415-repoint00004_v001.cdf"),
        ScienceInput("imap_hi_l1a_45sensor-diagfee_20250415-repoint00004_v001.cdf"),
        AncillaryInput("imap_hi_45sensor-cal-prod_20240101_v001.csv"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = input_collection.serialize()
    instrument = Hi(
        "l1b",
        "goodtimes",
        dependency_str,
        "20250415",
        "repoint00004",
        "v005",
        False,
    )

    instrument.process()

    # Verify load_cdf was called for DE files and HK file
    assert mocks["mock_load_cdf"].call_count == 9  # 7 DE + 1 HK + 1 DIAG_FEE

    # Verify hi_goodtimes was called with correct arguments
    assert mock_hi_goodtimes.call_count == 1
    call_args = mock_hi_goodtimes.call_args

    # Check that datasets (not paths) were passed for l1b_de_datasets and l1b_hk
    assert call_args.args[0] == "repoint00004"  # current_repointing
    assert isinstance(call_args.args[1], list)  # l1b_de_datasets is a list
    assert len(call_args.args[1]) == 7  # 7 DE datasets
    assert isinstance(call_args.args[2], xr.Dataset)  # l1b_hk is a dataset
    assert isinstance(call_args.args[3], xr.Dataset)  # l1a_diagfee is a dataset

    # goodtimes now returns xr.Dataset, so write_cdf should be called
    assert mocks["mock_write_cdf"].call_count == 1


@mock.patch("imap_processing.cli.lo_l2.lo_l2", autospec=True)
def test_lo_l2(mock_lo_l2, mock_instrument_dependencies):
    mocks = mock_instrument_dependencies

    descriptor = "some-ena-map-descriptor"

    mock_loaded_pset_1 = Mock(attrs={"Logical_source": "some_pset_logical_source"})
    pset_file_paths = [
        "imap_lo_l1c_pset_20250415_v001.cdf",
        "imap_lo_l1c_pset_20250416_v001.cdf",
    ]

    processing_input = ProcessingInputCollection(
        *[ScienceInput(file_path) for file_path in pset_file_paths],
    )

    mocks["mock_load_cdf"].side_effect = [mock_loaded_pset_1, sentinel.loaded_pset_2]
    mocks["mock_pre_processing"].return_value = processing_input

    output_l2_dataset = xr.Dataset()
    mock_lo_l2.return_value = [output_l2_dataset]

    instrument = Lo(
        "l2",
        descriptor,
        processing_input.serialize(),
        "20250415",
        "20250715",
        "v005",
        False,
    )
    instrument.process()

    mock_lo_l2.assert_called_once_with(
        {"some_pset_logical_source": [mock_loaded_pset_1, sentinel.loaded_pset_2]},
        [],
        descriptor,
    )
    mocks["mock_write_cdf"].assert_called_once_with(output_l2_dataset)


@mock.patch("imap_processing.cli.quaternions.process_quaternions", autospec=True)
def test_spacecraft(mock_spacecraft_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Spacecraft class"""

    test_dataset = xr.Dataset({}, attrs={"cdf_filename": "file0"})
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_spacecraft_l0_raw_20230822_v001.pkts")
    )
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_spacecraft_l1a.return_value = [test_dataset]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_spacecraft_l0_raw_20230822_v001.pkts"]}]'
    )

    instrument = Spacecraft(
        "l1a", "quaternions", dependency_str, "20230822", "20230822", "v001", False
    )

    instrument.process()
    assert mock_spacecraft_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


@mock.patch(
    "imap_processing.cli.pointing_frame.generate_pointing_attitude_kernel",
    autospec=True,
)
def test_spacecraft_pointing_kernel(
    mock_spacecraft_pointing, mock_instrument_dependencies
):
    """Test coverage for cli.Spacecraft class"""

    dependency_str = (
        '[{"type": "spice","files": ["naif0012.tls", '
        '"imap_sclk_0005.tsc", "imap_2024_100_2024_111_05.ah.bc"]}]'
    )
    input_collection = ProcessingInputCollection()
    input_collection.deserialize(dependency_str)
    mocks = mock_instrument_dependencies
    mocks["mock_query"].return_value = [{"file_path": "/path/to/file0"}]
    mocks["mock_download"].return_value = "file0"
    mock_spacecraft_pointing.return_value = [
        Path("imap_dps_2024_100_2024_111_05.ah.bc")
    ]
    mocks["mock_write_cdf"].side_effect = ["/path/to/file0"]
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Spacecraft(
        "l1a", "pointing-attitude", dependency_str, "20240410", "12345", "v005", False
    )

    instrument.process()
    assert mock_spacecraft_pointing.call_count == 1


@mock.patch("imap_processing.cli.ultra_l1a.ultra_l1a")
def test_ultra_l1a(mock_ultra_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1a.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l0_raw_20240207_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_ultra_l0_raw_20240207_v001.pkts"]}]'
    )
    instrument = Ultra(
        "l1a", "raw", dependency_str, "20240207", "20240208", "v001", False
    )

    instrument.process()
    assert mock_ultra_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1b.ultra_l1b")
def test_ultra_l1b(mock_ultra_l1b, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1b data level"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    mock_ultra_l1b.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1a_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1b", "de", "[]", "20240207", "20240208", "v001", False)

    instrument.process()
    assert mocks["mock_download"].call_count == 0
    assert mock_ultra_l1b.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l1c.ultra_l1c")
def test_ultra_l1c(mock_ultra_l1c, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l1c data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l1c.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1b_de_20240207_v001.cdf")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra("l1c", "pset", "[]", "20240207", "20240208", "v001", False)

    instrument.process()
    assert mock_ultra_l1c.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.ultra_l2.ultra_l2")
def test_ultra_l2(mock_ultra_l2, mock_instrument_dependencies):
    """Test coverage for cli.Ultra class with l2 data level"""
    mocks = mock_instrument_dependencies
    mock_ultra_l2.return_value = [xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240207_v001.cdf"),
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240208_v001.cdf"),
        ScienceInput("imap_ultra_l1c_90sensor-pset_20240209_v001.cdf"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    instrument = Ultra(
        "l2",
        "u90-ena-h-sf-nsp-full-hae-2deg-3mo",
        "[]",
        "20240207",
        "20240207",
        "v001",
        False,
    )

    instrument.process()
    assert mock_ultra_l2.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 1


@mock.patch("imap_processing.cli.idex_l1b")
def test_idex_l1b(mock_idex_l1b, mock_instrument_dependencies):
    """Test coverage for cli.Idex class with l1b data level"""
    mocks = mock_instrument_dependencies
    new_ds = xr.Dataset(data_vars={"epoch": [1]})
    old_ds = xr.Dataset(data_vars={"epoch": [0]})
    mocks["mock_load_cdf"].side_effect = [old_ds, new_ds]
    input_collection = ProcessingInputCollection(
        ScienceInput(
            "imap_idex_l1a_sci-1week_20251017_v001.cdf",
            "imap_idex_l1a_sci-1week_20251012_v001.cdf",
        ),
        SPICEInput("naif0012.tls", "imap_sclk_0000.tsc"),
        SpinInput("imap_2025_306_2025_307_01.spin"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = input_collection.serialize()
    instrument = Idex(
        "l1b", "sci-1week", dependency_str, "20251017", "20251017", "v001", False
    )

    instrument.process()
    assert mock_idex_l1b.call_count == 1
    # Assert that the dataset with the newer epoch value was passed to idex_l1b for
    # processing
    xr.testing.assert_equal(mock_idex_l1b.call_args[0][0], new_ds)


@mock.patch("imap_processing.cli.idex_l2b")
def test_idex_l2b(mock_idex_l2b, mock_instrument_dependencies):
    """Test coverage for cli.Idex class with l2b data level"""
    mocks = mock_instrument_dependencies
    mock_idex_l2b.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_idex_l1b_evt_20251015_v002.cdf"),
        ScienceInput("imap_idex_l2a_sci-1week_20251017_v018.cdf"),
        SPICEInput("naif0012.tls", "imap_sclk_0000.tsc"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = input_collection.serialize()
    instrument = Idex(
        "l2b", "all", dependency_str, "20100105", "20100101", "v001", False
    )

    instrument.process()
    assert mock_idex_l2b.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


@mock.patch("imap_processing.cli.hit_l1a")
def test_hit_l1a(mock_hit_l1a, mock_instrument_dependencies):
    """Test coverage for cli.Hit class with l1a data level"""
    mocks = mock_instrument_dependencies
    mock_hit_l1a.return_value = [xr.Dataset(), xr.Dataset()]
    mocks["mock_write_cdf"].side_effect = ["/path/to/product0", "/path/to/product1"]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_hit_l0_raw_20100105_v001.pkts")
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_hit_l0_raw_20100105_v001.pkts"]}]'
    )
    instrument = Hit(
        "l1a", "raw", dependency_str, "20100105", "20100101", "v001", False
    )

    instrument.process()
    assert mock_hit_l1a.call_count == 1
    assert mock_instrument_dependencies["mock_write_cdf"].call_count == 2


def test_spice_kernel_handling(spice_test_data_path, clear_spice_kernels):
    """Test coverage for ProcessInstrument.pre_processing method()."""
    kernels_to_furnish = ["naif0012.tls", "imap_sclk_0000.tsc"]
    dependency_obj = [
        {"type": "science", "files": ["imap_hi_l2a_sensor45-de_20100105_v001.cdf"]},
        {"type": "spice", "files": kernels_to_furnish},
        {"type": "repoint", "files": ["imap_2010_104_01.repoint.csv"]},
    ]
    dependency_str = json.dumps(dependency_obj)

    def download_side_effect(path: Path):
        """Copy kernels from spice_test_data_path to expected DATA_DIR location."""
        if (spice_test_data_path / path.name).exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spice_test_data_path / path.name, path)

    # Test that the expected kernels are furnished within the function that
    # gets substituted in for the processing function.
    def do_processing_side_effect(*args, **kwargs):
        """Check that the expected kernels are furnished"""
        kernel_count = spiceypy.ktotal("ALL")
        furnished_kernels = [
            Path(spiceypy.kdata(i, "ALL")[0]).name for i in range(kernel_count)
        ]
        np.testing.assert_array_equal(
            sorted(furnished_kernels), sorted(kernels_to_furnish)
        )
        return [xr.Dataset()]

    with (
        mock.patch("imap_data_access.processing_input.download") as mock_download,
        mock.patch("imap_processing.cli.load_cdf"),
        mock.patch("imap_processing.cli.Hi.do_processing") as mock_do_processing,
        mock.patch("imap_processing.cli.write_cdf"),
        mock.patch("imap_processing.cli.ProcessInstrument.upload_products"),
    ):
        mock_download.side_effect = download_side_effect
        mock_do_processing.side_effect = do_processing_side_effect

        instrument = Hi(
            "l1b", "sensor45-de", dependency_str, "20100105", None, "v001", True
        )
        # Verify no kernels are furnished prior to calling process
        assert spiceypy.ktotal("ALL") == 0
        # Verification that the expected kernels get furnished is done in the
        # mocked do_processing function
        instrument.process()
        # Verify that the furnished kernels get cleared in the post_processing method
        assert spiceypy.ktotal("ALL") == 0


def test_spin_and_repoint_table_handling():
    """Test ProcessInstrument.pre_processing setting of spin and repoint paths."""
    dependency_obj = [
        {"type": "science", "files": ["imap_hi_l2a_sensor45-de_20100105_v001.cdf"]},
        {"type": "repoint", "files": ["imap_2010_104_01.repoint.csv"]},
        {
            "type": "spin",
            "files": [
                "imap_2025_104_2025_107_01.spin.csv",
                "imap_2025_107_2025_112_01.spin.csv",
            ],
        },
    ]
    dependency_str = json.dumps(dependency_obj)

    # Test that expected spin and repoint paths are set
    def do_processing_side_effect(*args, **kwargs):
        """Check that the expected kernels are furnished"""
        assert spice_config._repoint_table_path.name == dependency_obj[1]["files"][0]
        np.testing.assert_array_equal(
            dependency_obj[2]["files"], [p.name for p in spice_config._spin_table_paths]
        )
        return [xr.Dataset()]

    with (
        mock.patch("imap_data_access.processing_input.download"),
        mock.patch("imap_processing.cli.load_cdf"),
        mock.patch("imap_processing.cli.Hi.do_processing") as mock_do_processing,
        mock.patch("imap_processing.cli.write_cdf"),
        mock.patch("imap_processing.cli.ProcessInstrument.upload_products"),
    ):
        mock_do_processing.side_effect = do_processing_side_effect

        instrument = Hi(
            "l1b", "sensor45-de", dependency_str, "20100105", None, "v001", True
        )
        # Verify no paths are set
        assert spice_config._spin_table_paths == []
        assert spice_config._repoint_table_path is None
        # Verification that the expected paths are set is done in the
        # do_processing_side_effect to ensure they are correct during processing
        instrument.process()


@mock.patch("imap_processing.cli.swe_l1a")
@pytest.mark.parametrize(
    "query_return, expected_warning",
    [
        ([], False),
        (
            [
                '{"file_path": '
                '"/path/to/imap_swe_l1a_test_20100105_v001.cdf", "instrument": "swe"}'
            ],
            True,
        ),
    ],
)
def test_post_processing(
    mock_swe_l1a, mock_instrument_dependencies, query_return, expected_warning
):
    """Test coverage for post processing"""
    mocks = mock_instrument_dependencies
    mocks["mock_download"].return_value = "dependency0"
    mocks["mock_write_cdf"].side_effect = [
        "/path/to/imap_swe_l1a_test_20100105_v001.cdf"
    ]
    mocks[
        "mock_write_cdf"
    ].return_value = "/path/to/imap_swe_l1a_test_20100105_v001.cdf"
    mocks["mock_query"].return_value = query_return

    # Setup for testing file exists handling in upload_products
    if expected_warning:
        # Mock the upload method to simulate a file exists error
        mocks["mock_upload"].side_effect = imap_data_access.io.IMAPDataAccessError(
            '409 Conflict: {"error": "FileAlreadyExists", '
            '"message": "The file already exists."}'
        )

    test_ds = xr.Dataset()
    mock_swe_l1a.return_value = [test_ds]
    input_collection = ProcessingInputCollection(
        ScienceInput("imap_swe_l0_raw_20100105_v001.pkts"),
        SPICEInput("naif0012.tls", "imap_sclk_0001.tsc"),
    )
    mocks["mock_pre_processing"].return_value = input_collection

    dependency_str = (
        '[{"type": "science","files": ["imap_swe_l0_raw_20100105_v001.pkts"]}, '
        '{"type": "spice", "files": ["naif0012.tls", "imap_sclk_0001.tsc"]}]'
    )
    instrument = Swe("l1a", "raw", dependency_str, "20100105", None, "v001", True)

    # Now we expect the process to complete without errors in both cases
    # The current implementation should skip over the file exists error
    with mock.patch("logging.Logger.warning") as mock_warning:
        instrument.process()
        if expected_warning:
            # Verify that we saw a warning about skipping upload
            assert any(
                "Skipping upload" in str(call) for call in mock_warning.call_args_list
            )
        else:
            # This is testing that one file was uploaded successfully
            assert mocks["mock_upload"].call_count == 1

    # Test parent injection
    assert test_ds.attrs["Parents"] == [
        "imap_swe_l0_raw_20100105_v001.pkts",
        "naif0012.tls",
        "imap_sclk_0001.tsc",
    ]
