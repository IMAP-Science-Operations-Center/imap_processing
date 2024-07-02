"""Testing for xtce generator template."""

import sys
from unittest import mock

import pytest

from tools.xtce_generation.xtce_generator_template import main


# General test
@mock.patch("tools.xtce_generation.xtce_generator_template.main")
def test_main_general(mock_input):
    """Testing base main function."""
    test_args = [
        "test_script",
        "--instrument",
        "swe",
        "--filename",
        "TLM_SWP_20231006-121021.xlsx",
        "--packets",
        '{"P_SWP_HK": 1184, "P_SWP_SCI": 1188, "P_SWP_AUT": 1192}',
    ]
    with mock.patch.object(sys, "argv", test_args):
        main()


# Testing invalid instrument
@mock.patch("tools.xtce_generation.xtce_generator_template.main")
def test_main_inval_instr(mock_input):
    """Testing with invalid instrument."""
    test_args = [
        "test_script",
        "--instrument",
        "ana",
        "--filename",
        "TLM_SWP_20231006-121021.xlsx",
        "--packets",
        '{"P_SWP_HK": 1184, "P_SWP_SCI": 1188, "P_SWP_AUT": 1192}',
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(
            ValueError, match="ana is not in the supported instrument list:"
        ):
            main()


# Testing without required arguments
@mock.patch("tools.xtce_generation.xtce_generator_template.main")
def test_main_inval_arg(mock_input):
    """Testing with invalid instrument."""
    test_args = [
        "test_script",
        "--instrument",
        "glows",
        "--packets",
        '{"P_SWP_HK": 1184, "P_SWP_SCI": 1188, "P_SWP_AUT": 1192}',
    ]
    with mock.patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            main()
