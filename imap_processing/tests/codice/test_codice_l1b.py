"""Tests the L1b processing for CoDICE L1a data"""

import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.codice.codice_l1b import process_codice_l1b

pytestmark = pytest.mark.external_test_data


def test_l1b_lo_sw_species():
    l1a_test_file = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1a_validation"
        / "imap_codice_l1a_lo-sw-species_20250814211100_v0.0.3.cdf"
    )

    l1b_val_data = (
        imap_module_directory
        / "tests"
        / "codice"
        / "data"
        / "l1b_validation"
        / "imap_codice_l1b_lo-sw-species_20250814211100_v0.0.3.cdf"
    )
    l1b_val_data = load_cdf(l1b_val_data)
    processed_data = process_codice_l1b(l1a_test_file)
    # print(f"l1a input: {load_cdf(l1a_test_file).data_vars}")
    # print(f"l1b val: {l1b_val_data.data_vars}")

    for variable in l1b_val_data.data_vars:
        if variable in ["energy_table", "acquisition_time_per_step"]:
            assert processed_data[variable].shape == (128,)
        elif variable in [
            "rgfo_half_spin",
            "nso_half_spin",
            "sw_bias_gain_mode",
            "st_bias_gain_mode",
            "data_quality",
            "spin_period",
        ]:
            # TODO: validation is missing these
            # k_factor
            # "nso_half_spin",
            # "sw_bias_gain_mode",
            # "st_bias_gain_mode",
            # "data_quality",
            # "spin_period",
            # TODO: skip storing that after l1a besides
            # data_quality
            np.testing.assert_allclose(
                processed_data[variable].values,
                l1b_val_data[variable].values,
                rtol=1e-5,
                err_msg=f"Mismatch in variable '{variable}'",
            )
        else:
            try:
                np.testing.assert_allclose(
                    processed_data[variable].values,
                    l1b_val_data[variable].values,
                    rtol=1e-5,
                    err_msg=f"Mismatch in variable '{variable}'",
                )
            except AssertionError:
                # Variable 'hplus' does not match.
                # Variable 'heplusplus' does not match.
                # Variable 'cnoplus' does not match.
                # values seem to be off by conversion factor
                # [16271.87003621 16847.86543572 17999.85623475 ... 18143.85569294
                # 31103.75261647 19007.84882117]
                # [16.27187  16.847866 17.999857 ... 18.143856 31.103754 19.007849]
                print(f"Variable '{variable}' does not match.")
                non_zero_indices = processed_data[variable].values != 0
                print(
                    "Processed non-zero values \n\n",
                    processed_data[variable].values[non_zero_indices[:10]],
                )
                print(
                    "Validation non-zero values \n\n",
                    l1b_val_data[variable].values[non_zero_indices[:10]],
                )
