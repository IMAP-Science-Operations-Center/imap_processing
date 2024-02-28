import dataclasses

import pytest
from cdflib.xarray import cdf_to_xarray

from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import ULTRA_AUX
from imap_processing.ultra.l1a.ultra_l1a import ultra_l1a, xarray_aux


@pytest.fixture()
def decom_ultra_aux(ccsds_path, xtce_path):
    """Data for decom_ultra_aux"""
    data_packet_list = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_AUX.apid[0])
    return data_packet_list


def test_xarray_aux(decom_ultra_aux, aux_test_path):
    """This function checks that an xarray was
    successfully created from the decom_ultra_aux data."""

    dataset = xarray_aux(decom_ultra_aux)

    # Spot check string data and attributes
    spin_period_valid_list = dataset.variables["SPINPERIODVALID"].values.tolist()
    spin_period_valid_attr = dataset.variables["SPINPERIODVALID"].attrs
    expected_spin_period_valid_attr = ultra_cdf_attrs.StringAttrs(
        depend_0="Epoch", catdesc="SPINPERIODVALID", fieldname="SPINPERIODVALID"
    )

    assert spin_period_valid_list == decom_ultra_aux["SPINPERIODVALID"]
    assert spin_period_valid_attr == expected_spin_period_valid_attr.output()

    # Spot check support data and attributes
    version_list = dataset.variables["VERSION"].values.tolist()
    version_attr = dataset.variables["VERSION"].attrs
    expected_version_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="VERSION",
        fieldname="VERSION",
    ).output()

    assert version_list == decom_ultra_aux["VERSION"]
    assert version_attr == expected_version_attr

    # Spot check metadata data and attributes
    shcoarse_list = dataset.variables["SHCOARSE"].values.tolist()
    shcoarse_attr = dataset.variables["SHCOARSE"].attrs

    expected_shcoarse_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_metadata_attrs,
        catdesc="SHCOARSE",
        fieldname="SHCOARSE",
        label_axis="SHCOARSE",
    ).output()

    assert shcoarse_list == decom_ultra_aux["SHCOARSE"]
    assert shcoarse_attr == expected_shcoarse_attr


def test_cdf_aux(ccsds_path, xtce_path, tmp_path, decom_ultra_aux):
    """Tests that CDF file is created and contains same attributes as xarray_aux."""
    # TODO: change test filename with new naming convention
    test_data_path = tmp_path / "ultra_l1a_aux_20210101-20210102_v01-01.cdf"

    assert not test_data_path.exists()
    ultra_l1a(ccsds_path, xtce_path, test_data_path)
    assert test_data_path.exists()

    dataset = xarray_aux(decom_ultra_aux)
    input_xarray = cdf_to_xarray(test_data_path)
    assert input_xarray.attrs.keys() == dataset.attrs.keys()
