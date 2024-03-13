import dataclasses

import pytest
from cdflib.xarray import cdf_to_xarray

from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import create_dataset, ultra_l1a


@pytest.fixture()
def decom_ultra_aux(ccsds_path, xtce_path):
    """Data for decom_ultra_aux"""
    data_packet_list = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_AUX.apid[0])
    return data_packet_list


@pytest.fixture()
def decom_ultra_rates(ccsds_path, xtce_path):
    """Data for decom_ultra_rates"""
    data_packet_list = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_RATES.apid[0])
    return data_packet_list


@pytest.fixture()
def decom_ultra_events(ccsds_path_events, xtce_path):
    """Data for decom_ultra_events"""
    data_packet_list = decom_ultra_apids(
        ccsds_path_events, xtce_path, ULTRA_EVENTS.apid[0]
    )
    return data_packet_list


@pytest.fixture()
def decom_ultra_tof(ccsds_path_tof, xtce_path):
    """Data for decom_ultra_tof"""
    data_packet_list = decom_ultra_apids(ccsds_path_tof, xtce_path, ULTRA_TOF.apid[0])
    return data_packet_list


def test_xarray_aux(decom_ultra_aux, aux_test_path):
    """This function checks that a xarray was
    successfully created from the decom_ultra_aux data."""

    dataset = create_dataset({ULTRA_AUX.apid[0]: decom_ultra_aux})

    # Spot check string data and attributes
    spin_period_valid_list = dataset.variables["SPINPERIODVALID"].values.tolist()
    spin_period_valid_attr = dataset.variables["SPINPERIODVALID"].attrs
    expected_spin_period_valid_attr = ultra_cdf_attrs.StringAttrs(
        depend_0="epoch", catdesc="spinperiodvalid", fieldname="spinperiodvalid"
    )

    assert spin_period_valid_list == decom_ultra_aux["SPINPERIODVALID"]
    assert spin_period_valid_attr == expected_spin_period_valid_attr.output()

    # Spot check support data and attributes
    version_list = dataset.variables["VERSION"].values.tolist()
    version_attr = dataset.variables["VERSION"].attrs
    expected_version_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="version",
        fieldname="version",
        label_axis="version",
    ).output()

    assert version_list == decom_ultra_aux["VERSION"]
    assert version_attr == expected_version_attr

    # Spot check metadata data and attributes
    shcoarse_list = dataset.variables["SHCOARSE"].values.tolist()
    shcoarse_attr = dataset.variables["SHCOARSE"].attrs

    expected_shcoarse_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="shcoarse",
        fieldname="shcoarse",
        label_axis="shcoarse",
    ).output()

    assert shcoarse_list == decom_ultra_aux["SHCOARSE"]
    assert shcoarse_attr == expected_shcoarse_attr


def test_xarray_rates(decom_ultra_rates, rates_test_path):
    """This function checks that a xarray was
    successfully created from the decom_ultra_rates data."""

    dataset = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2022-05-30T22:52:00.184000")["START_RF"]
    startrf_list = specific_epoch_data.values.tolist()
    startrf_attr = dataset.variables["START_RF"].attrs

    expected_startrf_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="start_rf",
        fieldname="start_rf",
        label_axis="start_rf",
    ).output()

    assert startrf_list == decom_ultra_rates["START_RF"][0]
    assert startrf_attr == expected_startrf_attr


def test_xarray_tof(decom_ultra_tof, tof_test_path):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""

    dataset = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2024-01-24T11:39:21.184000", sid=0)[
        "PACKETDATA"
    ]
    packetdata_attr = dataset.variables["PACKETDATA"].attrs

    expected_packetdata_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="packetdata",
        fieldname="packetdata",
        label_axis="packetdata",
        depend_1="sid",
        depend_2="row",
        depend_3="column",
        units="pixels",
        variable_purpose="primary_var",
    ).output()

    assert (specific_epoch_data == decom_ultra_tof["PACKETDATA"][0][0]).all()
    assert packetdata_attr == expected_packetdata_attr


def test_xarray_events(decom_ultra_events, decom_ultra_aux, events_test_path):
    """This function checks that a xarray was
    successfully created from the decom_ultra_events data."""

    # Create aux test data with timestamps within the same range as events data
    start_value = decom_ultra_events["SHCOARSE"][0] - 1
    increment = 14  # 14 seconds between each aux data packet
    sequence_length = len(decom_ultra_aux["SHCOARSE"])

    new_shcoarse_values = [start_value + i * increment for i in range(sequence_length)]

    decom_ultra_aux["SHCOARSE"] = new_shcoarse_values
    decom_ultra_aux["TIMESPINSTART"] = new_shcoarse_values

    dataset = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2023-08-21T16:14:10.917929")["COIN_TYPE"]
    cointype_list = specific_epoch_data.values.tolist()
    cointype_attr = dataset.variables["COIN_TYPE"].attrs

    expected_cointype_attr = dataclasses.replace(
        ultra_cdf_attrs.ultra_support_attrs,
        catdesc="coin_type",
        fieldname="coin_type",
        label_axis="coin_type",
    ).output()

    assert cointype_list == decom_ultra_events["COIN_TYPE"][0:1]
    assert cointype_attr == expected_cointype_attr


def test_cdf_aux(
    ccsds_path,
    xtce_path,
    tmp_path,
    decom_ultra_aux,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    # TODO: change test filename with new naming convention
    test_data_path_aux = tmp_path / "ultra_l1a_aux_20210101-20210102_v01-01.cdf"
    assert not test_data_path_aux.exists()

    ultra_l1a({ULTRA_AUX.apid[0]: ccsds_path}, xtce_path, test_data_path_aux)
    assert test_data_path_aux.exists()

    dataset_aux = create_dataset({ULTRA_AUX.apid[0]: decom_ultra_aux})
    input_xarray_aux = cdf_to_xarray(test_data_path_aux)

    assert input_xarray_aux.attrs.keys() == dataset_aux.attrs.keys()


def test_cdf_rates(
    ccsds_path,
    xtce_path,
    tmp_path,
    decom_ultra_rates,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    # TODO: change test filename with new naming convention
    test_data_path_rates = tmp_path / "ultra_l1a_rates_20210101-20210102_v01-01.cdf"
    assert not test_data_path_rates.exists()

    ultra_l1a({ULTRA_RATES.apid[0]: ccsds_path}, xtce_path, test_data_path_rates)
    assert test_data_path_rates.exists()

    dataset_rates = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})
    input_xarray_rates = cdf_to_xarray(test_data_path_rates)

    assert input_xarray_rates.attrs.keys() == dataset_rates.attrs.keys()


def test_cdf_tof(
    ccsds_path_tof,
    xtce_path,
    tmp_path,
    decom_ultra_tof,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    # TODO: change test filename with new naming convention
    test_data_path_tof = tmp_path / "ultra_l1a_tof_20210101-20210102_v01-01.cdf"
    assert not test_data_path_tof.exists()

    ultra_l1a({ULTRA_TOF.apid[0]: ccsds_path_tof}, xtce_path, test_data_path_tof)
    assert test_data_path_tof.exists()

    dataset_tof = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})
    input_xarray_tof = cdf_to_xarray(test_data_path_tof)

    assert input_xarray_tof.attrs.keys() == dataset_tof.attrs.keys()


def test_cdf_events(
    ccsds_path,
    ccsds_path_events,
    xtce_path,
    tmp_path,
    decom_ultra_events,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    # TODO: change test filename with new naming convention
    test_data_path_events = tmp_path / "ultra_l1a_events_20210101-20210102_v01-01.cdf"
    assert not test_data_path_events.exists()

    ultra_l1a(
        {ULTRA_AUX.apid[0]: ccsds_path, ULTRA_EVENTS.apid[0]: ccsds_path_events},
        xtce_path,
        test_data_path_events,
    )
    assert test_data_path_events.exists()

    decom_ultra_aux = decom_ultra_apids(ccsds_path, xtce_path, ULTRA_AUX.apid[0])
    dataset_events = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )
    input_xarray_events = cdf_to_xarray(test_data_path_events)

    assert input_xarray_events.attrs.keys() == dataset_events.attrs.keys()
