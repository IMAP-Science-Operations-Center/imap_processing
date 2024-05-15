import dataclasses
import pytest

from imap_processing import decom
from imap_processing.cdf.utils import load_cdf
from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import decom_ultra_apids
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import create_dataset, ultra_l1a
from imap_processing.utils import group_by_apid


@pytest.fixture()
def decom_ultra_aux(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_aux"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_AUX.apid[0]: grouped_data[ULTRA_AUX.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_AUX.apid[0])
    return data_packet_list


@pytest.fixture()
def decom_ultra_rates(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_rates"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_RATES.apid[0]: grouped_data[ULTRA_RATES.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_RATES.apid[0])
    return data_packet_list


@pytest.fixture()
def decom_ultra_events(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_events"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_EVENTS.apid[0]: grouped_data[ULTRA_EVENTS.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_EVENTS.apid[0])
    return data_packet_list

@pytest.fixture()
def decom_ultra_tof(ccsds_path_tof, xtce_path):
    """Data for decom_ultra_tof"""
    data_packet_list = decom_ultra_apids(ccsds_path_tof, xtce_path, ULTRA_TOF.apid[0])
    return data_packet_list

@pytest.fixture()
def decom_ultra_tof(ccsds_path_tof, xtce_path):
    """Data for decom_ultra_tof"""
    packets = decom.decom_packets(ccsds_path_tof, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_TOF.apid[0]: grouped_data[ULTRA_TOF.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_TOF.apid[0])
    return data_packet_list


@pytest.fixture()
def decom_ultra_tof_theta_00(ccsds_path_theta_0, xtce_path):
    """Data for decom_ultra_tof"""
    packets = decom.decom_packets(ccsds_path_theta_0, xtce_path)
    grouped_data = group_by_apid(packets)
    data = {ULTRA_TOF.apid[0]: grouped_data[ULTRA_TOF.apid[0]]}

    data_packet_list = decom_ultra_apids(data, ULTRA_TOF.apid[0])
    return data_packet_list


def test_xarray_aux(decom_ultra_aux):
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


def test_xarray_rates(decom_ultra_rates):
    """This function checks that a xarray was
    successfully created from the decom_ultra_rates data."""

    dataset = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2024-02-07T15:28:37.184000")["START_RF"]
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


def test_xarray_tof_theta_00(decom_ultra_tof):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    dataset = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2024-02-07T15:28:37.184000", sid=0)[
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

    dataset = create_dataset(
        {
            ULTRA_EVENTS.apid[0]: decom_ultra_events,
            ULTRA_AUX.apid[0]: decom_ultra_aux,
        }
    )

    # Spot check metadata data and attributes
    specific_epoch_data = dataset.sel(epoch="2024-02-07T15:28:37.184000")["COIN_TYPE"]
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


# def test_xarray_events(
#     decom_ultra_events, decom_ultra_aux, events_fsw_comparison_theta_0
# ):
#     """This function checks that a xarray was
#     successfully created from the decom_ultra_events data."""
#
#     dataset = create_dataset(
#         {
#             ULTRA_EVENTS.apid[0]: decom_ultra_events,
#             ULTRA_AUX.apid[0]: decom_ultra_aux,
#         }
#     )
#
#     mask = dataset.COIN_TYPE.data != GlobalConstants.INT_FILLVAL
#
#     df = pd.read_csv(events_fsw_comparison_theta_0)
#
#     np.testing.assert_array_equal(df["CoinType"], dataset.COIN_TYPE.data[mask])
#     np.testing.assert_array_equal(df["StartType"], dataset.START_TYPE.data[mask])
#     np.testing.assert_array_equal(df["StopType"], dataset.STOP_TYPE.data[mask])
#     np.testing.assert_array_equal(df["StartPosTDC"], dataset.START_POS_TDC.data[mask])
#     np.testing.assert_array_equal(df["StopNorthTDC"], dataset.STOP_NORTH_TDC.data[mask])
#     np.testing.assert_array_equal(df["StopEastTDC"], dataset.STOP_EAST_TDC.data[mask])
#     np.testing.assert_array_equal(df["StopSouthTDC"], dataset.STOP_SOUTH_TDC.data[mask])
#     np.testing.assert_array_equal(df["StopWestTDC"], dataset.STOP_WEST_TDC.data[mask])
#     np.testing.assert_array_equal(df["CoinNorthTDC"], dataset.COIN_NORTH_TDC.data[mask])
#     np.testing.assert_array_equal(df["CoinSouthTDC"], dataset.COIN_SOUTH_TDC.data[mask])
#     np.testing.assert_array_equal(df["CoinDiscrete"], dataset.COIN_DISCRETE_TDC.data[mask])
#     np.testing.assert_array_equal(df["EnergyPH"], dataset.ENERGY_PH.data[mask])
#     np.testing.assert_array_equal(df["PulseWidth"], dataset.PULSE_WIDTH.data[mask])
#     np.testing.assert_array_equal(df["Bin"], dataset.BIN.data[mask])
#     np.testing.assert_array_equal(df["CoinT"], dataset.EVENT_FLAG_CNT.data[mask])


def test_cdf_aux(
    ccsds_path_theta_0,
    decom_ultra_aux,
):
    """Tests that CDF file is created and contains same attributes as xarray."""

    test_data_path = ultra_l1a(ccsds_path_theta_0, apid=ULTRA_AUX.apid[0])
    assert test_data_path[0].exists()
    assert test_data_path[0].name == "imap_ultra_l1a_45aux_20240207_v001.cdf"

    dataset_aux = create_dataset({ULTRA_AUX.apid[0]: decom_ultra_aux})
    input_xarray_aux = load_cdf(test_data_path[0])

    assert input_xarray_aux.attrs.keys() == dataset_aux.attrs.keys()


def test_cdf_rates(
    ccsds_path,
    decom_ultra_rates,
):
    """Tests that CDF file is created and contains same attributes as xarray."""

    test_data_path = ultra_l1a(ccsds_path, apid=ULTRA_RATES.apid[0])
    assert test_data_path[0].exists()
    assert test_data_path[0].name == "imap_ultra_l1a_45rates_20220530_v001.cdf"

    dataset_rates = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})
    input_xarray_rates = load_cdf(test_data_path[0])

    assert input_xarray_rates.attrs.keys() == dataset_rates.attrs.keys()


def test_cdf_tof(
    ccsds_path_tof,
    decom_ultra_tof,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    test_data_path = ultra_l1a(ccsds_path_tof, apid=ULTRA_TOF.apid[0])
    assert test_data_path[0].exists()
    assert test_data_path[0].name == "imap_ultra_l1a_45tof_20240124_v001.cdf"

    dataset_tof = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})
    input_xarray_tof = load_cdf(test_data_path[0])

    assert input_xarray_tof.attrs.keys() == dataset_tof.attrs.keys()


def test_cdf_events(
    ccsds_path_theta_0,
    decom_ultra_aux,
    decom_ultra_events,
):
    """Tests that CDF file is created and contains same attributes as xarray."""
    test_data_path = ultra_l1a(ccsds_path_theta_0, apid=ULTRA_EVENTS.apid[0])
    assert test_data_path[0].exists()
    assert test_data_path[0].name == "imap_ultra_l1a_45events_20240207_v001.cdf"

    dataset_events = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )
    input_xarray_events = load_cdf(test_data_path[0])

    assert input_xarray_events.attrs.keys() == dataset_events.attrs.keys()
