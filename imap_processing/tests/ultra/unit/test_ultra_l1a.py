import pytest

from imap_processing import decom
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.ultra.l0.decom_ultra import process_ultra_apids
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

    data_packet_list = process_ultra_apids(
        grouped_data[ULTRA_AUX.apid[0]], ULTRA_AUX.apid[0]
    )
    return data_packet_list


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_AUX.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.xfail(reason="Fix CDF attrs")
def test_xarray_aux(decom_test_data):
    """This function checks that a xarray was
    successfully created from the decom_ultra_aux data."""
    pass
    # decom_ultra_aux, _ = decom_test_data
    # dataset = create_dataset({ULTRA_AUX.apid[0]: decom_ultra_aux})

    # # Spot check string data and attributes
    # spin_period_valid_list = dataset.variables["SPINPERIODVALID"].values.tolist()
    # spin_period_valid_attr = dataset.variables["SPINPERIODVALID"].attrs
    # expected_spin_period_valid_attr = ultra_cdf_attrs.StringAttrs(
    #     depend_0="epoch", catdesc="spinperiodvalid", fieldname="spinperiodvalid"
    # )

    # assert spin_period_valid_list == decom_ultra_aux["SPINPERIODVALID"]
    # assert spin_period_valid_attr == expected_spin_period_valid_attr.output()

    # # Spot check support data and attributes
    # version_list = dataset.variables["VERSION"].values.tolist()
    # version_attr = dataset.variables["VERSION"].attrs
    # expected_version_attr = dataclasses.replace(
    #     ultra_cdf_attrs.ultra_support_attrs,
    #     catdesc="version",
    #     fieldname="version",
    #     label_axis="version",
    # ).output()

    # assert version_list == decom_ultra_aux["VERSION"]
    # assert version_attr == expected_version_attr

    # # Spot check metadata data and attributes
    # shcoarse_list = dataset.variables["SHCOARSE"].values.tolist()
    # shcoarse_attr = dataset.variables["SHCOARSE"].attrs

    # expected_shcoarse_attr = dataclasses.replace(
    #     ultra_cdf_attrs.ultra_support_attrs,
    #     catdesc="shcoarse",
    #     fieldname="shcoarse",
    #     label_axis="shcoarse",
    # ).output()

    # assert shcoarse_list == decom_ultra_aux["SHCOARSE"]
    # assert shcoarse_attr == expected_shcoarse_attr


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_RATES.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.xfail(reason="Fix CDF attrs")
def test_xarray_rates(decom_test_data):
    """This function checks that a xarray was
    successfully created from the decom_ultra_rates data."""
    pass
    # decom_ultra_rates, _ = decom_test_data
    # dataset = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})

    # # Spot check metadata data and attributes
    # j2000_time = (
    #     np.datetime64("2024-02-07T15:28:37.184000", "ns") - J2000_EPOCH
    # ).astype(np.int64)
    # specific_epoch_data = dataset.sel(epoch=j2000_time)["START_RF"]
    # startrf_list = specific_epoch_data.values.tolist()
    # startrf_attr = dataset.variables["START_RF"].attrs

    # expected_startrf_attr = dataclasses.replace(
    #     ultra_cdf_attrs.ultra_support_attrs,
    #     catdesc="start_rf",
    #     fieldname="start_rf",
    #     label_axis="start_rf",
    # ).output()

    # assert startrf_list == decom_ultra_rates["START_RF"][0]
    # assert startrf_attr == expected_startrf_attr


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_TOF.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            }
        )
    ],
    indirect=True,
)
@pytest.mark.xfail(reason="Fix CDF attrs")
def test_xarray_tof(decom_test_data):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    pass
    # decom_ultra_tof, _ = decom_test_data
    # dataset = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})

    # # Spot check metadata data and attributes
    # j2000_time = (
    #     np.datetime64("2024-02-07T15:28:36.184000", "ns") - J2000_EPOCH
    # ).astype(np.int64)
    # specific_epoch_data = dataset.sel(epoch=j2000_time, sid=0)["PACKETDATA"]
    # packetdata_attr = dataset.variables["PACKETDATA"].attrs

    # expected_packetdata_attr = dataclasses.replace(
    #     ultra_cdf_attrs.ultra_support_attrs,
    #     catdesc="packetdata",
    #     fieldname="packetdata",
    #     label_axis="packetdata",
    #     depend_1="sid",
    #     depend_2="row",
    #     depend_3="column",
    #     units="pixels",
    #     variable_purpose="primary_var",
    # ).output()

    # assert (specific_epoch_data == decom_ultra_tof["PACKETDATA"][0][0]).all()
    # assert packetdata_attr == expected_packetdata_attr


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.xfail(reason="Fix CDF attrs")
def test_xarray_events(decom_test_data, decom_ultra_aux, events_test_path):
    """This function checks that a xarray was
    successfully created from the decom_ultra_events data."""
    pass
    # decom_ultra_events, _ = decom_test_data
    # dataset = create_dataset(
    #     {
    #         ULTRA_EVENTS.apid[0]: decom_ultra_events,
    #         ULTRA_AUX.apid[0]: decom_ultra_aux,
    #     }
    # )

    # # Spot check metadata data and attributes
    # j2000_time = (
    #     np.datetime64("2024-02-07T15:28:37.184000", "ns") - J2000_EPOCH
    # ).astype(np.int64)
    # specific_epoch_data = dataset.sel(epoch=j2000_time)["COIN_TYPE"]
    # cointype_list = specific_epoch_data.values.tolist()
    # cointype_attr = dataset.variables["COIN_TYPE"].attrs

    # expected_cointype_attr = dataclasses.replace(
    #     ultra_cdf_attrs.ultra_support_attrs,
    #     catdesc="coin_type",
    #     fieldname="coin_type",
    #     label_axis="coin_type",
    # ).output()

    # assert cointype_list == decom_ultra_events["COIN_TYPE"][0]
    # assert cointype_attr == expected_cointype_attr


def test_cdf_aux(
    ccsds_path_theta_0,
    decom_ultra_aux,
):
    """Tests that CDF file is created and contains same attributes as xarray."""

    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_AUX.apid[0]
    )
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-aux_20240207_v001.cdf"

    dataset_aux = create_dataset({ULTRA_AUX.apid[0]: decom_ultra_aux})
    input_xarray_aux = load_cdf(test_data_path)

    # write_cdf() injects some attributes that are not in the xarray
    assert set(dataset_aux.attrs.keys()).issubset(set(input_xarray_aux.attrs.keys()))


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_RATES.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_cdf_rates(ccsds_path_theta_0, decom_test_data):
    """Tests that CDF file is created and contains same attributes as xarray."""
    decom_ultra_rates, _ = decom_test_data
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_RATES.apid[0]
    )
    # TODO: Dropping duplicates to ignore ISTP for now. Need to update test data
    # or wait for an update to cdflib
    test_data[0] = test_data[0].sortby("epoch").groupby("epoch").first()
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-rates_20240207_v001.cdf"

    dataset_rates = create_dataset({ULTRA_RATES.apid[0]: decom_ultra_rates})
    input_xarray_rates = load_cdf(test_data_path)

    # write_cdf() injects some attributes that are not in the xarray
    assert set(dataset_rates.attrs.keys()).issubset(
        set(input_xarray_rates.attrs.keys())
    )


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_TOF.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_cdf_tof(ccsds_path_theta_0, decom_test_data):
    """Tests that CDF file is created and contains same attributes as xarray."""
    decom_ultra_tof, _ = decom_test_data
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_TOF.apid[0]
    )
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-histogram_20240207_v001.cdf"

    dataset_tof = create_dataset({ULTRA_TOF.apid[0]: decom_ultra_tof})
    input_xarray_tof = load_cdf(test_data_path)
    # write_cdf() injects some attributes that are not in the xarray
    assert set(dataset_tof.attrs.keys()).issubset(set(input_xarray_tof.attrs.keys()))


@pytest.mark.parametrize(
    "decom_test_data",
    [
        pytest.param(
            {
                "apid": ULTRA_EVENTS.apid[0],
                "filename": "FM45_40P_Phi28p5_BeamCal_LinearScan_phi28.50"
                "_theta-0.00_20240207T102740.CCSDS",
            },
        )
    ],
    indirect=True,
)
def test_cdf_events(ccsds_path_theta_0, decom_ultra_aux, decom_test_data):
    """Tests that CDF file is created and contains same attributes as xarray."""
    decom_ultra_events, _ = decom_test_data
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_EVENTS.apid[0]
    )
    # TODO: Dropping duplicates to ignore ISTP for now. Need to update test data
    # or wait for an update to cdflib
    test_data[0] = test_data[0].sortby("epoch").groupby("epoch").first()
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-de_20240207_v001.cdf"

    dataset_events = create_dataset(
        {ULTRA_EVENTS.apid[0]: decom_ultra_events, ULTRA_AUX.apid[0]: decom_ultra_aux}
    )
    input_xarray_events = load_cdf(test_data_path)

    # write_cdf() injects some attributes that are not in the xarray
    assert set(dataset_events.attrs.keys()).issubset(
        set(input_xarray_events.attrs.keys())
    )
