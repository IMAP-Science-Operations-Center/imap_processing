import numpy as np
import pytest

from imap_processing.cdf.utils import write_cdf
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import (
    get_event_id,
    ultra_l1a,
)
from imap_processing.spice.time import TTJ2000_EPOCH


def test_xarray_aux(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_aux data."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_AUX.apid[0]
    )

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0])["spinperiodvalid"]

    assert (specific_epoch_data == test_data[0]["spinperiodvalid"][0]).all()


def test_xarray_rates(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_rates data."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_RATES.apid[0]
    )
    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0])["start_rf"]
    assert (specific_epoch_data == test_data[0]["start_rf"][0]).all()


def test_xarray_tof(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_TOF.apid[0]
    )

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], sid=0)["PACKETDATA"]
    assert (specific_epoch_data == test_data[0]["PACKETDATA"][0][0]).all()


def test_xarray_events(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_events data."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_EVENTS.apid[0]
    )
    # TODO: add epoch and eventid as coordinates/indices
    # Spot check metadata data and attributes
    j2000_time = (
        np.datetime64("2024-02-07T15:28:37.184000", "ns") - TTJ2000_EPOCH
    ).astype(np.int64)
    specific_epoch_data = test_data[0].sel(epoch=760591717184000000)["COIN_TYPE"]
    cointype_list = specific_epoch_data.values.tolist()
    cointype_attr = test_data.variables["COIN_TYPE"].attrs

    assert cointype_list == test_data["COIN_TYPE"][0]
    assert cointype_attr == expected_cointype_attr


def test_cdf_aux(ccsds_path_theta_0):
    """Tests that CDF file can be created."""

    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_AUX.apid[0]
    )
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-aux_20240207_v001.cdf"


def test_cdf_rates(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_RATES.apid[0]
    )
    test_data_path = write_cdf(test_data[0], istp=False)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-rates_20240207_v001.cdf"


def test_cdf_tof(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_TOF.apid[0]
    )
    test_data_path = write_cdf(test_data[0])
    # TODO: why is this the time?
    # TODO: add depends on information
    # TODO: add event id information
    # TODO: improve these tests
    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-ang_20000101_v001.cdf"


def test_cdf_events(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, data_version="001", apid=ULTRA_EVENTS.apid[0]
    )
    test_data_path = write_cdf(test_data[0], istp=False)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-de_20240207_v001.cdf"


def test_get_event_id():
    """Test get_event_id"""
    decom_ultra_dict = {
        ULTRA_EVENTS.apid[0]: {"SHCOARSE": [445015662, 445015663, 445015664, 445015664]}
    }
    decom_events = get_event_id(decom_ultra_dict)
    counters_for_met = []
    for i in range(len(decom_events["EVENTID"])):
        event_id = decom_events["EVENTID"][i]
        met_extracted = event_id >> np.int64(31)

        assert met_extracted == np.int64(
            decom_ultra_dict[ULTRA_EVENTS.apid[0]]["SHCOARSE"][i]
        )
        counters_for_met.append(event_id & np.int64(0x7FFFFFFF))

    assert counters_for_met == [0, 0, 0, 1]
