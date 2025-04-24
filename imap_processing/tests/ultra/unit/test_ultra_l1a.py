"""Test ULTRA L1a CDFs."""

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ultra.l0.decom_ultra import get_event_id
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.ultra.l1a.ultra_l1a import (
    ultra_l1a,
)


def test_xarray_aux(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_aux data."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_AUX.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0])[
        "spinperiodvalid"
    ]

    assert (specific_epoch_data == test_data[0]["spinperiodvalid"][0]).all()


def test_xarray_rates(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_rates data."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_RATES.apid[0])
    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0])["start_rf"]
    assert (specific_epoch_data == test_data[0]["start_rf"][0]).all()


def test_xarray_tof(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_TOF.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], sid=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


def test_xarray_events(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_events data."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_EVENTS.apid[0])
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0])["coin_type"]
    assert (specific_epoch_data == test_data[0]["coin_type"][0]).all()


def test_xarray_hk(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_hk data."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=869)

    assert isinstance(test_data[0], xr.Dataset)


def test_cdf_aux(ccsds_path_theta_0):
    """Tests that CDF file can be created."""

    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_AUX.apid[0])
    test_data_path = write_cdf(test_data[0])

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-aux_20240207_v999.cdf"


def test_cdf_rates(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_RATES.apid[0])
    test_data_path = write_cdf(test_data[0], istp=False)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-rates_20240207_v999.cdf"


def test_cdf_tof(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_TOF.apid[0])
    test_data_path = write_cdf(test_data[0])
    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-ang_20240207_v999.cdf"
    )


def test_cdf_events(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_EVENTS.apid[0])
    test_data_path = write_cdf(test_data[0], istp=False)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-de_20240207_v999.cdf"


def test_cdf_hk(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=869)
    data = test_data[0]
    data.attrs["Data_version"] = "v999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-status_20240207_v999.cdf"


def test_get_event_id():
    """Test get_event_id"""
    data = np.array([445015662, 445015663, 445015664, 445015664])
    decom_events = get_event_id(data)
    counters_for_met = []
    for i in range(len(decom_events)):
        event_id = decom_events[i]
        met_extracted = event_id >> np.int64(31)

        assert met_extracted == np.int64(data[i])
        counters_for_met.append(event_id & np.int64(0x7FFFFFFF))

    assert counters_for_met == [0, 0, 0, 1]
