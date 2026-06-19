"""Test ULTRA L1a CDFs."""

import pytest
import xarray as xr

from imap_processing.cdf.utils import write_cdf
from imap_processing.ultra.l0.decom_ultra import get_event_id
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_ENERGY_EVENTS,
    ULTRA_ENERGY_RATES,
    ULTRA_ENERGY_SPECTRA,
    ULTRA_EVENTS,
    ULTRA_EXTOF_HIGH_ANGULAR,
    ULTRA_EXTOF_HIGH_ENERGY,
    ULTRA_EXTOF_HIGH_TIME,
    ULTRA_MACROS_CHECKSUM,
    ULTRA_PHXTOF_HIGH_ANGULAR,
    ULTRA_PHXTOF_HIGH_ENERGY,
    ULTRA_PHXTOF_HIGH_TIME,
    ULTRA_PRI_1_EVENTS,
    ULTRA_PRI_2_EVENTS,
    ULTRA_PRI_3_EVENTS,
    ULTRA_PRI_4_EVENTS,
    ULTRA_RATES,
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


def test_xarray_phxtof_high_angular(ccsds_path_theta_0):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, apid_input=ULTRA_PHXTOF_HIGH_ANGULAR.apid[0]
    )

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


@pytest.mark.external_test_data
def test_xarray_phxtof_high_energy(ccsds_path_extra):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PHXTOF_HIGH_ENERGY.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


@pytest.mark.external_test_data
def test_xarray_phxtof_high_time(ccsds_path_extra):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PHXTOF_HIGH_TIME.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


@pytest.mark.external_test_data
def test_xarray_extof_high_angular(ccsds_path_extra):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_ANGULAR.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


@pytest.mark.external_test_data
def test_xarray_extof_high_energy(ccsds_path_extra):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_ENERGY.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
        "packetdata"
    ]
    assert (specific_epoch_data == test_data[0]["packetdata"][0][0]).all()


@pytest.mark.external_test_data
def test_xarray_extof_high_time(ccsds_path_extra):
    """This function checks that a xarray was
    successfully created from the decom_ultra_tof data."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_TIME.apid[0])

    # Spot check metadata data and attributes
    specific_epoch_data = test_data[0].sel(epoch=test_data[0].epoch[0], plane=0)[
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
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-aux_20240207-repoint99999_v999.cdf"
    )


def test_cdf_rates(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_RATES.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-rates_20240207-repoint99999_v999.cdf"
    )


def test_cdf_energy_rates(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_functional,
        apid_input=ULTRA_ENERGY_RATES.apid[0],
        create_derived_l1b=True,
    )
    assert len(test_data) == 2  # l1a and l1b

    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-energy-rates_20240122-repoint99999_v001.0001.cdf"
    )

    # L1b dataset
    assert "1B" in test_data[1].attrs["Data_type"]
    test_data[1].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[1], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1b_45sensor-energy-rates_20240122-repoint99999_v001.0001.cdf"
    )


def test_cdf_macrodump(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_functional, apid_input=871)
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-macrodump_20240122-repoint99999_v999.cdf"
    )


def test_cdf_memdump(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_functional, apid_input=868)
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-memdump_20240122-repoint99999_v999.cdf"
    )


def test_cdf_phxtof_high_angular(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_theta_0, apid_input=ULTRA_PHXTOF_HIGH_ANGULAR.apid[0]
    )
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-ang_"
        "20240207-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_phxtof_high_energy(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PHXTOF_HIGH_ENERGY.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-nrg_"
        "20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_phxtof_high_time(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PHXTOF_HIGH_TIME.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ena-phxtof-hi-time_"
        "20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_extof_high_angular(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_ANGULAR.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ena-extof-hi-ang_"
        "20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_extof_high_time(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_TIME.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ion-extof-hi-time_"
        "20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_extof_high_energy(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_EXTOF_HIGH_ENERGY.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)
    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_45sensor-histogram-ion-extof-hi-nrg_"
        "20240122-repoint99999_v999.cdf"
    )


def test_cdf_events(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=ULTRA_EVENTS.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-de_20240207-repoint99999_v999.cdf"
    )


def test_cdf_energy_events(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_functional, apid_input=ULTRA_ENERGY_EVENTS.apid[0])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-energy-de_20240122-repoint99999_v999.cdf"
    )


def test_cdf_macros_checksum(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(
        ccsds_path_functional, apid_input=ULTRA_MACROS_CHECKSUM.apid[0]
    )
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-macroschecksum_20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_pri_1_events(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PRI_1_EVENTS.apid[0])

    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-priority-1-de_20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_pri_2_events(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PRI_2_EVENTS.apid[0])

    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-priority-2-de_20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_pri_3_events(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PRI_3_EVENTS.apid[0])

    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-priority-3-de_20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_pri_4_events(ccsds_path_extra):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_extra, apid_input=ULTRA_PRI_4_EVENTS.apid[0])

    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-priority-4-de_20240122-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_energy_spectra(ccsds_path_startup):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_startup, apid_input=ULTRA_ENERGY_SPECTRA.apid[1])
    test_data[0].attrs["Data_version"] = "999"
    test_data[0].attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(test_data[0], istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_90sensor-energy-spectra_20230711-repoint99999_v999.cdf"
    )


def test_cdf_hk(ccsds_path_theta_0):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_theta_0, apid_input=869)
    data = test_data[0]
    data.attrs["Data_version"] = "999"
    data.attrs["Repointing"] = "repoint99999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name
        == "imap_ultra_l1a_45sensor-status_20240207-repoint99999_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_cmdtxt(ccsds_path_all_apids):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_all_apids, apid_input=939)
    data = test_data[0]
    data.attrs["Data_version"] = "999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_90sensor-cmdtext_20260924_v999.cdf"


def test_cdf_cmdecho(ccsds_path_functional):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_functional, apid_input=865)
    data = test_data[0]
    data.attrs["Data_version"] = "999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_45sensor-cmdecho_20240122_v999.cdf"


@pytest.mark.external_test_data
def test_cdf_monitorlimits(ccsds_path_all_apids):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_all_apids, apid_input=937)
    data = test_data[0]
    data.attrs["Data_version"] = "999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert (
        test_data_path.name == "imap_ultra_l1a_90sensor-monitorlimits_20260924_v999.cdf"
    )


@pytest.mark.external_test_data
def test_cdf_startup(ccsds_path_all_apids):
    """Tests that CDF file can be created."""
    test_data = ultra_l1a(ccsds_path_all_apids, apid_input=941)
    data = test_data[0]
    data.attrs["Data_version"] = "999"
    test_data_path = write_cdf(data, istp=True)

    assert test_data_path.exists()
    assert test_data_path.name == "imap_ultra_l1a_90sensor-imgparams_20260924_v999.cdf"


def test_get_event_id():
    """Test get_event_id"""
    # example event bytes
    event_data = (
        b"\x929\xc4=\x05\x13\xf2dC\x0c`\x002\xb2\xb3\x80\nUQ\xb5BH"
        b'\xe6\x114\x10O\t\xb1\x08\x0e`\x00\xd6\x89"\x00)UF\xd6I'
    )
    met = 445015657
    count = 2
    event_ids = get_event_id(event_data, count, met, 166)
    assert len(event_ids) == count
    # Check that they are all unique
    assert len(set(event_ids)) == count
    # Check that they are all strings of 50 characters
    assert all(
        isinstance(event_id, str) and len(event_id) == 50 for event_id in event_ids
    )
