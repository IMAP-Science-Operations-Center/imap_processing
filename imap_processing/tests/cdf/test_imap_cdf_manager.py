from pathlib import Path

# from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes


def test_add_instrument_global_attrs():
    # Create an ImapCdfAttributes object, set to correct file path
    imap_cdf_manager = ImapCdfAttributes()
    imap_cdf_manager.source_dir = Path(__file__).parent.parent / "cdf" / "test_data"
    imap_cdf_manager.add_instrument_global_attrs("instrument1")

    # Testing data loaded in
    imap_instrument = imap_cdf_manager.get_global_attributes("imap_test_T1_test")
    assert imap_instrument["Data_type"] == "T1_test-one>Test-1 test one"
    assert imap_instrument["Project"] == "STP>Solar Terrestrial Probes"

    # Testing reloading data
    imap_cdf_manager.add_instrument_global_attrs("instrument2")

    # Testing data carried over, and overwritten
    instrument2_instrument = imap_cdf_manager.get_global_attributes("imap_swe_l1a_sci")
    assert instrument2_instrument["Data_type"] == "L1A_SCI>Level-1A Science data"
    assert instrument2_instrument["Project"] == "STP>Solar Terrestrial Probes"


def test_add_instrument_variable_attrs():
    # Create an ImapCdfAttributes object
    imap_cdf_manager = ImapCdfAttributes()
    imap_cdf_manager.source_dir = Path(__file__).parent.parent / "cdf" / "test_data"
    imap_cdf_manager.add_instrument_variable_attrs("instrument1", "level1")

    # Testing the actual function
    imap_instrument = imap_cdf_manager.get_variable_attributes("test_field_1")
    assert imap_instrument["DEPEND_0"] == "test_depend"
    assert imap_instrument["CATDESC"] == "test time"
    assert imap_instrument["FIELDNAM"] == "test_field_1"
    assert imap_instrument["UNITS"] == "test_units"

    # Testing reloading data
    imap_cdf_manager.add_instrument_variable_attrs("instrument2", "level2")

    # Testing again
    instrument2_instrument = imap_cdf_manager.get_variable_attributes("epoch")
    assert instrument2_instrument["DEPEND_0"] == "epoch"
    assert instrument2_instrument["DISPLAY_TYPE"] == "time_series"
    assert instrument2_instrument["VALIDMAX"] == 9223372036854775807
    assert (
        instrument2_instrument["CATDESC"]
        == "Time, number of nanoseconds since J2000 with leap seconds included"
    )
    assert instrument2_instrument["UNITS"] == "ns"
