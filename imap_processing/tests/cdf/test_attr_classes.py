import numpy as np
import pytest

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    GlobalConstantAttrs,
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
    StringAttrs,
)


def test_global_attrs():
    inst = GlobalInstrumentAttrs(
        version="1",
        descriptor="2",
        text="3",
        instrument_type="4",
    )

    expected = GlobalConstantAttrs().output() | {
        "PI_name": ("Dr. David J. McComas"),
        "PI_affiliation": (
            "Princeton Plasma Physics Laboratory",
            "100 Stellarator Road, Princeton, NJ 08540",
        ),
        "Data_version": "1",
        "Descriptor": "2",
        "TEXT": "3",
        "Instrument_type": "4",
    }

    assert inst.output() == expected

    data_level = GlobalDataLevelAttrs(
        data_type="1", logical_source="2", logical_source_desc="3", instrument_base=inst
    )
    expected_level = inst.output() | {
        "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
        "Data_type": "1",
        "Logical_source": "2",
        "Logical_source_description": "3",
    }
    assert data_level.output() == expected_level


def test_science_attr():
    with pytest.raises(TypeError):
        ScienceAttrs(np.int64(0), np.int64(1))

    science_attr = ScienceAttrs(
        np.int64(0), np.int64(1), depend_0="0", fieldname="1", variable_purpose="2"
    )

    expected = {
        "CATDESC": None,
        "DISPLAY_TYPE": None,
        "FIELDNAM": "1",
        "FILLVAL": GlobalConstants.INT_FILLVAL,
        "FORMAT": None,
        "LABLAXIS": None,
        "UNITS": "",
        "VALIDMIN": np.int64(0),
        "VALIDMAX": np.int64(1),
        "VAR_TYPE": "support_data",
        "SCALETYP": "linear",
        "DEPEND_0": "0",
        "VARIABLE_PURPOSE": "2",
    }

    assert science_attr.output() == expected


def test_string_attrs():
    str_attrs = StringAttrs("0")

    expected = {
        "CATDESC": None,
        "DEPEND_0": "0",
        "FORMAT": "A80",
        "DISPLAY_TYPE": "no_plot",
        "FIELDNAM": None,
        "VAR_TYPE": "metadata",
    }

    assert str_attrs.output() == expected
