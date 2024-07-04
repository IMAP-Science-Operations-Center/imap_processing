"""
Contains common attribute classes to use as a base for CDF files.

All the classes with "Global" in their name are intended for use for global attributes
in CDF files. The rest are attributes for individual data fields within the CDF file.

The attributes classes generally contain intelligent defaults, but currently do not
check if all the attributes are valid of if all the required attributes are present.
This check occurs during the xarray_to_cdf step, which has an example in
imap_processing/cdf/utils.py.

References
----------
For more information on attributes, refer to the SPDF documentation at:
https://spdf.gsfc.nasa.gov/istp_guide/vattributes.html

Examples
--------
Additional examples on how to use these dataclasses are in
imap_processing/idex/idex_cdf_attrs.py and imap_processing/idex/idex_packet_parser.py.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Optional, Union

import numpy as np

from imap_processing.cdf.defaults import GlobalConstants


class GlobalConstantAttrs:
    """
    Global base of attributes for all CDF files.

    This is automatically included in InstrumentBase.

    Attributes
    ----------
    GLOBAL_BASE:
        Global file attributes, including project, source_name, discipline, PI_name,
        PI_affiliation, and mission_group. This should be the same
        for all instruments.
    """

    # TODO: ask about how to add optional parameter in File_naming_convention.
    # Need optional parameter for date range and repoint number.
    # If File_naming_convention was not set, it uses default setting:
    #   source_datatype_descriptor_yyyyMMdd
    # which would result in a file name like:
    #   imap_l1_sci_idex_20220101_v001.cdf
    # For IMAP naming convention, it should be:
    #   source_descriptor_datatype_yyyyMMdd_vNNN
    # which would result in a file name like:
    #   imap_idex_l1_sci_20220101_v001.cdf
    GLOBAL_BASE: Final[dict] = {
        "Project": "STP>Solar-Terrestrial Physics",
        "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
        "Discipline": "Solar Physics>Heliospheric Physics",
        # TODO: CDF docs say this value should be "IMAP"
        "Mission_group": "IMAP>Interstellar Mapping and Acceleration Probe",
        "PI_name": "Dr. David J. McComas",
        "PI_affiliation": (
            "Princeton Plasma Physics Laboratory",
            "100 Stellarator Road, Princeton, NJ 08540",
        ),
        "File_naming_convention": "source_descriptor_datatype_yyyyMMdd_vNNN",
    }

    def output(self) -> dict:
        """
        Generate the output for the global attributes as a dictionary.

        This returns the contents of the GLOBAL_BASE attribute.

        Returns
        -------
        dict
            Global base attributes.
        """
        return self.GLOBAL_BASE


class ConstantCoordinates:
    """
    Return a dictionary with global base attributes.

    Attributes
    ----------
    EPOCH:
        Default values for "epoch" coordinate value.
    """

    EPOCH: ClassVar[dict] = {
        # By default the time is assumed to correspond to instantaneous time or
        # center of accumulation/measurement period. But if, for some good reason,
        # time corresponds to the beginning or end or any other part of
        # accumulation/measurement period, that has to be stated in CATDESC.
        "CATDESC": "Time, number of nanoseconds since J2000 with leap seconds included",
        "FIELDNAM": "epoch",
        "FILLVAL": GlobalConstants.INT_FILLVAL,
        "LABLAXIS": "epoch",
        "FORMAT": "",  # Supposedly not required, fails in xarray_to_cdf
        "UNITS": "ns",
        "VALIDMIN": GlobalConstants.MIN_EPOCH,
        "VALIDMAX": GlobalConstants.MAX_EPOCH,
        "VAR_TYPE": "support_data",
        "SCALETYP": "linear",
        "MONOTON": "INCREASE",
        "TIME_BASE": "J2000",
        "TIME_SCALE": "Terrestrial Time",
        "REFERENCE_POSITION": "Rotating Earth Geoid",
    }


@dataclass
class GlobalInstrumentAttrs:
    """
    Each instrument should extend this class and replace the info as needed.

    Attributes
    ----------
    version : str
        The software version.
    descriptor : str
        Descriptor of the instrument (Ex: "IDEX>Interstellar Dust Experiment").
        NOTE:
        Instrument name on the left side of the ">" will need to match what it
        appears in the filename.
    text : str
        Explanation of the instrument, usually as a paragraph.
    instrument_type : str default="Particles (space)"
        This attribute is used to facilitate making choices of instrument type. More
        than one entry is allowed. Valid IMAP values include:
        [Electric Fields (space), Magnetic Fields (space), Particles (space),
        Plasma and Solar Wind, Ephemeris].
    """

    version: str
    descriptor: str
    text: str
    instrument_type: str = "Particles (space)"

    def output(self) -> dict:
        """
        Generate the output for the instrument as a dictionary.

        Returns
        -------
        dict
            Dictionary of correctly formatted values for the data_version, descriptor,
            text, and logical_file_id, added to the global attributes from GlobalBase.
        """
        return GlobalConstantAttrs().output() | {
            "Data_version": self.version,
            "Descriptor": self.descriptor,
            "TEXT": self.text,
            "Instrument_type": self.instrument_type,
        }


@dataclass
class GlobalDataLevelAttrs:
    """
    Class for all the attributes for the data level base.

    This is used to make the attributes for each data level, and includes
    InstrumentBase for the required instrument attributes and global attributes.

    Attributes
    ----------
    data_type : str
        The data level and descriptor separated by underscore.
        Eg. "L1A_DE>Level-1A Direct Event"
    logical_source : str
        The source of the data, ex "imap_idex_l1_sci"
    logical_source_desc : str
        The description of the data, ex "IMAP Mission IDEX Instrument Level-1 Data."
    instrument_base : GlobalInstrumentAttrs
        The InstrumentBase object describing the basic instrument information.
    """

    data_type: str
    logical_source: str
    logical_source_desc: str
    instrument_base: GlobalInstrumentAttrs

    def output(self) -> dict:
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        dict
            Dictionary of correctly formatted values for the attributes in the class and
            the attributes from InstrumentBase.
        """
        return self.instrument_base.output() | {
            # TODO: rework cdf_utils.write_cdf to leverage dataclasses
            "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
            "Data_type": self.data_type,
            "Logical_source": self.logical_source,
            "Logical_source_description": self.logical_source_desc,
        }


@dataclass
class AttrBase:
    """
    The general class for attributes, with some reasonable defaults.

    Attributes
    ----------
    validmin : np.float64 | np.int64
        The valid minimum value, required.
    validmax : np.float64 | np.int64
        The valid maximum value, required.
    display_type : str default="no_plot"
        The display type of the plot (ex "no_plot"), required.
    catdesc : str, default=None
        The category description, "CATDESC" attribute, required.
        Max 80 characters. If need to write more, use "var_notes".
    fieldname : str, default=None
        The fieldname, "FIELDNAM" attribute. Max 30 characters.
        fieldname and label_axis parameter are closely related.
        fieldname and label_axis are used in plot. fieldname
        is used as title of the plot, label_axis is used as axis label.
        For example, fieldname='Time of flight', label_axis='TOF'
    var_type : str, default="support_data"
        The type of data. Valid options are:
            1. "data" - plotted and listed on CDAWeb and accessed by API,
            2. "support_data" - not plotted
            3. "metadata" - reserved for character type variables.
    fill_val : np.int64, default=Constants.INT_FILLVAL
        The values for filling data
    scale_type : str, default="linear"
        The scale of the axis, "SCALETYP" attribute.
    label_axis : str, default=None
        Axis label, "LABLAXIS" attribute. Required. Should be close to 6 letters.
        See fieldname parameter for more information.
    format : str, default=None
        The format of the data, in Fortran format.
    units : str, default=None
        The units of the data.
    """

    validmin: Union[np.float64, np.int64]
    validmax: Union[np.float64, np.int64]
    display_type: str = "no_plot"
    catdesc: Optional[str] = None
    fieldname: Optional[str] = None
    var_type: str = "support_data"
    fill_val: np.int64 = GlobalConstants.INT_FILLVAL
    scale_type: str = "linear"
    label_axis: Optional[str] = None
    format: Optional[str] = None
    units: str = ""

    def output(self) -> dict:
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        dict
            Dictionary of correctly formatted values for the attributes in the class.
        """
        return {
            "CATDESC": self.catdesc,
            "DISPLAY_TYPE": self.display_type,
            "FIELDNAM": self.fieldname,
            "FILLVAL": self.fill_val,
            "FORMAT": self.format,
            "LABLAXIS": self.label_axis,
            "UNITS": self.units,
            "VALIDMIN": self.validmin,
            "VALIDMAX": self.validmax,
            "VAR_TYPE": self.var_type,
            "SCALETYP": self.scale_type,
        }


@dataclass
class ScienceAttrs(AttrBase):
    """
    The class for Science attributes, in particular with depend_0 attributes.

    It also contains all the attributes and defaults from the generic TypeBase as well.

    Attributes
    ----------
    depend_0 : str = None
        The first degree of dependent coordinate variables.
        Although this is an optional keyword, it is required for every instance.
        This should be the "epoch" dimension, and should be type datetime64.
    depend_1 : str = None, optional
        The second degree of dependent coordinate variables. This is used for 2d data.
    depend_2 : str = None, optional
        The third degree of dependent coordinate variables. This is used for 3d data.
        If this variable is used, there must also be a depend_1 value.
    variable_purpose : str = None, optional
        The variable purpose attribute tells which variables are worth plotting.
    var_notes : str = None, optional
        Notes on the variable.
    """

    depend_0: Optional[str] = None
    depend_1: Optional[str] = None
    depend_2: Optional[str] = None
    depend_3: Optional[str] = None
    variable_purpose: Optional[str] = None
    var_notes: Optional[str] = None
    labl_ptr: Optional[str] = None

    def __post_init__(self) -> None:
        """If depend_0 is not set, raise an error, as this attribute is required."""
        if self.depend_0 is None:
            raise TypeError("ScienceBase requires depend_0 attribute.")

    def output(self) -> Any:
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        dict
            Dictionary of correctly formatted values for the attributes in the class.
            If the optional parameters are not defined, they are not included as
            attributes in the output.
        """
        endval = {"DEPEND_0": self.depend_0}
        if self.depend_1 is not None:
            endval["DEPEND_1"] = self.depend_1

        if self.depend_2 is not None:
            endval["DEPEND_2"] = self.depend_2

        if self.depend_3 is not None:
            endval["DEPEND_3"] = self.depend_3

        if self.variable_purpose is not None:
            endval["VARIABLE_PURPOSE"] = self.variable_purpose

        if self.var_notes is not None:
            endval["VAR_NOTES"] = self.var_notes

        if self.labl_ptr is not None:
            endval["LABL_PTR_1"] = self.labl_ptr

        return super().output() | endval


@dataclass
class FloatAttrs(ScienceAttrs):
    """
    The float version of ScienceBase with defaults to use for float-based data.

    Attributes
    ----------
    format : str, default="F64.5"
        The format of the data, in Fortran format
    fill_val : np.float64, default=Constants.DOUBLE_FILLVAL
        The values for filling data.
    units : str, default="float"
        The units of the data.
    """

    format: str = "F64.5"
    fill_val: np.float64 = GlobalConstants.DOUBLE_FILLVAL


@dataclass
class StringAttrs:
    """
    A base for String-based data, with string based defaults.

    This class does not have the same required attributes as ScienceBase, as it is a
    standalone class that doesn't extend any other class.

    Attributes
    ----------
    depend_0 : str
        The first degree of dependent coordinate variables.
    catdesc : str, default=None
        The category description, "CATDESC" attribute, required.
    fieldname : str, default=None
        The fieldname, "FIELDNAM" attribute.
    format : str, default="A80"
        The format of the data, in Fortran format.
    var_type : str, default="metadata"
        The type of data.
    display_type : str, default="no_plot"
        The display type of the plot.
    """

    depend_0: str
    catdesc: Optional[str] = None
    fieldname: Optional[str] = None
    format: str = "A80"
    var_type: str = "metadata"
    display_type: str = "no_plot"

    def output(self) -> dict:
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        dict
            Dictionary of correctly formatted values for the attributes in the class.
        """
        return {
            "CATDESC": self.catdesc,
            "DEPEND_0": self.depend_0,
            "FORMAT": self.format,
            "DISPLAY_TYPE": self.display_type,
            "FIELDNAM": self.fieldname,
            "VAR_TYPE": self.var_type,
        }
