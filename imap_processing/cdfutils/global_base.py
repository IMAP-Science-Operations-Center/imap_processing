from dataclasses import dataclass

import numpy as np


@dataclass
class Constants:
    """
    Class for shared constants across CDF classes.

    Attributes
    ----------
    INT_FILLVAL: np.int64
        Recommended FILLVAL for all integers (numpy int64 min)
    INT_MAXVAL: np.int64
        Recommended maximum value for INTs (numpy int64 max)
    DOUBLE_FILLVAL: np.float64
        Recommended FILLVALL for all floats
    MIN_EPOCH: int
        Recommended minimum epoch based on MMS approved values
    MAX_EPOCH: int
        Recommended maximum epoch based on MMS approved values
    """

    INT_FILLVAL = np.iinfo(np.int64).min
    INT_MAXVAL = np.iinfo(np.int64).max
    DOUBLE_FILLVAL = np.float64(-1.0e31)
    MIN_EPOCH = -315575942816000000
    MAX_EPOCH = 946728069183000000


class Epoch:
    """Shared attributes for the Epoch coordinate variables."""

    @staticmethod
    def output():
        """
        Return a dictionary with global base attributes.

        Returns
        -------
         Dictionary of epoch attributes
        """
        return {
            "CATDESC": "Default time",
            "FIELDNAM": "Epoch",
            "FILLVAL": Constants.INT_FILLVAL,
            "FORMAT": "a2",
            "LABLAXIS": "Epoch",
            "UNITS": "ns",
            "VALIDMIN": Constants.MIN_EPOCH,
            "VALIDMAX": Constants.MAX_EPOCH,
            "VAR_TYPE": "support_data",
            "SCALETYP": "linear",
            "MONOTON": "INCREASE",
            "TIME_BASE": "J2000",
            "TIME_SCALE": "Terrestrial Time",
            "REFERENCE_POSITION": "Rotating Earth Geoid",
        }


class GlobalBase:
    """
    Global base of attributes for all CDF files.

    This is automatically included in InstrumentBase.
    """

    @staticmethod
    def output():
        """
        Return a dictionary with global base attributes.

        Returns
        -------
        Global file attributes, including project, source_name, discipline, PI_name,
        PI_affiliation, instrument_type, and mission_group. This should be the same
        for all instruments.
        """
        return {
            "Project": "STP>Solar-Terrestrial Physics",
            "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
            "Discipline": "Solar Physics>Heliospheric Physics",
            "PI_name": "Dr. David J. McComas",
            "PI_affiliation": [
                "Princeton Plasma Physics Laboratory",
                "100 Stellarator Road, Princeton, NJ 08540",
            ],
            "Instrument_type": "Particles (space)",
            "Mission_group": "IMAP>Interstellar Mapping and Acceleration Probe",
        }


@dataclass
class InstrumentBase(GlobalBase):
    """Each instrument should extend this class and replace the info as needed.

    Attributes
    ----------
    version: str
        The software version
    descriptor: str
        Descriptor of the instrument (Ex: "IDEX>Interstellar Dust Experiment")
    text: str
        Explanation of the instrument, usually as a paragraph.

    """

    version: str
    descriptor: str
    text: str

    def output(self):
        """
        Generate the output for the instrument as a dictionary.

        Returns
        -------
        dictionary of correctly formatted values for the data_version, descriptor, text,
         and logical_file_id, added on to the global attributes from GlobalBase
        """
        return {
            "Data_version": self.version,
            "Descriptor": [self.descriptor],
            "TEXT": [self.text],
            "Logical_file_id": ["FILL ME IN AT FILE CREATION"],
        } | GlobalBase.output()


@dataclass
class DataLevelBase:
    """
    Class for all the attributes for the data level base.

    This is used to make the attributes for each data level, and includes
    InstrumentBase for the required instrument attributes and global attributes.

    Attributes
    ----------
    data_type: str
        The level of data, ex "L1>Level-1"
    logical_source: str
        The source of the data, ex "imap_idex_l1"
    logical_source_desc: str
        The description of the data, ex "IMAP Mission IDEX Instrument Level-1 Data."
    instrument_base: InstrumentBase
        The InstrumentBase object describing the basic instrument information
    """

    data_type: str
    logical_source: str
    logical_source_desc: str
    instrument_base: InstrumentBase

    def output(self):
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        dictionary of correctly formatted values for the attributes in the class and
        the attributes from InstrumentBase
        """
        return {
            "Data_type": self.data_type,
            "Logical_source": self.logical_source,
            "Logical_source_description": self.logical_source_desc,
        } | self.instrument_base.output()


@dataclass
class TypeBase:
    """
    The general class for attributes, with some reasonable defaults.

    Attributes
    ----------
    validmin: np.float64 | np.int64
        The valid minimum value, required
    validmax: np.float64 | np.int64
        The valid maximum value, required
    display_type: str default=None
        The display type of the plot (ex "no_plot"), required
    catdesc: str, default=None
        The category description, "CATDESC" attribute, required
    fieldname: str, default=None
        The fieldname, "FIELDNAM" attribute
    var_type: str, default="support_data"
        The type of data
    fill_val: np.int64, default=Constants.INT_FILLVAL
        The values for filling data
    scale_type: str, default="linear"
        The scale of the axis, "SCALETYP" attribute
    label_axis: str, default=None
        Axis label, "LABLAXIS" attribute
    format: str, default=None
        The format of the data, in Fortran format
    units: str, default=None
        The units of the data
    """

    validmin: np.float64 | np.int64
    validmax: np.float64 | np.int64
    display_type: str = None
    catdesc: str = None
    fieldname: str = None
    var_type: str = "support_data"
    fill_val: np.int64 = Constants.INT_FILLVAL
    scale_type: str = "linear"
    label_axis: str = None
    format: str = None
    units: str = None

    def output(self):
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        Dictionary of correctly formatted values for the attributes in the class
        """
        return {
            "CATDESC": self.catdesc,
            "DISPLAY_TYPE": self.display_type,
            "FIELDNAM": self.fieldname,
            "FILLVAL": Constants.INT_FILLVAL,
            "FORMAT": self.format,
            "LABLAXIS": self.label_axis,
            "UNITS": self.units,
            "VALIDMIN": self.validmin,
            "VALIDMAX": self.validmax,
            "VAR_TYPE": self.var_type,
            "SCALETYP": self.scale_type,
        }


@dataclass
class ScienceBase(TypeBase):
    """
    The class for Science attributes, in particular with depend_0 attributes.

    It also contains all the attributes and defaults from the generic TypeBase as well.

    Attributes
    ----------
    depend_0: str = None
        The first degree of dependent coordinate variables.
        Although this is an optional keyword, it is required for every instance.
    depend_1: str = None, optional
        The second degree of dependent coordinate variables. This is used for 2d data.
    depend_2: str = None, optional
        The third degree of dependent coordinate variables. This is used for 3d data.
        If this variable is used, there must also be a depend_1 value.
    variable_purpose: str = None, optional
        The variable purpose attribute tells which variables are worth plotting.
    var_notes: str = None, optional
        Notes on the variable
    """

    depend_0: str = None
    depend_1: str = None
    depend_2: str = None
    variable_purpose: str = None
    var_notes: str = None

    def output(self):
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
        Dictionary of correctly formatted values for the attributes in the class.
        If the optional parameters are not defined, they are not included as attributes
        in the output.
        """
        if self.depend_0 is None:
            raise TypeError("ScienceBase requires depend_0 attribute.")

        endval = {"DEPEND_0": self.depend_0}
        if self.depend_1 is not None:
            endval["DEPEND_1"] = self.depend_1

        if self.depend_2 is not None:
            endval["DEPEND_2"] = self.depend_2

        if self.variable_purpose is not None:
            endval["VARIABLE_PURPOSE"] = self.variable_purpose

        if self.var_notes is not None:
            endval["VAR_NOTES"] = self.var_notes
        return endval | super().output()


@dataclass
class IntBase(ScienceBase):
    """
    The Int version of ScienceBase with defaults to use for int-based data.

    Attributes
    ----------
    format: str, default="I18"
        The format of the data, in Fortran format
    units: str, default="int"
        The units of the data
    """

    format: str = "I18"
    units: str = "int"


@dataclass
class FloatBase(ScienceBase):
    """
    The float version of ScienceBase with defaults to use for float-based data.

    Attributes
    ----------
    format: str, default="F64.5"
        The format of the data, in Fortran format
    fill_val: np.float64, default=Constants.DOUBLE_FILLVAL
        The values for filling data
    units: str, default="float"
        The units of the data
    """

    format: str = "F64.5"
    fill_val: np.float64 = Constants.DOUBLE_FILLVAL
    units: str = "float"


@dataclass
class StringBase:
    """
    A base for String-based data, with string based defaults.

    This class does not have the same required attributes as ScienceBase, as it is a
    standalone class that doesn't extend any other class.

    Attributes
    ----------
    depend_0: str
        The first degree of dependent coordinate variables.
    catdesc: str, default=None
        The category description, "CATDESC" attribute, required
    fieldname: str, default=None
        The fieldname, "FIELDNAM" attribute
    format: str, default="A80"
        The format of the data, in Fortran format
    var_type: str, default="metadata"
        The type of data
    display_type: str, default="no_plot"
        The display type of the plot
    """

    depend_0: str
    catdesc: str = None
    fieldname: str = None
    format: str = "A80"
    var_type: str = "metadata"
    display_type: str = "no_plot"

    def output(self):
        """
        Generate the output for the data level as a dictionary.

        Returns
        -------
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
