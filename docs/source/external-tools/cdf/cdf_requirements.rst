#####################
CDF File Requirements
#####################

The following page describes the necessary contents inside of a CDF file generated for the IMAP mission.

For a more general introduction to CDF files, see :ref:`CDF Introduction`.

For sample code that will generate a file that matches the below guidelines, see :ref:`IMAP xarray_to_cdf Example`

*********************************************************
Required Attributes of ISTP Compliant CDF File Components
*********************************************************
CDF files are composed of these main components: Global Attributes, Data Variables, Support Data Variables, and Metadata Variables. Each of these components has a set of required attributes that must be included in the CDF file.

Required Global Attributes
==========================

* Data_level
* Data_type
* Data_version
* Descriptor
* Discipline
* Instrument_type
* Logical_file_id
* Logical_source
* Logical_source_description
* Mission_group
* PI_affiliation
* PI_name
* Project
* Source_name
* TEXT

Section :ref:`global-attributes` provides more information on each of these attributes.

Required Data Attributes
========================

* CATDESC
* DEPEND_0
* DEPEND_i (if applicable)
* DISPLAY_TYPE
* FIELDNAM
* FILLVAL
* FORMAT or FORM_PTR. Required to have at least one of these attributes.
* LABLAXIS or LABL_PTR_i. Required to have at least one of these attributes.
* UNITS or UNIT_PTR. Required to have at least one of these attributes.
* VALIDMIN
* VALIDMAX
* VAR_TYPE = “data”

Section :ref:`data-attributes` provides more information on each of these attributes.

Required Support Data Attributes
================================

* CATDESC
* DEPEND_0 (if time varying)
* FIELDNAM
* FILLVAL (if time varying)
* FORMAT or FORM_PTR. Required to have at least one of these attributes.
* SI_CONVERSION
* UNITS/UNIT_PTR
* VALIDMIN (if time varying)
* VALIDMAX (if time varying)
* VAR_TYPE = “support_data”

Variables appearing in a data variable's ``DEPEND_i`` attribute require a minimal set of their own attributes to fulfill their role in supporting the data variable.

Section :ref:`data-attributes` provides more information on each of these attributes.

Required Metadata Data Attributes
=================================

* CATDESC
* DEPEND_0 (if time varying)
* FIELDNAM
* FILLVAL (if time varying)
* FORMAT or FORM_PTR. Required to have at least one of these attributes.
* VAR_TYPE = “metadata”

Section :ref:`data-attributes` provides more information on each of these attributes.

See :ref:`metadata-attributes` for more information on metadata data attributes for ``LABL_PTR_i``.

.. _global-attributes:

*****************
Global Attributes
*****************

Global attributes are used to provide information about the data set as an entity. Together with variables and variable attributes, the global attributes make the data correctly and independently usable by someone not connected with the instrument team, and hence, a good archive product.

Global attributes that have been identified for use with IMAP data products are listed below. Additional Global attributes can be defined but they must start with a letter and can otherwise contain letters, numbers and the underscore character (no other special characters allowed). Note that CDF attributes are case-sensitive and must exactly follow what is shown here.

ISTP Compliant Global Attributes are listed here: `https://spdf.gsfc.nasa.gov/istp_guide/gattributes.html <https://spdf.gsfc.nasa.gov/istp_guide/gattributes.html>`_, and notes about how they are used on IMAP are below -

``Data_type``
=============
It is a combination of the following filename components: ``data level``, and data product ``descriptor``.  It contains both the short form and the long form of the name, separated by a ``>`` (e.g. ``L1A_norm>Level-1A normal rate``). The short form is separated by a ``_``.

``Data_version``
================
This attribute identifies the version of a particular CDF data file. (e.g. ``v001``)

``Descriptor``
==============
This attribute identifies the name of the instrument or sensor that collected the data. Note that this is not the same as ``descriptor`` from ``Data_type``, but the name of the instrument that collected the data.

Both a long name and a short name are given.  For any data file, only a single value is allowed.

For IMAP, one of the following must be used:

* CoDICE>Compact Dual Ion Composition Experiment
* GLOWS>GLObal Solar Wind Structure
* HIT>High-energy Ion Telescope
* IDEX>Interstellar Dust Experiment
* IMAP-Hi>Interstellar Mapping and Acceleration Probe High
* IMAP-Lo>Interstellar Mapping and Acceleration Probe Low
* IMAP-Ultra>Interstellar Mapping and Acceleration Probe Ultra
* MAG>Magnetometer
* SWAPI>Solar wind and Pickup Ions
* SWE>Solar Wind Electrons

``Discipline``
==============
For IMAP, this value should always be ``Space Physics>Heliospheric Physics``. This attribute describes both the science discipline and sub discipline.

``Generation_date``
===================
Date stamps the creation of the file using the syntax YYYYMMDD, e.g. ``20150923``.

``Instrument_type``
===================
This attribute is used to facilitate making choices of instrument type. More than one entry is allowed. Valid IMAP values include:

* Electric Fields (space)
* Magnetic Fields (space)
* Particles (space)
* Plasma and Solar Wind
* Ephemeris

``Logical_file_id``
===================
This attribute stores the name of the CDF file as described in Section 3.1 but without the file extension or version (e.g. ``.cdf``). This attribute is required to avoid loss of the original source in the case of accidental (or intentional) renaming. This attribute must be manually set by the user during creation.

``Logical_source``
==================
This attribute determines the file naming convention and is used by CDA Web.  It is composed of the following other attributes:

* Source_name - (e.g. ``imap``)
* Descriptor - (e.g. the instrument, see above)
* Data_type - (e.g. data level, and descriptor)

These three attributes can be ordered in different ways based on the user's filename convention. SPDF uses default naming conventions of ``source_datatype_descriptor_yyyyMMdd`` but IMAP uses this convention, ``source_descriptor_datatype_yyyyMMdd_vNNN``. For example, ``source_datatype_descriptor_yyyyMMdd`` would result in filename convention like this, ``imap_l1a_norm-raw_mag_20241122``, whereas ``source_descriptor_datatype_yyyyMMdd_vNNN`` would result in filename convention like this, ``imap_mag_l1a_norm-raw_20241122_v001``. See :ref:`naming-conventions` for more details of IMAP filename convention.

``Logical_source_description``
==============================
This attribute writes out the full words associated with the encrypted Logical_source above, e.g., ``Level 1 Dual Electron Spectrometer Survey Data``. Users on CDAWeb see this value on their website.

``Mission_group``
=================
This attribute has a single value and is used to facilitate making choices of source through CDAWeb.  This value should be ``IMAP``.

``PI_affiliation``
==================
This attribute value should include the IMAP mission PI affiliation followed by a comma separated list of any Co-I affiliations that are responsible for this particular dataset. The following are valid IMAP values, of which the abbreviations should be used exclusively within this attribute value, and the full text of the affiliation included in the general text attribute as it is used solely in plot labels.

* JHU/APL - Applied Physics Laboratory
* GSFC - Goddard Space Flight Center
* LANL - Los Alamos National Laboratory
* LASP - Laboratory for Atmospheric and Space Physics
* SWRI - Southwest Research Institute
* UCLA - University of California Los Angeles
* UNH - University of New Hampshire

``PI_name``
===========
This attribute value should include first initial and last name of the IMAP mission PI followed by a comma-separated list of any Co-Is that are responsible for this particular dataset. For example, a single PI entry in this attribute would be: ``Dr. David J. McComas``.

``Project``
===========
This attribute identifies the name of the project and indicates ownership. For IMAP, this value should be ``STP>Solar-Terrestrial Physics``.

``Source_name``
===============
This attribute identifies the observatory where the data originated. For IMAP, this should simply be ``IMAP``

``TEXT``
========
This attribute is an SPDF standard global attribute, which is a text description of the experiment whose data is included in the CDF. A reference to a journal article(s) or to a webpage describing the experiment is essential, and constitutes the minimum requirement. A written description of the data set is also desirable. This attribute can have as many entries as necessary to contain the desired information. Typically, this attribute is about a paragraph in length and is not shown on CDAWeb.

``MODS``
========
This attribute is an SPDF standard global attribute, which is used to denote the history of modifications made to the CDF data set. The MODS attribute should contain a description of all significant changes to the data set, essentially capturing a log of high-level release notes. This attribute can have as many entries as necessary and should be updated if there is a major version change.

``Parents``
===========
This attribute lists the parent data files for files of derived and merged data sets. The syntax for a CDF parent is: ``CDF>logical_file_id``. Multiple entry values are used for multiple parents. This attribute is required for any data products that are derived from 2 or more data sources and the file names of parent data should be clearly identified. CDF parents may include source files with non-cdf extensions.

Example
========

Here is an example of what this looks like for CoDICE:

    .. code-block:: python

        {'Project': ['STP>Solar-Terrestrial Physics'],
         'Source_name': ['IMAP>Interstellar Mapping and Acceleration Probe'],
         'Discipline': ['Solar Physics>Heliospheric Physics'],
         'Mission_group': ['IMAP>Interstellar Mapping and Acceleration Probe'],
         'PI_name': ['Dr. David J. McComas'],
         'PI_affiliation': ['Princeton Plasma Physics Laboratory',
          '100 Stellarator Road, Princeton, NJ 08540'],
         'File_naming_convention': ['source_descriptor_datatype_yyyyMMdd_vNNN'],
         'Data_version': ['001'],
         'Descriptor': ['CoDICE>Compact Dual Ion Composition Experiment'],
         'TEXT': ['The Compact Dual Ion Composition Experiment (CoDICE) will measure the distributions and composition of interstellar pickup ions (PUIs), particles that make it through the heliosheath into the heliosphere. CoDICE also collects and characterizes solar wind ions including the mass and composition of highly energized particles (called suprathermal) from the Sun. CoDICE combines an electrostatic analyzer(ESA) with a Time-Of-Flight versus Energy (TOF / E) subsystem to simultaneously  measure the velocity, arrival direction, ionic charge state, and mass of specific species of ions in the LISM. CoDICE also has a path for higher energy particles to skip the ESA but still get measured by the common TOF / E system. These measurements are critical in determining the Local Interstellar Medium (LISM) composition and flow properties, the origin of the enigmatic suprathermal tails on the solar wind distributions and advance understanding of the acceleration of particles in the heliosphere.'],
         'Instrument_type': ['Particles (space)'],
         'Logical_file_id': ['imap_codice_l1a_lo-sw-species-counts_20240319_v001'],
         'Data_type': ['L1A_lo-sw-species-counts->Level-1A Lo Sunward Species Counts Data'],
         'Logical_source': ['imap_codice_l1a_lo-sw-species-counts'],
         'Logical_source_description': ['IMAP Mission CoDICE Instrument Level-1A Lo Sunward Species Counts Data']
        }



**************
IMAP Variables
**************

There are three types of variables that should be included in CDF files: data, support data, and metadata. Additionally, required attributes are listed with each variable type listed below.

To facilitate data exchange and software development, variable names should be consistent across the IMAP instruments. Additionally, it is preferable that data types are consistent throughout all IMAP data products (e.g. all real variables are ``CDF_REAL4``, all integer variables are ``CDF_INT4``, and flag/status variables are ``UINT4``).
This is not to imply that only these data types are allowable within IMAP CDF files. All CDF supported data types are available for use by IMAP. For detailed information and examples, please see the following ISTP/IACG webpage:
`http://spdf.gsfc.nasa.gov/istp_guide/variables.html <http://spdf.gsfc.nasa.gov/istp_guide/variables.html>`_

Data
====
These are variables of primary importance (e.g., density, magnetic field, particle flux). Data are always time (record) varying, but can be of any dimensionality or CDF supported data type. Real or Integer data are always defined as having one element.

Required Epoch Variable
-----------------------
All IMAP CDF Data files must contain at least one variable of data type ``CDF_TIME_TT2000`` named ``epoch``.  All time varying variables in the CDF data set will depend on either this ``epoch`` or another variable of type ``CDF_TIME_TT2000``.  More than one ``CDF_TIME_TT2000`` variable is allowed in a data set to allow for more than one time resolution.  It is recommended that all such time variable use ``epoch`` within their variable name.

.. note::
   In the ``xarray_to_cdf`` function described in :ref:`cdflib.xarray_to_cdf`, all variables with ``epoch`` in their name will be converted to ``CDF_TT2000`` if the flag "istp=True" is given.

For ISTP compliance, the time value of a record refers to the **center** of the accumulation period if the measurement is not an instantaneous one.

``CDF_TT2000`` is defined as an 8-byte signed integer with the following characteristics:

* Time_Base=J2000 (Julian date 2451545.0 TT or 2000 January 1, 12h TT)
* Resolution=nanoseconds
* Time_Scale=Terrestrial Time (TT)
* Units=nanoseconds
* Reference_Position=rotating Earth Geoid

Given a current list of leap seconds, conversion between TT and UTC is straightforward (``TT = TAI + 32.184s``; ``TT = UTC + deltaAT + 32.184s``, where ``deltaAT`` is the sum of the leap seconds since 1960; for example, for 2009, ``deltaAT = 34s``). Pad values of ``-9223372036854775808 (0x8000000000000000)`` which corresponds to ``1707-09-22T12:13:15.145224192``; recommended ``FILLVAL`` is same.

It is proposed that the required data variables ``VALIDMIN`` and ``VALIDMAX`` are given values corresponding to the dates ``1990-01-01T00:00:00`` and ``2100-01-01T00:00:00`` as these are well outside any expected valid times.

.. _data-attributes:

Data Attributes
---------------

``CATDESC``
^^^^^^^^^^^
This is a human readable description of the data variable. Generally, this is an 80-character string which describes the variable and what it depends on.

``DEPEND_0``
^^^^^^^^^^^^
Explicitly ties a data variable to the time variable on which it depends (i.e. - the `epoch` variable). All variables which change with time must have a ``DEPEND_0`` attribute defined.

Even if there is no time dependency of the data *within* the CDF file, SPDF has stated it would still be ideal to include a ``DEPEND_0`` for each variable to ensure continuity *between* CDF files.

``DEPEND_i``
^^^^^^^^^^^^
Ties a dimensional data variable to a ``SUPPORT_DATA`` variable on which the i-th dimension of the data variable depends. The number of ``DEPEND`` attributes must match the dimensionality of the variable, i.e., a one-dimensional variable must have a ``DEPEND_1``, a two-dimensional variable must have a ``DEPEND_1`` and a ``DEPEND_2`` attribute, etc. The value of the attribute must be a variable in the same CDF data set. It is strongly recommended that ``DEPEND_i`` variables hold values in physical units. ``DEPEND_i`` variables also require their own attributes, as described in the following sections.

``DISPLAY_TYPE``
^^^^^^^^^^^^^^^^
This tells automated software, such as CDAWEB, how the data should be displayed.
Examples of valid values include

* time_series
* spectrogram
* stack_plot
* image

``FIELDNAM``
^^^^^^^^^^^^
A shortened version of ``CATDESC`` which can be used to label a plot axis or as a data listing heading. This is a string, up to ~30 characters in length.

``FILLVAL``
^^^^^^^^^^^
Identifies the fill value used where data values are known to be bad or missing.
``FILLVAL`` is required for time-varying variables. Fill data are always non-valid data. The ISTP standard fill values are listed below:

* BYTE ---- -128
* INTEGER*2 ---- -32768
* INTEGER*4 ---- -2147483648
* INTEGER*8 ---- -9223372036854775808
* Unsigned INTEGER*1 ---- 255
* Unsigned INTEGER*2 ---- 65535
* Unsigned INTEGER*4 ---- 4294967295
* REAL*4 ---- -1.0E31
* REAL*8 ---- -1.0E31
* EPOCH ---- -1.0E31 (9999-12-31:23:59:59.999)
* EPOCH16 ---- -1.0E31 (9999-12-31:23:59:59.999999999999)
* TT2000 ---- -9223372036854775808LL (9999-12-31:23:59:59.999999999999)

.. note::
   Using ``xarray_to_cdf``, these values are automatically cast to be the same type of data as the CDF variable they are attached to.  For example, if your data is ``REAL4`` and you specify your ``VALIDMIN=0``, the function will know to store the ``0`` as a ``REAL4`` type as well.

``FORMAT``
^^^^^^^^^^
This field allows software to properly format the associated data when displayed on a screen or output to a file. Format can be specified using either Fortran or C format codes. For instance, ``F10.3`` indicates that the data should be displayed across 10 characters where 3 of those characters are to the right of the decimal.

``LABLAXIS``
^^^^^^^^^^^^
Required if not using ``LABL_PTR_i``.

Used to label a plot axis or to provide a heading for a data listing. This field is generally 6-10 characters.

``LABL_PTR_i``
^^^^^^^^^^^^^^
Required if not using ``LABLAXIS``.

Used to label a dimensional variable when one value of ``LABLAXIS`` is not sufficient to describe the variable or to label all the axes. ``LABL_PTR_i`` is used instead of ``LABLAXIS``, where i can take on any value from 1 to n where n is the total number of dimensions of the original variable.

The value of ``LABL_PTR_1`` is another variable within the same CDF, which will contain the short character strings to describe the first dimension of the original variable.

An example of how ``LABL_PTR_i`` can be used is found at this following link: `https://spdf.gsfc.nasa.gov/istp_guide/variables.html#data_eg2 <https://spdf.gsfc.nasa.gov/istp_guide/variables.html#data_eg2>`_

``UNITS``
^^^^^^^^^
A 6-20 character string that identifies the units of the variable (e.g. ``nT`` for magnetic field). Use a blank character, rather than ``None`` or ``unitless``, for variables that have no units (e.g., a ratio or a direction cosine).

``VALIDMIN``
^^^^^^^^^^^^
The minimum value for a particular variable that is expected over the lifetime of the mission. Used by application software to filter out values that are out of range. The value must match the data type of the variable.

.. note::
   Using ``xarray_to_cdf``, these values are automatically cast to be the same type of data as the CDF variable they are attached to

``VALIDMAX``
^^^^^^^^^^^^
The maximum value for a particular variable that is expected over the lifetime of the mission. Used by application software to filter out values that are out of range. The value must match the data type of the variable.

.. note::
   Using ``xarray_to_cdf``, these values are automatically cast to be the same type of data as the CDF variable they are attached to

``VAR_TYPE``
^^^^^^^^^^^^
Used in CDAWeb to indicate if the data should be used directly by users. Possible values:

* ``data`` - integer or real numbers that are plottable
* ``support_data`` - integer or real "attached" or secondary data variables
* ``metadata`` - labels or character variables
* ``ignore_data`` - data that can be ignored by CDAWeb plots. For example, packet header info.

Support Data
============
These are variables of secondary importance employed as ``DEPEND_i`` variables, but they may also be used for housekeeping or other information not normally used for scientific analysis.

``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR``
------------------------------------------

``DEPEND_i`` variables are typically physical values along the corresponding i-th dimension of the parent data variable, such as energy levels or spectral frequencies. The discrete set of values are located with respect to the sampling bin by ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR``, which hold the variable name containing the distance from the value to the bin edge. It is strongly recommended that IMAP ``DEPEND_i`` variables include ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR`` attributes that point to the appropriate variable(s) located elsewhere in the CDF file.

For example, for a variable ``energy_level`` that is the ``DEPEND_i`` of a particle distribution, if ``energy_dplus`` and ``energy_dminus`` are two variables pointed to by ``energy_level``’s ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR``, then element [n] corresponds to the energy bin ``(energy_level[n]-energy_dminus[n])`` to ``(energy_level[n]+energy_dplus[n])``. ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR`` can point to the same variable which implies that ``energy_level[n]`` is in the center of the bin. ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR`` must have the same number of values as the size of the corresponding dimension of the parent variable, or hold a single constant value which applies for all bins. They can be record-varying, in which case they require a ``DEPEND_0`` attribute.

In the case of the ``DEPEND_0`` ``timetag`` variable, ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR`` together with the ``timetag`` identify the time interval over which the data was sampled, integrated, or otherwise regarded as representative of. ``DELTA_PLUS_VAR`` and ``DELTA_MINUS_VAR`` variables require ``FIELDNAM``, ``UNITS`` and ``SI_CONVERSION`` attributes; in principle, these could differ from those of the ``DEPEND_i`` parent. They also require ``VAR_TYPE=SUPPORT_DATA``. Other standard attributes might be helpful.

.. _metadata-attributes:

Metadata Data
=============
When data has more than one dimension, in addition to ``DEPEND_i``, it requires to have ``LABL_PTR_i``
attributes for each extra dimension of the data. ``LABL_PTR_i`` of corresponding ``DEPEND_i`` will
required to have these attributes:

* CATDESC
* FIELDNAM
* FORMAT
* VAR_TYPE = “metadata”

In this case, ``FORMAT`` is required to be of character type. For example, ``FORMAT: A10`` indicating that the data is a string of 10 characters.

This is used to label the axes of the data in a plot. For example, if the data is a 2D plot of energy vs. time, the ``LABL_PTR_1`` would be the energy axis. Time is already labeled by the ``DEPEND_0`` attribute.

**************************
Variable Naming Convention
**************************

Data Variables
==============

IMAP data variables should adhere to the following naming conventions:

``parameter[_coordinateSystem][_timeInterval]``

An underscore is used to separate different fields in the variable name.
It is strongly recommended that variable name employ further fields, qualifiers, and information designed to identify unambiguously the nature of the variable, and instrument mode.
These variable names may only include lowercase letters, numbers, and underscores.  No upper-case letters, hyphens, or other special characters are allowed.


Required
---------
* parameter - a short representation of the physical parameter held in the variable. (e.g. ``density``, ``temperature``, ``energy``)

Optional
---------
* coordinateSystem - an identifier for the coordinate system in which the parameter is set (e.g. ``gse``, ``gsm``, ``rtn``)
* timeInterval - an identifier for the time interval over which the parameter is valid (e.g. ``1sec``, ``10sec``, ``1min``, ``1hour``)


Support Data Variables
======================

Support data variable names must begin with a letter and can contain numbers and underscores, but no other special characters. Support data variable names need not follow the same naming convention as Data Variables (5.1.1) but may be shortened for convenience.


**********************
File Naming Convention
**********************

See :ref:`naming-conventions` for a description of the file naming convention for IMAP CDF files.
