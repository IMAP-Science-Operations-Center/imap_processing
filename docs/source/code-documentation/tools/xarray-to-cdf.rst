.. _xarray-cdf:

####################
cdflib.xarray_to_cdf
####################

The Python library `cdflib <https://github.com/MAVENSDC/cdflib>`_ is a pure-python implementation of the CDF specification.  Of particular note is the function `xarray_to_cdf <https://cdflib.readthedocs.io/en/latest/api/cdflib.xarray.xarray_to_cdf.html#cdflib.xarray.xarray_to_cdf>`_.
Because IMAP L0->L2 processing is using xarray for internal data representation, **IMAP L0->L2 processing will be using this function to generate all CDF files**.

This function takes an `xarray.Dataset <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_ object as input, and will output a CDF file.  Further details about how it works are below.

Xarray Introduction
===================

Xarray is a powerful python library for handling multi-dimensional data.  It has a strong connection with the netCDF file format.  The developers intended xaray to be an "in-memory" representation of a netCDF file.

Perhaps the largest difference between netCDF and CDF is that netCDF has built-in methods to attach *dimensions* and *coordinates* to data.  Similarly, xarray Datasets have this capability as well.
Full documentation about xarray Dataset objects are located here `https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html <https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html>`_

Within a CDF file, there is no inherent way to attach coordinates to a variable.  CDF is a simpler format, and only has the concept of Attributes and Variables.  The "CDF-native" way of specifying dimensions to a variable is to use "DEPEND" attributes (see sections :ref:`ISTP Compliance` and :ref:```DEPEND_i```).

.. note:: the netCDF file format was created by NOAA a few years after the CDF file format to plug in some of the shortfallings of the CDF file format.  These days, the original reasons for the split between the two formats have largely disappeared.

xarray_to_cdf
=============

The following is a minimal example of working with xarray_to_cdf -

.. code-block:: python

   from cdflib.xarray import xarray_to_cdf
   import xarray as xaray
   var_data = [[1, 2, 3], [4, 2, 6], [7, 8, 9]]
   var_dim = ['epoch', 'energy']
   epoch_data = [1, 2, 3]
   epoch = xr.Variable(epoch_dims, epoch_data)
   ds = xr.Dataset(data_vars={'data': data, 'epoch': epoch})
   xarray_to_cdf(ds, 'hello.cdf')

The xarray_to_cdf function converts an xarray.Dataset object into a CDF file.

For IMAP, we will be using the following flags on xarray_to_cdf

* datetime64_to_cdftt2000
   * Working with numpy datetime64 objects for time is recommended because of the precision afforded, as well as the speed of calculations
* istp
   * Enabled by default.  This flag will perform some checks to verify that the function is able to make ISTP compliant CDF files from the Dataset object.
* terminate_on_warning
   * This flag ensures that the function terminates if there are any issues found, so we can avoid creating incomplete or non-compliant files.

xarray_to_cdf with the above flags will perform the following steps -

Verification
------------

#. Verifies that required global attributes are present to meet ISTP compliance
#. Verifies that variables have a VAR_TYPE attribute of either "data", "support_data", or "metadata"
#. Verifies that all variables have the appropriate number of DEPEND_{i} attributes, based on the number of dimensions of the variable
#. Verifies that the DEPEND_{i} attributes point to variables that are the size and shape expected to act as coordinates
#. Verifies that each variable has the expected variable attributes to be ISTP compliant

Conversion
----------

#. Converts all variables with the word "epoch" in their name from datetime64 objects into the CDF native time format of CDF_TT2000 (nanoseconds since the year 2000)
#. Converts all other data into one of the following CDF native formats - CDF_INT8, CDF_DOUBLE, CDF_UINT4, or CDF_CHAR
#. Converts the attributes VALIDMIN, VALIDMAX, and FILLVAL to have the same data type as the data they are attached to
   * For example, if you specify the attribute VALIDMIN=5 for a variable named epoch, it will be converted to a CDF_TT2000 data type when written to the file
#. Writes the Dataset attributes to the CDF file as the CDF's Global Attributes
#. Writes the Dataset's variables to the CDF file
   * Any variable with DEPEND_0 = epoch as an attribute will be set to vary across records in the CDF file.  Otherwise, the data will be contained within a single record.
#. Writes the attributes attached to the Dataset Variables to the file


Shortcomings
-------------
While the above steps get a CDF file a large way towards ISTP compliance, there are several important caveats to take note of -

* This code does not check the values *within* the attributes (except VAR_TYPE and DEPEND_{i}), only that the attributes exist!
* This function does not put the "epoch" variable as the first thing in the file, which was recommended (but not required) by the SPDF
* This code creates variables in a row-major format.  Column-major is recommended (but not required) by the SPDF.

**************************
IMAP xarray_to_cdf Example
**************************

Suppose there is an instrument on IMAP named JIM that took 3 samples of ions at 3 different energies, such that the data looks like the following -

+------------------------+------------+------------+------------+
| Time \                 | 0.5-1.5 eV | 1.5-2.5 eV | 2.5-3.5 eV |
| Energy Bins            |            |            |            |
+========================+============+============+============+
| 2025-01-01T00:00:01    | 1          | 2          | 3          |
+------------------------+------------+------------+------------+
| 2025-01-01T00:00:02    | 4          | 5          | 6          |
+------------------------+------------+------------+------------+
| 2025-01-01T00:00:03    | 7          | 8          | 9          |
+------------------------+------------+------------+------------+

The following python code is the **minimum** code you'd need to store this data in a CDF file that is fully compliant with the documentation in :ref:`CDF File Requirements` -

.. code-block:: python

   from cdflib.xarray import xarray_to_cdf
   import numpy as np
   import xarray as xr


   INT_FILLVAL = np.iinfo(np.int64).min # Recommended FILLVAL for all integers
   DOUBLE_FILLVAL = np.float64(-1.0e31) # Recommended FILLVALL for all floats
   MIN_EPOCH = -315575942816000000 # Recommended min/max epoch
   MAX_EPOCH = 946728069183000000 # Recommended min/max epoch

   global_attrs = {
      # Project, Source_name, Descipline, Mission Group, and PI info will all
      # stay the same for all instruments on IMAP.
      "Project": "STP>Solar-Terrestrial Physics",
      "Source_name": "IMAP>Interstellar Mapping and Acceleration Probe",
      "Discipline": "Solar Physics>Heliospheric Physics",
      "Mission_group": "IMAP>Interstellar Mapping and Acceleration Probe",
      "PI_name": "Dr. David J. McComas",
      "PI_affiliation": [
         "Princeton Plasma Physics Laboratory",
         "100 Stellarator Road, Princeton, NJ 08540",
      ],
      # The following attributes are specific to JIM.
      "Instrument_type": "Particles (space)",
      "Data_type": "L1_Mode_Description>Level-1 Mode Description",
      "Data_version": "001",
      "Descriptor": "JIM>Just an Ion Monitor",
      "TEXT": (
         "JIM is a fictitious instrument that counts ions at 3 different energies on "
         "the IMAP mission. This is where a detailed description of the instrument "
         "goes, as well as the type of data in the file.  For example, if a (mode) "
         "or (descriptor) exist they can be described here."
      ),
      "Logical_file_id": "imap_jim_l1_mode-description_20250101_v001",
      "Logical_source": "imap_jim_l1_mode-description",
      "Logical_source_description": "IMAP Mission JIM Instrument Level-1 (mode) (description) Data.",
   }

   # These epoch attributes will remain the same across all instruments
   epoch_attrs = {
      "CATDESC": "Default time",
      "FIELDNAM": "epoch",
      "FILLVAL": INT_FILLVAL,
      "FORMAT": "a2",
      "LABLAXIS": "Epoch",
      "UNITS": "ns",
      "VALIDMIN": MIN_EPOCH,
      "VALIDMAX": MAX_EPOCH,
      "VAR_TYPE": "support_data",
      "SCALETYP": "linear",
      "MONOTON": "INCREASE",
      "TIME_BASE": "J2000",
      "TIME_SCALE": "Terrestrial Time",
      "REFERENCE_POSITION": "Rotating Earth Geoid",
   }

   # Describes a variable that holds the energy bins for the real data
   energy_bins_attrs = {
      "CATDESC": "The energy bins for the JIM instrument",
      "FIELDNAM": "Energy bins",
      "FILLVAL": np.float64(-1.0e31),
      "FORMAT": "I12", # Display up to 12 numbers of an integer
      "LABLAXIS": "Energy",
      "UNITS": "eV",
      "VALIDMIN": 1, #Fabricated value
      "VALIDMAX": 3, #Fabricated value
      "VAR_TYPE": "support_data",
      "SCALETYP": "linear",
      "VAR_NOTES": (
         "We pretend there are 3 energy bins that data is collected for. "
         "The 3 Energy bins are 0.5-1.5, 1.5-2.5, and 2.5-3.5 electron Volts. "
         "These are used by the variable imap_jim_counts_(mode)_(description)"
         "Ideally we would actually have additional variables that describe "
         "the minimum and the maximum values in each of these bins, but for our "
         "purposes of creating a minimal example, we'll leave extra variables out. "
         "Longer notes about the variable can be placed into this attribute."
      ),
   }

   # Describes a variable that holds the "counts" data
   counts_attrs = {
      "DEPEND_0": "epoch",
      "DEPEND_1": "energy",
      "DISPLAY_TYPE": "spectrogram",
      "FILLVAL": INT_FILLVAL,
      "FORMAT": "I12", # Display up to 12 numbers of an integer
      "UNITS": "counts",
      "VALIDMIN": 0, # Fabricated value
      "VALIDMAX": 5000, # Fabricated value
      "VAR_TYPE": "data",
      "SCALETYP": "linear",
      "VARIABLE_PURPOSE": "PRIMARY",
      "CATDESC": "80 character description of the measurement goes here. For JIM, we mention it is counts per energy bin per time. ",
      "FIELDNAM": "Counts per Energy Bin",
      "LABLAXIS": "Ion Counts",
      "VAR_NOTES": (
         "The number of ions counted at each energy bin. "
         "Sampled at 1 sample per second, at 1, 2, and 3 eV. "
      ),

   }

   # Create 3 data arrays, one for time, one for the energies, and one for the real data
   epoch_xr = xr.DataArray(
                  name="epoch",
                  data=[np.datetime64("2025-01-01T00:00:01"), np.datetime64("2025-01-01T00:00:02"), np.datetime64("2025-01-01T00:00:03")],
                  dims=("epoch"),
                  attrs=epoch_attrs,
               )
   energy_xr = xr.DataArray(
                  name="energy",
                  data=[1,2,3], # Fabricated energy bins
                  dims=("energy_dim"),
                  attrs=energy_bins_attrs,
               )
   counts_xr = xr.DataArray(
               name="imap_jim_counts_(mode)_(descriptor)",
               data=[[1,2,3], [4,5,6], [7,8,9]], # Fabricated data
               dims=("epoch", "energy_dim"),
               attrs=counts_attrs,
         )
   # Combine the data into an xarray.Dataset object
   data = xr.Dataset(data_vars={
                           "imap_jim_counts_mode_descriptor": counts_xr,
                     },
                     coords={
                           "epoch": epoch_xr,
                           "energy": energy_xr
                     },
                     attrs=global_attrs
                  )
   # Convert the data into a CDF using cdflib
   xarray_to_cdf(
         data,
         "imap_jim_l1_mode_description_20250101_v01.cdf",
         datetime64_to_cdftt2000=True,
         terminate_on_warning=True,
      )