.. _calibration_files:

Ancillary Files
=================

Ancillary files are files used in processing that are provided by instrument teams
to the SDC. These are files which contain values that could change over the course of
the mission. This could include calibration data, processing settings, lookup tables,
or other data provided by the instrument teams to be used in processing.

Any values which could be changed by the instrument team should be in a ancillary file.

These files will be tracked in the SDC and uploaded to the `SPDF archive <https://spdf.gsfc.nasa.gov/>`_.
Therefore, they should use unique filenames which clearly distinguish them from each other. Beyond that, they
can follow any format that is convenient for the instrument team and the SDC.

This document lays out the requirements for ancillary files in the SDC and the expected file naming
convention.

File overview
^^^^^^^^^^^^^^

In this document, there are two main kinds of ancillary files:

#. Universal ancillary files
#. Time window ancillary files

**Universal ancillary files** are files which are expected to change rarely. Generally, there should be one file
which covers all times.

**Time window ancillary files** are files which are expected to change more frequently, and are only valid for a certain
time range.

The main distinction between these types of files is the way they are applied to reprocessing. Universal ancillary files would generally
use the most recent version for all time ranges and reprocessing. Time window ancillary files should be defined only for a specific time range,
and reprocessing should usually use the same file as the first processing.

These two types of files are mostly used for creating the filename and processing conventions, so if your file doesn't neatly fit into
either category, that's okay. The main thing is that all files should try and follow the conventions in this document.

Specific examples
^^^^^^^^^^^^^^^^^

As an example, we will be looking at the MAG calibration files. These are supplemental files provided by the instrument team for use in
L1 and L2 processing.

The first file (L1B rotation matrices) is expected to change rarely,
with the most recent version being used for all data. This file contains rotation calibration matrices from
the boom to the instrument. These files would only change if there was a physical change to the boom or if there is
some more accurate information available, so they are universal ancillary files.

The second file contains matrices which are applied to create the L2 and L2pre data. They are updated
approximately monthly, and contain a start and end time for validity. During reprocessing, the data file with the most
recent version and a time range that includes the data will be used. Since these are files which specifically cover a time range,
they are time window ancillary files.

These two cases cover the two main types of ancillary files. They both will be uploaded to the SDC
with the API, and the correct file to use will be determined by the filename at processing time.

File naming convention
^^^^^^^^^^^^^^^^^^^^^^

The file name for each file should be the following format::

    imap_<instrument>_<description>_<start_date>_<end_date>_<version>.<extension>

Since these names are only used in instrument processing, they are flexible and can be adapted to
accommodate different needs. This is more of a rule of thumb than a strict requirement.

* ``<instrument>``: The instrument name associated with the ancillary file
* ``<description>``: A descriptive name for the ancillary file which distinguishes between
  other ancillary files used by the instrument. Rather than "ancillary" or "calibration", this should be similar to
  "rotation-matrices" or "l2pre-calibration." Multiple words should be split by hyphens.
* ``<start_date>``: The start time of the validity of the ancillary file, in the format "YYYYMMDD". This is required for
  all file types. For universal ancillary files, the start time should be the mission launch date.
* ``<end_date>``: The end time of the validity of the ancillary file, in the format "YYYYMMDD". This is optional for files,
  with the understanding that if end_date is not provided, the file is valid until a file with a later start_date and no end_date.
* ``<version>``: The version of the file, with the format `v000`. This version should be incremented with EVERY new file
  for a type of ancillary file, even if the time span does not overlap. This is used to resolve conflicts if multiple files could be used.
* ``<extension>``: The file extension. This is likely to be CDF, but could be any format that is convenient for the instrument team and SDC.

File naming examples
^^^^^^^^^^^^^^^^^^^^

An L1B MAG rotation matrix ancillary file:

* ``imap_mag_l1b-rotation-calibration_[launch date]_v001.cdf``

This file is assumed to be valid for all times.

An L1B MAG rotation matrix ancillary file with a start date:

* ``imap_mag_l1b-rotation-calibration-file_20250101_v002.cdf``

This file is only valid after 2025-01-01. If both of these files were uploaded, the second file would be used
for all times after 2025-01-01, since the version is higher than ``imap_mag_l1b-rotation-calibration-file_v001.cdf``. However,
dates before 2025-01-01 would use the first file.

Most global ancillary files should be applied to all time ranges, but there may be cases where a file should
only be used after a certain date - for example, if something physical occurred on the spacecraft on a date which changes
all data after that date, but does not affect data before that date.

An L2 MAG time window ancillary file:

* ``imap_mag_l2-calibration_20250101_20250201_v001.cdf``

This is valid between 2025-01-01 and 2025-02-01, inclusive.

* ``imap_mag_l2-calibration_20250201_v002.cdf``

This is valid starting at 2025-02-01, and is assumed to be valid until a new file without an end date is uploaded.

If there is a file in the time range with a higher version, then that file should be used. Say we have
an additional file uploaded:

``imap_mag_l2-calibration_20250301_20250302_v003.cdf``

For all data between 2025-03-01 and 2025-03-02, this file would be used. However, for data after 2025-03-02,
``imap_mag_l2-calibration_20250201_v002.cdf`` would be used, because that is still the highest version file for that time range.

This way, it is possible to upload corrections for smaller time ranges without affecting the long term ancillary files.

.. note::
    The versions increment for all new `imap_mag_l2-calibration` files, not just those with overlapping time spans.
    This helps make it clear in every case which file is newer and should be used. This requirement should be
    enforced by the SDC API.

    Ancillary files with different names can share versions, so the mag_l2-calibration and mag_l1b-rotation-calibration can have duplicate versions.

If a new file is uploaded, with the intent that it should be used indefinitely into the future:

``imap_mag_l2-calibration_20250301_v004.cdf``

Then, the previous file with no end date ``imap_mag_l2-calibration_20250201_v002.cdf`` would be used for all data between 2025-02-01 and 2025-03-01,
and all data after 2025-03-01 would use ``imap_mag_l2-calibration_20250301_v004.cdf``.

.. note::
    We haven't decided if uploading a new file with no end date should cause the previous file to be renamed
    (so, moving ``imap_mag_l2-calibration_20250201_v002.cdf`` to ``imap_mag_l2-calibration_20250201_20250301_v002.cdf`` in
    the previous example).


