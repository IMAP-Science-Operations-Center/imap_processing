Data Dependency Management
==========================

The IMAP science data center utilizes an event-based processing system that allows for
processing as soon as data is available. This system is designed to be flexible to
accommodate the various requirements and inter-dependencies for all 10 instruments.

As part of our requirements, we need some way to explicitly describe the dependencies
for each file. We also need to be able to flexibly update the dependencies on a regular
basis, to accommodate changing requirements.

Overview
--------

When a file lands in the SDC, it is added to our data bucket (Also called S3 or S3 bucket.) This bucket, as the name implies, is a simple collection which contains all the files in the SDC, organized
like a file system.

Each data file is put into a specific subfolder depending on the file name. For example, a file named ``imap_swe_l0_sci_20240105_20240105_v00-01.pkts`` would be placed in the ``imap/swe/l0/2024/01`` folder.
More information about the naming conventions can be found in :ref:`naming-conventions`.

When a file of any level arrives in the bucket, it triggers the rest of processing. This is how we manage file processing within the SDC, rather than waiting until all files have arrived
or running at particular times of day. This allows us to quickly process data as soon as all the required pieces are available to us, and create a flexible system which can easily be updated
to add exceptions or new requirements on a per-instrument or per-level basis.

Detailed Description of File Processing
---------------------------------------

`Overview chart in Galaxy <https://lasp.colorado.edu/galaxy/display/IMAP/SDC+Processing+Architecture+Overview>`_

