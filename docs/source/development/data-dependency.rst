Data Dependency Management
==========================

The IMAP Science Data Center (SDC) utilizes an event-based processing system that allows for
processing as soon as data is available. This system is designed to be flexible to
accommodate the various requirements and inter-dependencies for all 10 instruments.

As part of our requirements, we need some way to explicitly describe the dependencies
for each file. We also need to be able to flexibly update the dependencies on a regular
basis, to accommodate changing requirements.

Overview
--------

When a file lands in the SDC, it is added to our data bucket (Also called S3 or S3 bucket.)
This bucket, as the name implies, is a simple collection which contains all the files in the
SDC, organized like a file system.

Each data file is put into a specific subfolder depending on the file name. For example,
a file named ``imap_swe_l0_sci_20240105_20240105_v00-01.pkts`` would be placed in the
``imap/swe/l0/2024/01`` folder. More information about the naming conventions can be
found in :ref:`naming-conventions`.

When a file of any level arrives in the bucket, it triggers the rest of processing. This is
how we manage file processing within the SDC, rather than waiting until all files have arrived
or running at particular times of day. This allows us to quickly process data as soon as all the
required pieces are available to us, and create a flexible system which can easily be updated
to add exceptions or new requirements on a per-instrument or per-level basis.

.. note::
    This document, and our tooling, uses the terms "upstream dependencies" and
    "downstream dependencies" to describe the relationships between files. A
    "downstream dependency" for a given file means that the current file is required for
    processing of the downstream files. For example, an L2 file is a downstream dependency
    of an L1 file. An "upstream dependency" is the opposite, describing a file which is required
    to begin processing the current file. For example, an L1 file is an upstream dependency of an
    L2 file.

Detailed Description of File Processing
---------------------------------------

For explicit descriptions of the tools and technical choices of the IMAP SDC, please refer to
`this Galaxy page <https://lasp.colorado.edu/galaxy/display/IMAP/SDC+Architecture>`_.
This section is intended to act as a high level overview for the data processing architecture of
the IMAP SDC, in less technical terms.

.. image:: ../_static/architecture_overview.png

`Up to date overview chart in Galaxy <https://lasp.colorado.edu/galaxy/display/IMAP/SDC+Processing+Architecture+Overview>`_

Each science file that arrives is treated the same, regardless of level or instrument. When a file
is placed in the file storage system, it triggers a step to index the file ("indexer lambda").
This step adds the file to the database and triggers the next step in processing ("IMAP SDC job handler system").

After indexing, the job handler is triggered in order to determine what jobs may be ready for processing.
For each file that arrives, the system checks to see what job may need to be run by looking
at the downstream dependencies are.For example, if a MAG L1A file arrived, this step would
determine that the MAG L1B ``mago`` and ``magi`` files are dependent on
the L1A file, and therefore MAG L1B may be ready to begin processing.

Then, for each possible job, the job handler process checks to see if all the upstream
dependencies are met. Although we know we have one of the upstream dependencies for an
expected job, it's possible that there are other required dependencies that have not yet
arrived. If we are missing any required dependencies, then the system does not kick off the
processing job. When the missing upstream dependency arrives, it will trigger the same process of checking
for all upstream dependencies. This time all required dependencies will be found and the
processing job will be started.

The upstream look up system will determine if it has a complete list of dependencies.
Several scenarios can cause the dependency list to be incomplete. Missing files in the database
represent the primary cause. Anomalies such as Loss of Orientation Insertion (LOI) or Trajectory
Correction Maneuver (TCM) events, as well as solar wind conditions, may also result in incomplete
dependencies, though support for these scenarios is not yet implemented. Similarly, delays in
repoint data or downlink delays can cause incompleteness, but handling for these cases is also
planned for future implementation. Additionally, if any required dependencies are missing or if a
job is still in progress, the dependency list cannot be considered complete.

For example, SWAPI L3 requires both SWAPI L2 files and MAG L1D (previously called L2pre)
files. The SWAPI L2 job and the MAG L1D job are run independently, so there is no guarantee
that they will finish at the same time. Let's assume that the MAG L1D job finishes first,
since it is the lower level. When that file arrives, one of the downstream dependencies is
going to be the SWAPI L3 processing. However, when job handler checks the upstream
dependencies for SWAPI L3, it will find that SWAPI L2 is missing. Therefore, processing
won't start. Once the SWAPI L2 processing finishes, and the SWAPI L2 file arrives, the job
handler is triggered with that file. Once again, SWAPI L3 is a downstream dependency, but
this time, both upstream dependencies for SWAPI L2 are present. Therefore, processing for
SWAPI L3 can begin.

The status of each job is recorded in the status tracking table as "in progress", "complete",
or "failed." Through this, we can track processing for specific files and determine if a
file exists quickly.

Dependency Config File
----------------------

How does the SDC track which files are dependent on others? In order to decide what the
downstream or upstream dependencies of a file are, and what the nature of those dependencies
are, we need some way to request the upstream or downstream dependencies of a given file.
The current dependencies between instruments are recorded in `sds-data-manager Repo
<https://github.com/IMAP-Science-Operations-Center/sds-data-manager/tree/dagster/sds_data_manager/orchestration/dependencies>`_.

We handle and track dependencies using a YAML config file that acts like a database. This YAML
config file expects a specific format, and is used to determine the upstream and downstream
dependencies of each product.

Each product has entry for inputs, outputs and partition. The inputs are list of inputs needed to start
processing the product, and outputs are the products that will be produced after processing. Partition
specifies the partitioning scheme for the product, which is used to determine the date range of
the data of the file produced from the processing job. For example, if the partition is daily,
then the file produced from the processing job will have a date range of 1 day. If the partition
is repoint, then the file produced from the processing job will have a date range of 1 pointing.

Filename convention
~~~~~~~~~~~~~~~~~~~~
imap_<instrument>_dependencies.yaml


Dependency Types
~~~~~~~~~~~~~~~~~

The YAML config file stores the upstream dependencies for each data product. This information
is used across all instruments to determine both upstream and downstream relationships:

**UPSTREAM**
An upstream dependency is a file required to begin processing the current product.
The dependency config file explicitly defines these upstream dependencies for each data product.

**DOWNSTREAM**
A downstream dependency is a product whose processing depends on the current file.
Downstream dependencies are determined at runtime by querying which products list the current
file as an upstream dependency.

Valid Fields for Dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _imap-data-init: https://github.com/IMAP-Science-Operations-Center/imap-data-access/blob/main/imap_data_access/__init__.py
.. _imap-data-validation: https://github.com/IMAP-Science-Operations-Center/imap-data-access/blob/main/imap_data_access/file_validation.py

Source
^^^^^^^^^^^^^^^

Source can be one of the following:

- IMAP instrument name listed in the ``VALID_INSTRUMENTS`` dictionary in this file:
  `imap-data-access Repo <imap-data-init_>`_

- SPICE data type listed in the ``_SPICE_DIR_MAPPING`` dictionary in this file:
  `imap-data-access validation file <imap-data-validation_>`_


Data Type
^^^^^^^^^^^^^^^^^^

Data type can be one of the following:

- IMAP data level listed in the ``VALID_DATALEVELS`` dictionary in this file:
  `imap-data-access Repo <imap-data-init_>`_

- ``spice``

- ``spin``

- ``repoint``

- ``ancillary``


Descriptor
^^^^^^^^^^^^^^^^^^^^^

Descriptor can be one of the following:

- For science or ancillary data, the descriptors are defined by the instrument and SDC.

- For ``spice`` data types, ``historical``, and ``best`` are the valid descriptors.

- For ``spin`` and ``repoint`` data types, ``historical`` is the only valid descriptor.

Required (Optional)
^^^^^^^^^^^^^^^^^^^

**Default:** ``true``

Specifies whether this upstream dependency must be available before a processing
job can begin.
If set to true, the product cannot be processed until this dependency is available.
If set to false, the product can be processed even if this dependency is missing.


Trigger_job (Optional)
^^^^^^^^^^^^^^^^^^^^^^

**Default:** ``true``

Whether the arrival of this upstream dependency should trigger a processing job.
For most cases, this is true, meaning that when this file arrives, it will trigger job.

At the same time, there are cases where we do not want to start a job when certain upstream data arrives.
For example, upstream inputs such as spacecraft clock or leapseconds data should not change
frequently, and processing jobs should not be triggered when these files are updated.
Setting this to false allows for more controlled processing and may require additional
review before updating these types of dependencies.

[Past_days, Future_days] (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Default:**

- ENA and GLOWS: ``[0p, 0p]``
- Rest of in-situ instruments: ``[0d, 0d]``

Most science files are produced daily or per pointing. Due to this cadence, the default is
daily for most in-situ instruments and per pointing for ENA and GLOWS instruments. However,
this feature provides flexibility to query for upstream data beyond the daily date range of
the current product.

Supported values for past_days and future_days fields:

- ``p`` - pointing
- ``h`` - hourly
- ``d`` - days
- ``l`` - last_processed
- ``nd``- nearest day
- ``np``- nearest pointing

Days can be used to support longer durations and different cadences. For example, weekly
processing can use 7 days, and yearly processing can use 365 days.


File content structure
~~~~~~~~~~~~~~~~~~~~~~
The YAML config has the following structure:

.. code-block:: yaml

   (level, descriptor):
     partition: daily
     inputs:
       - source,
         data_type,
         descriptor,
         required(bool),
         trigger_job(bool),
         [past_days, future_days]
       - source,
         data_type,
         descriptor,
         required(bool),
         trigger_job(bool),
         [past_days, future_days]
       ....
     outputs:
       - source,
         data_type,
         descriptor
       - source,
         data_type,
         descriptor
       ....

File content Example
~~~~~~~~~~~~~~~~~~~~~~

**imap_hit_dependencies.yaml**

.. code-block:: yaml

  spice_basics: &spice_basics
    - source: leapseconds
      data_type: spice
      descriptor: historical
      trigger_job: false
    - source: spacecraft_clock
      data_type: spice
      descriptor: historical
      trigger_job: false

  l0_data: &l0_data
    - source: hit
      data_type: l0
      descriptor: raw

  (l1a, all):
    partition: daily
    inputs:
      - *spice_basics
      - *l0_data
    outputs:
      - source: hit
        data_type: l1a
        descriptor: counts-standard
      - source: hit
        data_type: l1a
        descriptor: counts-sectored
      - source: hit
        data_type: l1a
        descriptor: direct-events

  (l1b, hk):
    partition: daily
    inputs:
      - *spice_basics
      - *l0_data
    outputs:
      - source: hit
        data_type: l1b
        descriptor: hk

**imap_hi_dependencies.yaml**

.. code-block:: yaml

  spice_basic: &spice_basic
    - source: leapseconds
      data_type: spice
      descriptor: historical
      trigger_job: false
    - source: spacecraft_clock
      data_type: spice
      descriptor: historical
      trigger_job: false

  (l1b, 45sensor-goodtimes):
    partition: repoint
    inputs:
      - *spice_basic
      - source: repoint
        data_type: repoint
        descriptor: historical
        trigger_job: false
      - source: hi
        data_type: ancillary
        descriptor: 45sensor-cal-prod
      - source: hi
        data_type: l1a
        descriptor: 45sensor-diagfee
      - source: hi
        data_type: l1b
        descriptor: 45sensor-de
        date_range: ["8np",]
      - source: hi
        data_type: l1b
        descriptor: 45sensor-hk
    outputs:
      - source: hi
        data_type: l1b
        descriptor: 45sensor-goodtimes
