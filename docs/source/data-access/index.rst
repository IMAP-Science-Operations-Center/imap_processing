.. _data-access-api:

Data Access
===========

The `imap-data-access <https://github.com/IMAP-Science-Operations-Center/imap-data-access>`_
repository provides programmatic access and a command-line utility for
interacting with the API. It is the preferred way to use the API.

The SDC provides a REST API that allows users to upload and download files, as
well as query for file metadata. The following documentation describes the
various endpoints that are supported and how to use them.

*Note: Several sections and links begin with* [WIP]. *As development on the API is ongoing, this indicates
that the full implementation of the functionality is yet to be completed.*

The API can be accessed from the following URL [WIP]: https://api.dev.imap-mission.com

.. openapi:: openapi.yml
   :group:
   :include: /upload

When uploading files to the API, ensure these files are stored properly in a ``data`` directory. Then,
ensure your working directory is one level above the ``data`` directory in order to properly upload files.

[WIP] Certain ancillary files can also be uploaded to the API. For more specific information regarding these files, visit
`Ancillary Files <https://imap-processing.readthedocs.io/en/latest/data-access-api/calibration-files.html>`_

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/upload/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "body": "https://sds-data-<aws_account_number>.s3.amazon.com/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts?<credentials-string>"}
   {"statusCode": 400, "body": "Invalid filename. Expected - <mission>_<instrument>_<datalevel>_<descriptor>_<startdate>_<enddate>_<version>.<extension>"}
   {"statusCode": 400, "body": "Invalid mission."}
   {"statusCode": 400, "body": "Invalid instrument. Please choose from {'ultra-45', 'codice', 'glows', 'hit', 'lo', 'mag', 'swe', 'hi-45', 'idex', 'ultra-90', 'hi-90', 'swapi'}"}
   {"statusCode": 400, "body": "Invalid data level. Please choose from {'l0', 'l1', 'l1a', 'l1b', 'l1c', 'l1d', 'l2'}"}
   {"statusCode": 400, "body": "Invalid start date format. Please use YYYYMMDD format."}
   {"statusCode": 400, "body": "Invalid end date format. Please use YYYYMMDD format."}
   {"statusCode": 400, "body": "Invalid version format. Please use vxx-xx format."}
   {"statusCode": 400, "body": "Invalid extension. Extension should be pkts for data level l0 and cdf for data level higher than l0"}
   {"statusCode": 409, "body": "https://sds-data-<aws_account_number>.s3.amazon.com/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts already exists."}


.. openapi:: openapi.yml
   :group:
   :include: /download

It is important to note that your working directory will be established as the default directory. I.e, the ``data``
directory--which files are downloaded to--will automatically be placed in this file path. Choose your working directory
accordingly to suit your desires.

When downloading a file from the API, different folders within the ``data`` directory will be made to better
organize the files. See the example file path: ``data/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts``

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/download/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts

**Possible Responses:**

.. code-block:: json

   {"statusCode": 302, "headers": {"Content-Type": "text/html", "Location": "s3://sds-data/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01"}, "body": {"download_url": "s3://sds-data/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01"}}
   {"statusCode": 400, "body": "No file requested for download. Please provide a filename in the path. Eg. /download/path/to/file/filename.pkts"}
   {"statusCode": 404, "body": "File not found, make sure you include the full path to the file in the request, e.g. /download/path/to/file/filename.pkts"}


.. openapi:: openapi.yml
   :group:
   :include: /query

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/query?instrument=swe&data_level=l0&descriptor=sci&start_date=20240105&end_date=20240105&extension=pkts

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": [{"file_path": "imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-05.pkts", "instrument": "swe", "data_level": "l0", "descriptor": "sci", "start_date": "20240105", "end_date": "20240105", "version": "v00-05", "extension": "pkts"}]}
   {"statusCode": 400, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": "<param> is not a valid query parameter. Valid query parameters are: ['file_path', 'instrument', 'data_level', 'descriptor', 'start_date', 'end_date', 'version', 'extension']"}

Other pages
===========

.. toctree::
    :maxdepth: 1

    calibration-files
    data-dependency