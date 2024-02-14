.. _data-access-api:

Data Access API
===============

The SDC provides a REST API that allows users to upload and download files, as
well as query for file metadata. The following documentation describes the
various endpoints that are supported and how to use them.

The API can be accessed from the following URL: https://api.dev.imap-mission.com

The `imap-data-access <https://github.com/IMAP-Science-Operations-Center/imap-data-access>`_
repository also provides programmatic access and a command-line utility for
interacting with the API.


.. openapi:: openapi.yml
   :group:
   :include: /upload

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