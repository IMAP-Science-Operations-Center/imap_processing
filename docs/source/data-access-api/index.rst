.. _data-access-api:

Data Access API
===============

The SDC provides a REST API that allows users to upload and download files, as
well as query for file metadata. The following documentation describes the
various endpoints that are supported and how to use them.

The API can be accessed from the following URL: https://api.dev.imap-mission.com


.. openapi:: openapi.yml
   :group:
   :include: /upload/science

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/upload/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "body": "https://sds-data.s3.amazon.com/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf?<credentials-string>"}
   {"statusCode": 400, "body": "A pre-signed URL could not be generated. Please ensure that the file name matches mission file naming conventions."}


.. openapi:: openapi.yml
   :group:
   :include: /download

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" /codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf
   curl -X GET -H "Accept: application/json" imap_codice_l1a_lo_20260101_20260102_v01-01.cdf

**Possible Responses:**

.. code-block:: json

   {"statusCode": 302, "headers": {"Content-Type": "text/html", "Location": "s3://sds-data/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01"}, "body": {"download_url": "s3://sds-data/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01"}}
   {"statusCode": 400, "body": "Not a valid S3 URI.  Example input: s3://bucket/path/file.ext"}


.. openapi:: openapi.yml
   :group:
   :include: /query

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/query?instrument=mag&data_level=l0&descriptor=burst&start_date=20230112&end_date=20230113&version=*&extension=pkts

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": "[('mag', 'l0', 'burst', '20230112', '20230113', 'v01-01', 'pkts'), ('mag', 'l0', 'burst', '20230112', '20230113', 'v01-02', 'pkts')]"}
   {"statusCode": 400, "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"}, "body": "<param> is not a valid query parameter. Valid query parameters are: <valid parameters>"}