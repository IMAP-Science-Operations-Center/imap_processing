.. _sdc-api:

Data Access API
===============

The SDC provides a REST API that allows users to upload and download files, as
well as query for file metadata. The following documentation describes the
various endpoints that are supported and how to use them.

The API can be accessed from the following URL: https://api.dev.imap-mission.com


.. openapi:: openapi.yml
   :group:
   :include: /upload

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/upload/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "body": "https://sds-data.s3.amazon.com/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf?<credentials-string>"}
   {"statusCode": 400, "body": "A pre-signed URL could not be generated. Please ensure that the file name matches mission file naming conventions."}


.. openapi:: openapi.yml
   :group:
   :include: /download

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/download?s3_uri=s3://sds-data/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01.cdf

**Possible Responses:**

.. code-block:: json

   {"statusCode": 302, "body": {"download_url": "s3://sds-data/imap/codice/l1a/2026/01/imap_codice_l1a_lo_20260101_20260102_v01-01"}}
   {"statusCode": 400, "body": "Not a valid S3 URI.  Example input: s3://bucket/path/file.ext"}


.. openapi:: openapi.yml
   :group:
   :include: /query

**Example Usage:**

.. code-block:: bash

   curl -X GET -H "Accept: application/json" https://api.dev.imap-mission.com/query?mission=imap&level=l0&instrument=mag&date=20230112&version=*&extension=pkts

**Possible Responses:**

.. code-block:: json

   {"statusCode": 200, "body": {"mission": "imap", "level": "l0", "instrument": "mag", "date": "20230112", "version": "*", "extension": "pkts"}}
