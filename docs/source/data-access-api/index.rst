.. _data-access-api:

Data Access API
===============

The `imap-data-access <https://github.com/IMAP-Science-Operations-Center/imap-data-access>`_
repository provides a command-line utility and python package for
interacting with the API programmatically. It is the preferred way to use the API.

Users may also download, upload, and query via the REST API directly through the browser, or via `curl` commands.
The `REST API Specification`_ section describes the various endpoints that are supported, and how to use them.

*Note: Several sections and links begin with* [WIP]. *As development on the API is ongoing, this indicates
that the full implementation of the functionality is yet to be completed.*

The API can be accessed from the following URL [WIP]: https://api.dev.imap-mission.com

Command Line Utility
--------------------
To Install
^^^^^^^^^^

Run the following command to use the API CLI:

    .. code-block:: bash

        pip install imap-data-access

Base Command Arguments
^^^^^^^^^^^^^^^^^^^^^^

The following are base command arguments for the CLI:

    .. code-block:: bash

        imap-data-access -h # or
        imap-data-access query # or
        imap-data-access download # or
        imap-data-access upload

Add the -h flag with any base command for more information on use and functionality.

Query
^^^^^

To query for files, you can use several parameters: ``--instrument``, ``--data-level``, ``--descriptor``, etc.

Further information is found in in the ``query -h`` menu. You can use parameters alone, or in combination.

**Example Usage:**

    .. code-block:: bash

        imap-data-access query --start-date 20240101 --end-date 20241231 --output-format json
        # The following line is returned:
        [{'file_path': 'imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_v001.pkts', 'instrument': 'swe',
            'data_level': 'l0', 'descriptor': 'sci', 'start_date': '20240105', 'version': 'v001', 'extension': 'pkts'},
        {'file_path': 'imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_v001.pkts', 'instrument': 'swe',
            'data_level': 'l0', 'descriptor': 'sci', 'start_date': '20240105', 'version': 'v001', 'extension': 'pkts'}]

Download
^^^^^^^^

To download files using the CLI tool, use the command ``download``. The downloaded files will be placed in a ``data`` directory.

It is important to note that your working directory will be established as the default directory. I.e, the ``data``
directory will automatically be placed in this file path. Choose your working directory
accordingly to suit your desires.

When downloading a file from the API, different folders within the ``data`` directory will be made to better
organize the downloaded files. See the example path: ``data/imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_20240105_v00-01.pkts``.
The ``data`` directory and its structure is further described here: `Data Directory`_

**Example Usage:**

    .. code-block:: bash

        imap-data-access download imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_v001.pkts


Upload
^^^^^^

Similarly, files can be uploaded to the API using the command ``upload``.

When uploading files to the API, ensure these files are stored properly in a ``data`` directory (see the `Data Directory`_ section below for more information). Then,
ensure your working directory is one level above ``data`` in order to properly upload files.

[WIP] Certain ancillary files can also be uploaded to the API. For more specific information regarding these files, visit
`Ancillary Files <https://imap-processing.readthedocs.io/en/latest/data-access-api/calibration-files.html>`_

**Example Usage:**

    .. code-block:: bash

        imap-data-access upload /imap/swe/l1a/2024/01/imap_swe_l1a_sci_20240105_v001.cdf

Importing as a package
----------------------
Imap data access can also be imported and used as a python package.

**Example Usage:**

    .. code-block:: bash

        import imap_data_access

        # Search for files
        results = imap_data_access.query(instrument="mag", data_level="l0")
        # results is a list of dictionaries
        # [{'file_path': 'imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_v001.pkts', 'instrument': 'swe',
                'data_level': 'l0', 'descriptor': 'sci', 'start_date': '20240105','version': 'v001', 'extension': 'pkts'},
            {'file_path': 'imap/swe/l0/2024/01/imap_swe_l0_sci_20240105_v001.pkts', 'instrument': 'swe',
                'data_level': 'l0', 'descriptor': 'sci', 'start_date': '20240105', 'version': 'v001', 'extension': 'pkts'}]

        # Download a file that was returned from the search
        imap_data_access.download("imap/mag/l0/2024/01/imap_mag_l0_raw_202040101_v001.pkts")

        # Upload a calibration file that exists locally
        imap_data_access.upload("imap/swe/l1a/2024/01/imap_swe_l1a_sci_20240105_v001.cdf")

Configuration
--------------

.. _data-directory:

Data Directory
^^^^^^^^^^^^^^

The folder structure for data files within the IMAP SDC is rigidly defined, so the data access api will mimic that structure to make sure all data is stored in the same hierarchical structure as the SDC. This will enable seamless transition between a user's local system and the SDC. This is only used for downloads.
A user's root data location can be specified as an environment variable ``IMAP_DATA_DIR`` or through a configuration dictionary within the package itself (``imap_data_access.config["DATA_DIR"]``). If the ``IMAP_DATA_DIR`` variable is not set, the program defaults to the user's current working directory + ``data/``.
The following is the directory structure the IMAP SDC uses.

    .. code-block:: bash

        <IMAP_DATA_DIR>/
          imap/
            <instrument>/
              <data_level>/
                <year>/
                  <month>/
                    <filename>

for example, with ``IMAP_DATA_DIR=/data:``

    .. code-block:: bash

        /data/
          imap/
            swe/
              l0/
                2024/
                  01/
                    imap_swe_l0_sci_20240105_v001.pkts

Data Access URL
^^^^^^^^^^^^^^^

To change the default URL that the package accesses, you can set the environment variable ``IMAP_DATA_ACCESS_URL`` or within the package ``imap_data_access.config["DATA_ACCESS_URL"]``. The default is the development server (``https://api.dev.imap-mission.com``).

File Validation
---------------

This package validates filenames and paths to check they follow our standards, as defined by the `filename conventions <https://imap-processing.readthedocs.io/en/latest/development-guide/style-guide/naming-conventions.html>`_. There is also a class available for use by other packages to create filepaths and filenames that follow the IMAP SDC conventions.
To use this class, use ``imap_data_access.ScienceFilePath``.

Usage:

    .. code-block:: bash

        science_file = imap_data_access.ScienceFilePath("imap_swe_l0_sci_20240101_v001.pkts")

        # Filepath = /imap/swe/l0/2024/01/imap_swe_l0_sci_20240101_v001.pkts
        filepath = science_file.construct_path()

Troubleshooting
---------------
Network Issues
^^^^^^^^^^^^^^

**SSL**

If you encounter SSL errors similar to the following:

    .. code-block:: bash

        urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)>

That generally means the Python environment you're using is not finding your system's root certificates properly. This means you need to tell Python how to find those certificates with the following potential solutions.

#. Upgrade the certifi package

    .. code-block:: bash

            pip install --upgrade certifi

#. Install system certificates -- Depending on the Python version you installed the program with, the command will look something like this:

    .. code-block:: bash

            /Applications/Python\ 3.10/Install\ Certificates.command

**HTTP Error 502: BadGateway**

This could mean that the service is temporarily down. If you continue to encounter this, reach out to the IMAP SDC at imap-sdc@lasp.colorado.edu.

FileNotFoundError
^^^^^^^^^^^^^^^^^

This could mean that the local data directory is not set up with the same paths as the SDC. See the `Data Directory`_ section for an example of how to set this up.

.. _rest-api-specification:

REST API Specification
----------------------
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

Other pages
-----------

.. toctree::
    :maxdepth: 1

    calibration-files