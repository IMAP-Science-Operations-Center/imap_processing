Technology Stack
----------------

This page lists the various technologies and libraries that the IMAP SDC
utilizes along with a few notes on what they are used for, why they were chosen,
and what other options were considered. It is worth noting that the development
of the SDC began in ~2022-2023, and so many of the technology decisions reflect
the technological ecosystem of that time.

AWS
~~~

The SDC uses Amazon Web Services (AWS) as the solution for `cloud
infrastructure <https://lasp.colorado.edu/galaxy/display/IMAP/SDC+Architecture>`_
used for data processing. Amongst cloud vendors, AWS was chosen mostly because
of its popularity within scientific software applications. Other cloud vendors
such as Microsoft Azure and Google Cloud were not really considered.

AWS was chosen over an on-prem architecture due to the general movement within
the community to move data processing and storage to the cloud.

The following sections describe the particular cloud services used in the SDC:

CDK
"""


API Gateway
"""""""""""

Lambda
""""""

Batch
"""""

EventBridge
"""""""""""

RDS
"""

`RDS <https://lasp.colorado.edu/galaxy/display/IMAP/SDC+Database+Tables>`_ was
famously chosen after an extremely lengthy discussion going back-and-forth on
pros and cons of various database technologies (OpenSearch and DynamoDB were the
other major considerations). RDS was ultimately chosen for being the "old
reliable" option, as it has been extensively used and is extremely well
documented.

S3
""

cdflib + Xarray
~~~~~~~~~~~~~~~

The Python `xarray <https://docs.xarray.dev/en/stable/>`_ and `cdflib
<https://cdflib.readthedocs.io>`_ libraries are used for creating data
structures for IMAP data and reading/writing those data structures to CDF files,
respectively.  ``cdflib`` was chosen for CDF file I/O because of its convenient
``xarray_to_cdf()`` and ``cdf_to_xarray()`` functions. Additionally, the main
developer for ``cdflib`` (Bryan Harter) is also a developer for the IMAP SDC,
and so there is a lot of in-house knowledge of the library. ``xarray`` was
chosen for its support of data structures that closely match the format of CDF
files (i.e. use of data variables, data attributes, time coordinates, etc.).


Common Data Format
~~~~~~~~~~~~~~~~~~

The Common Data Format (CDF) was selected as the file format for IMAP data from
requirements. CDF is a widely used data format within the Heliophsyics
community. This decision was based purely on `requirements
<https://lasp.colorado.edu/galaxy/display/IMAP/IMAP+SDC+to+Instrument+Team+ICD#IMAPSDCtoInstrumentTeamICD-1.3FormatStandards>`_.
As such, no other data formats were considered.

Docker
~~~~~~

GitHub
~~~~~~

The SDC uses `GitHub <https://github.com/IMAP-Science-Operations-Center>`_ for
version controlling its software, as well as keeping track of development tasks
(i.e. GitHub Issues) and progress (i.e. GitHub Projects), and performing code
reviews. GitHub was chosen over other solutions like GitLab and Bitbucket mainly
for its collaborative features and unlimited free public repositories. As the
IMAP SDC strives to comply with the `NASA SMD SPD-41a policies
<https://smd-cms.nasa.gov/wp-content/uploads/2023/08/smd-information-policy-spd-41a.pdf>`_,
(adopted in 2022) this open-source collaborative GitHub solution made the most
sense.

Poetry
~~~~~~


Pytest
~~~~~~

The `pytest <https://docs.pytest.org>`_ python library was chosen for writing
and executing unit tests. This library was chosen because it has a large and
active community of users, is well documented, and has several features that
allow for a robust testing framework (e.g. fixtures, parametrized testing,
object mocking, etc.). There are other tools/libraries for unit testing in
Python (e.g. ``unittest``, ``nose``, etc.), but those were not really
considered.


Python
~~~~~~

The SDC uses Python as the primary programming language for the implementation
of the vast majority of the system and its ancillary tools. A few other
languages are supported for Level 3 `algorithm development containers
<https://github.com/IMAP-Science-Operations-Center/imap_matlab_processing_example>`_,
and a lot of the SDC cloud infrastructure is built with AWS CDK, but otherwise
everything is written in Python. Python was chosen mostly because at the time of
development, it was the most widely used and supported language for scientific
software development, and the main programming language used within the
Data Systems group at LASP.


Space Packet Parser
~~~~~~~~~~~~~~~~~~~

The SDC uses the `space_packet_parser
<https://space-packet-parser.readthedocs.io/en/stable/>`_ library for the
decommutation of CCSDS packets and processing L0-level data. This library was
chosen for its support of XTCE format for telemetry definitions. Another benefit
to using this library is that it was developed and is actively maintained
in-house here in Data Systems at LASP (Gavin Medley), and so the library can be
updated to help meet the needs of the IMAP SDC. The other option for packet
decommutation would have been `CCSDSPy <https://docs.ccsdspy.org/en/latest/>`_.

Sphinx + ReadTheDocs
~~~~~~~~~~~~~~~~~~~~

The SDC uses the Python `sphinx <https://www.sphinx-doc.org/en/master/>`_
library for generating project documentation as well as reference documentation
that gets automatically rendered from code docstrings. `ReadTheDocs
<https://about.readthedocs.com/?ref=readthedocs.com>`_ is used to host the
documentation via the HTML files that ``sphinx`` generates. Together, these
tools product the 'official' `IMAP SDC documentation
<https://imap-processing.readthedocs.io>`_.

These tools were chosen because they are widely used and have great integration
with Poetry and GitHub.

Sqlalchemy
~~~~~~~~~~
