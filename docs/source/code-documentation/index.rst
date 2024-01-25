.. _code-documentation:

Code Documentation
==================

.. currentmodule:: imap_processing

This page gives an overview of the routines within the imap_processing package.
The package is organized by instrument, meaning each instrument is its own
module and has submodules for each processing step.

Instruments
-----------

.. toctree::
   :maxdepth: 1

   codice
   glows
   hit
   swe
   hi
   lo
   swapi

Utility functions can be found in modules within the top package level.

Processing
----------

To process an instrument, a command line utility is installed with the
package.  The command line utility is called ``imap_cli`` and
takes the instrument and level as arguments.  For example, to process
the CODICE instrument at level 1, the command would be

.. code:: text

    imap_cli --instrument codice --level 1

This will write output files to the default location, which is
the current working directory + "/imap-data". To change the data
directory, use the ``--data-dir`` option, or the environment
variable ``IMAP_DATA_DIR``. For example to use a temporary directory

.. code:: text

      imap_cli --instrument codice --level 1 --data-dir /tmp/imap-data
      # or equivalently with an environment variable
      IMAP_DATA_DIR=/tmp/imap-data imap_cli --instrument codice --level 1

Tools
-----

.. toctree::
   :maxdepth: 2

   tools/index