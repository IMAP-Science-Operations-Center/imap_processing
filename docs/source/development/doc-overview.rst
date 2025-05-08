Contributing to Documentation
=============================

Using Sphinx
------------
IMAP uses `Sphinx <https://www.sphinx-doc.org/en/master/index.html>`_ to generate and maintain documentation. Sphinx allows for automatic usage of Python docstrings, as well features for hierarchical structure, automatic code highlighting, and different output formats.

To add a new documentation file, create a new ``.rst`` file under :file:`docs/source`. RST stands for Restructured Text, which is a type of rich text format used by Sphinx. It is a flexible and powerful format which allows for lots of features which are useful in documentation.

For more information on restructured text, Sphinx has `extensive documentation <https://www.sphinx-doc.org/en/master/usage/index.html>`_.

Documentation Style
-------------------
Here are the main rules for documentation in IMAP:

#. :ref:`doc-rule-short`
#. :ref:`doc-rule-link`
#. :ref:`doc-rule-english`
#. :ref:`doc-rule-organize`

.. _doc-rule-short:

Keep it short and simple
^^^^^^^^^^^^^^^^^^^^^^^^
Keep documentation to the minimal length that is required. You want to keep the documentation to the point so it is easy to link to specific topics. This prevents duplicate info, which helps keep documentation clear and up to date. In general, if there's doubt that a particular section is directly relevant to the overall page, split it out to its own page and link it. Additionally, if the section might be useful to other documents, please put it into its own document.

.. _doc-rule-link:

Link to other docs, don't rewrite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before writing any section, double check that there isn't an existing document that covers that info. If we only keep the information written down once, it can easily be updated, deprecated, or shared without having to comb through other documents.

You can link to other documents in these docs using the `"docs" role <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-doc>`_.

Basically, you can add a link using this syntax: ``:doc:`parrot```.

It is also useful to link to specific sections within documentation. If you have sections which may be useful to link to, you can add an internal `ref <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-ref>`_ to allow other documents to link to your section. You can create this ref with ``.. _hyperlink-name:`` before the referenced text. You can use these internal links with the `ref <https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html#role-ref>`_ role:

::

    .. _my-reference-label:

    Section to cross-reference
    --------------------------

    This is the text of the section.

    It refers to the section itself, see ``:ref:`my-reference-label`.``

These reference names should be unique across all documentation. You can also create references to figures or tables.

.. _doc-rule-english:

Write in clear english
^^^^^^^^^^^^^^^^^^^^^^^
To allow for everyone to understand documentation, write clearly and informally. Obviously technical documentation often uses jargon or acronyms, but where it is possible, please try and keep the documentatation simple and well-formatted.

.. tip::
    Noting particularly important pieces of information in special formats, like this tip box, helps with formatting and clarity.

.. _doc-rule-organize:

Use document hierarchy to organize information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Documents should be organized under larger sections for easier access of information. If you are writing a document, and the section doesn't yet exist, create the section. Each section should have its own folder under :file:`docs/source` and include a landing page (named index.rst) for the overall section which includes general information, key links, and a `toctree <https://www.sphinx-doc.org/en/master/usage/quickstart.html#defining-document-structure>`_ directive. This directive should include all the files in the section. Finally, the index file for the section should be added to the main :doc:`index` page.

.. note::
    This document overview also acts as a reference file for restructured text. You can use it as an example for some of the basic formatting tools in RST.