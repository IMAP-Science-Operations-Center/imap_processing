AWS Setup
=========

Download/requirements
~~~~~~~~~~~~~~~~~~~~~

Ensure you have installed nodejs (newer than version 14), AWS CLI, and Docker

- `nodejs <https://nodejs.org/en/download/>`_
- `AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
- `Docker <https://docs.docker.com/get-docker/>`_

.. _aws-new-user:

New User
~~~~~~~~~

#. Log-in to `AWS console <https://aws.amazon.com/console/>`_
#. Use *imap-sdc-development* for Account ID
#. Enter user name and password
#. Select IAM from *Services* menu or search IAM
#. Select *Users* from left menu
#. Click your user button
#. Select *Security Credentials* tab
#. Click *Create Access Key*
#. Make note of *Access Key ID* and *Secret Access Key*
#. You can download the .csv file if you want
#. Click *Close*

Existing User
~~~~~~~~~~~~~
#. In command line, run the following command to set up the aws environment::

    aws configure

#. Enter your *Access Key ID* and *Secret Access Key* (If you don't have them, see :ref:`aws-new-user`)
#. For region, set it to the AWS region you'd like to set up your SDS. For IMAP, we're using "us-west-2"::

    [imap]
    region=us-west-2
    aws_access_key_id=<Access Key>
    aws_secret_access_key=<Secret Key>

#. Then, you can set the profile used by cdk by setting the AWS_PROFILE environment variable to the profile name (in this case, imap)::

    export AWS_PROFILE=imap

