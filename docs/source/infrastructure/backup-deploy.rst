Deploying to Backup Account
===========================

Some of the CDK code is used to deploy a backup bucket for S3 replication. This page covers how to deploy this code into a different account. This backup code can be deployed into the normal dev account as well.

Steps for deploying
^^^^^^^^^^^^^^^^^^^
.. note::
    First, you will need to deploy the SdsDataManager stack to the source account (dev or prod). The steps to set up and deploy that are in the :doc:`cdk-deployment` doc.

#. Set up your backup account credentials if you're using another account. This can be done by adding a new profile to AWS (i.e. ``imap-backup``)
#. Set your ``AWS_PROFILE`` environment variable to the backup account profile
#. Set the ``source_account`` variable in the "backup" section of the cdk.json file to the account number for the _source_ bucket (where the backed up items originate)
#. Update the ``account_name`` variable in the ``cdk.json`` file to "backup"
#. Finally, you will need to copy the Role arn deployed in SdsDataManager into :file:`sds_data_manager/stacks/backup_bucket_construct.py`. This arn can be found by going to the IAM console in the source account and searching for "BackupRole".
#. Run your ``cdk bootstrap`` command if you haven't already done so for the backup account.
#. Run ``cdk synth --context account_name=backup`` and check to confirm that the only stack being deployed is the "BackupBucket" stack
#. Run ``cdk deploy --context account_name=backup`` to deploy!

.. note::
    Once the Backup bucket is deployed to the other account, you still need to set up :doc:`s3-replication` in the source account.
