Deploy CDK to Personal AWS Account
==================================

1. Get a personal AWS account set up by submitting a `Service Desk ticket <https://servicedesk.lasp.colorado.edu/servicedesk/customer/portals>`_:

   a. Use the request type "Create AWS Account" under the "Cloud" section.
   b. For "Project/Program Name", use ``IMAP``
   c. For "Speedtype", use the speedtype for the "IMAP SOC SDC SW and Sys Devl" project
   d. For "Root Group Email Address to Associate with Account", use your LASP email address
   e. For "Lead Technical Contact Name", use `Greg Lucas`

2. Once the account is set up, `log in to AWS <https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fconsole.aws.amazon.com%2Fconsole%2Fhome%3FhashArgs%3D%2523%26isauthcode%3Dtrue%26state%3DhashArgsFromTB_us-east-2_bdb6cea710bddb8a&client_id=arn%3Aaws%3Asignin%3A%3A%3Aconsole%2Fcanvas&forceMobileApp=0&code_challenge=DSG8NACFeAjbOsDQjFFz6TvuW-ohRykiAIXkuEQriOI&code_challenge_method=SHA-256>`_ and create a user in IAM.

   a. You'll likely want to give your user the ``AdministratorAccess`` policy.

3. Next, go to ``Security Credentials`` for your new user and create an access key. This will be needed in step 6, so save the access key/secret access key.
4. On your local system, add a new profile to ``~/.aws/config`` for your personal account. This can be named anything you want, we'll call it ``my_profile`` for this example.
5. In your local command line, type ``aws configure`` to set up your credentials.
6. Paste your IAM access key and secret access key from step 3.
7. In the IMAP ``sds-data-manager`` repository, add a new context to ``cdk.json``. We'll call it ``my_context`` for this example. For example::

    "backup": {
        "account": "012345678901",
        "source_account": "dev"
    },
    "my_context": {
        "account":"234567890123",
        "region":"us-west-2"
    },
    "dev": {
        "account": "111122223333",
        "domain_name": "example.com",
        "region": "us-west-2"
    },

*Do not put a domain field for your personal context*

8. In your local terminal, go to the top of the ``sds-data-manager`` repository and run the following:

   a. ``cdk bootstrap --profile my_profile --context account_name=my_context``
   b. ``cdk synth --profile my_profile --context account_name=my_context``
   c. ``cdk deploy --profile my_profile --context account_name=my_context``

      * Add ``--require-approval never`` to the end of step 9c if you don't want to be prompted for each stack.
      * Note: Not sure if this happens to anyone else, but when the deployment gets this part of the deployment, there's no prompt and it hangs there forever. However, if I hit ``Enter``, a ``y/n`` prompt appears asking if I want to deploy. So if it gets stuck at that point, hit ``Enter``, then you can select ``y``. Terminal output example::

            0af60606a7e1: Pushed
            03f331fd9b29: Pushed
            03f331fd9b29: Pushed
            c3418b25bdeef0d97b84f41688fa22e9d1c8bdf0927d3774d77588d87763a2f9: digest: sha256:83e8fd9ae28cee020091b2caa4faa421a400505e4ddfdb29fd693dec8b2a7a1d size: 2628
            29143fef993fc62aeb3447a224679eafa9e60eaba00aa3bfaa60f0de9f9815bb: digest: sha256:e3c8f122ade7a0c1f598b3c7bbc08488c694aa9b7279e1367227ed0d0fba6c33 size: 2628

9. Before committing any changes, make sure to revert your ``cdk.json`` file to its original state.
