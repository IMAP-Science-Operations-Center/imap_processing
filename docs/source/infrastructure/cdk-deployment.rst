.. _cdk-deployment:

CDK Deployment
==============

.. note::
    In most cases, deployments should be performed via GitHub Actions rather
    than manually. Manual deployment is intended for special cases only.

Deploy
~~~~~~
The deploy configuration is controlled through the ``cdk.json`` file.
Each account has a name and the associated configuration parameters associated with it.
For example::

    "dev": {
        "account": "123456789012",
        "domain_name": "website.com",
        "region": "us-west-2"
    }

The parameters in this section are what control the deployment of the application.
The account and region are the environment to deploy the stacks into, and the
``domain_name`` is if you want to deploy it into a hosted domain name that you control.
Optionally, if you want to deploy to a new account or different subdomain you can
add an entirely new section with appropriate parameters.

To deploy the application, you must have local credentials to be able to deploy to
the specified account and region. This can be done through environment variables
and AWS access key credentials::

        export AWS_PROFILE=<profile>

You'll then need to synthesize the CDK code with the command::

        cdk synth --context account_name="dev"

Then you can deploy the architecture with the following command::

    cdk deploy --context account_name="dev" [ stack | --all ]

After about 20-30 minutes or so, you should have a brand-new SDS set up in AWS.
The cloud infrastructure CDK source for the IMAP mission is maintained in the
``sds-data-manager`` repository; this page is mirrored in ``imap_processing``
for documentation and reference only.

Cleanup Resources
~~~~~~~~~~~~~~~~~
During development, if you are creating resources that aren't intended to be shared,
and you do not intend to use the resources for more than a couple of days you should
do a destroy to avoid charges, especially with databases.::

        cdk destroy --context account_name="dev" [ stack | --all ]

Automatic Deployments
~~~~~~~~~~~~~~~~~~~~~
The CDK code is automatically deployed to the dev account when a pull request is
merged into the dev branch. This is done through a GitHub Action that runs
the ``cdk deploy`` command. The GitHub Action is defined in the
``.github/workflows/deploy.yml`` file. It uses short lived credentials to deploy
the code to the dev account. The credentials are obtained through an OIDC
authentication process and an aws provided credentials action step.

With a new account, there are several steps that are required to set this up
and authorize GitHub to deploy to the account. A summary link to the steps
is given here: https://github.com/aws-actions/configure-aws-credentials#OIDC
with the explicit steps as follows:

#. Setup GitHub as an OIDC provider in your AWS account. This can be done
   manually via the console or through a cloudformation template. These links give
   a step-by-step process that you can follow.
   https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services#adding-the-identity-provider-to-aws
   https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html
#. Add a role and trust policy in AWS IAM that allows GitHub to assume the role.
   https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services#configuring-the-role-and-trust-policy
   Specifically, make sure you restrict this role to only be invoked on the given
   branches of your repository and not wide-open trust to the entire organization.
#. Attach a policy to the role that allows the role to deploy the CDK stacks, giving
   it the appropriate permissions for the resources you require.

Troubleshooting Lambda Architectures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project can be built on a local machine, but creating the layers can
sometimes cause issues if a developer is running on an ARM architecture (e.g. Macs).
The Lambda layers are designed to be built on an x86 architecture (GitHub Actions CI),
so if you are running on ARM architecture you may need to set some additional
environment variables to force the build to run on x86.

Remove all Docker images and build artifacts before running the build command::

    docker builder prune -a
    docker image prune -a
    # Remove the cdk.out directory where build artifacts are stored.
    rm -rf cdk.out

Then, set the environment variable to force Docker to use the x86 architecture::

    DOCKER_DEFAULT_PLATFORM=linux/amd64 cdk synth --context account_name="dev"

