Docker Workflow
----------------

This page describes how to build and run a Docker Image Locally and in AWS.

Building and Running a Docker Image Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the image run the following command from the directory containing the Dockerfile. You might add -t option to tag your image
and --rm to remove intermediate containers after the build is done.

    `docker build -f Dockerfile.efs -t <image name>:<tag name> --rm .`

Now we can run our image.

    `docker run --rm -it --volume="$(pwd)/imap_processing/efs:/mnt/spice" <image name>:<tag name> --instrument <instrument> --level <level>`

Building and running a docker image in AWS
"""""""""""""""""""""""""""""""""""""""""""

Make certain to set the correct AWS Profile.

    `export AWS_PROFILE=<your profile>`

Login to the ECR.

    `aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <ECR URI>`

Build the Docker image.

    `docker build -t <image name> . --no-cache`

Tag the image and push to the ECR.

    `docker tag <tag> <ECR URI>`

    `docker push <ECR URI>`