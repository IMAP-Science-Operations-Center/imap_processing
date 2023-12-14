Docker Workflow
----------------

This page describes how to build and run a Docker Image Locally and in AWS.

Building and Running a Docker Image Locally
^^^^^^^^^^^^^^^^^

To build the image run the following command from the Dockerfile directory. You might add -t option to tag your image
and --rm to remove intermediate containers after the build is done.

`docker build -t my-image --rm .`

Now we can run our image. Example below runs the image and executes the ois-ingest command with the str option.

`docker run --rm -it my-image:latest ois-ingest --ccsds str`

Building and running a docker image in AWS
""""""""""""""""

In order to push a Docker Image to the ECR, navigate to the AWS Console to retrieve the AWS ECR registry URI:
https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2

Follow the instructions in the link to push to the AWS ECR.
https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

The ECR can be browsed through the AWS Console:
https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2
