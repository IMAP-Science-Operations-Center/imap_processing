# Building and running a docker image locally

To build the image run the following command from the Dockerfile directory.
Note: for non-LASP users you must retrieve your MATLAB licensing information
from your own institute. More information is available here:
https://github.com/mathworks-ref-arch/matlab-dockerfile 

```shell
docker build -t matlab-image --rm . --build-arg \
LICENSE_SERVER=27000@lasp-lmgr.colorado.edu
```

Now we can run our image using bind mounting. 

# For local directory

```shell
docker run --rm -it   -e PROCESSING_DROPBOX=/opt/data/dropbox/   \
--volume="$(pwd)/container_example_data:/opt/data"   matlab-image:latest \
"algorithm_example /opt/data/dropbox/input_manifest_20220923t000000.json"
```

# Building and running a docker image in AWS

You will be given AWS console logins in the future. Don't worry about this for now.

In order to push a Docker Image to the ECR, navigate to the AWS Console to 
retrieve the AWS ECR registry URI:

https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2

Follow the instructions in the link to push to the AWS ECR.

https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

The ECR can be browsed through the AWS Console:

https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2
