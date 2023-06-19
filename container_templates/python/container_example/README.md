 # Container Example Code

This the basic code for creating a Docker container for the Level 2 pipeline. 
For questions contact: laura.sandoval@lasp.colorado.edu or contact Laura Sandoval via Slack.

## Python Setup

It is recommended that a virtual environment is created within the project:
1. Make a virtual environment using `python -m venv venv` and activate it 
with `source venv/bin/activate`. Python 3.10 is required.
2. Generate requirements.txt only if additional libraries are required. Otherwise, use the default project requirements.txt.
3. pip install -r requirements.txt

Note: If an error is thrown when installing psycopg2, you may need to read requirements for installing psycopg2: 
https://www.psycopg.org/docs/install.html#build-prerequisites. On Mac simply install Postgres.app, 
which comes with a Postgres server, client, and command line utilities (https://postgresapp.com/)


## Example Algorithm Usage

To run
`python src/algorithm_example.py --help`


## Building and Running a Docker Image Locally

To build the image run the following command from the Dockerfile directory. You might add -t option to tag your image 
and --rm to remove intermediate containers after the build is done.

`docker build -t my-image --rm .`

Now we can run our image using bind mounting. In our example, we will name the container ‘my_app’. 
Adding --rm option will remove the container automatically after the container exits.

`docker run --rm -it \
  -e PROCESSING_DROPBOX=/opt/data/dropbox \
  --volume="$(pwd)/container_example_data:/opt/data" \
  my-image:latest /opt/data/dropbox/libera_input_manifest_20220923t000000.json`

The script `run_container_example.sh` should contain this code as well. You can build and run the example container
with `./run_container_example.sh`. Docker must be running, you must have permissions to execute the script, and
you must be in the same directory as the script. If you get a permission error, 
run `chmod 755 run_container_example.sh`

## Building and running a docker image in AWS

You will be given AWS console logins in the future. Don't worry about this for now.

In order to push a Docker Image to the ECR, navigate to the AWS Console to retrieve the AWS ECR registry URI:
https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2

Follow the instructions in the link to push to the AWS ECR.
https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-push-ecr-image.html

The ECR can be browsed through the AWS Console:
https://us-west-2.console.aws.amazon.com/ecr/repositories?region=us-west-2
