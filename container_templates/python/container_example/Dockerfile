#Dockerfile is used to create an image

#Set the base image.
FROM python:3.9-slim

RUN apt-get update
RUN apt-get install -y gcc libpq-dev

#Set the work directory in the container
WORKDIR /home

# copy the dependencies file to the working directory
COPY requirements.txt .

# Ensure pip is up to date
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# Define the entrypoint of the container. Passing arguments (known as the COMMAND) when running the
# container will be passed as arguments to the function. Ultimately what is executed is the ENTRYPOINT concatenated
# with the COMMAND. You may add custom options to ENTRYPOINT but note that those options will always be set when
# running in AWS.
ENTRYPOINT ["python","algorithm_example.py"]
