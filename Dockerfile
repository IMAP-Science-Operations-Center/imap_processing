# Dockerfile that installs imap_processing and its dependencies
FROM public.ecr.aws/docker/library/python:3.10-slim
USER root

# Location for Core package installation location.
ENV IMAP_PROCESS_DIRECTORY=/opt/imap
WORKDIR $IMAP_PROCESS_DIRECTORY

# Turn off interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Install git
RUN apt-get update && apt-get install -y git

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# TODO: install imap_processing instead of copying it once a new version is released
# Install imap_processing
#RUN pip install imap_processing
# TODO: delete this section once new version released for imap_processing
RUN pip install git+https://github.com/laspsandoval/imap_processing.git@dev

# Copy over only the necessary scripts
#COPY imap_processing/cli.py $IMAP_PROCESS_DIRECTORY/cli.py

# Create the /mnt/spice directory
RUN mkdir -p /mnt/spice

# Define the entrypoint of the container
ENTRYPOINT ["imap_cli"]
#ENTRYPOINT ["python", "/opt/imap/cli.py"]
