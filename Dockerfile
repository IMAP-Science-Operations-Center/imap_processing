# Dockerfile that installs imap_processing and its dependencies
FROM public.ecr.aws/docker/library/python:3.10-slim
USER root

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# TODO: create this section once imap_processing is released
#RUN pip install imap_processing

# TODO: delete this section once imap_processing is released
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git
RUN pip install git+https://github.com/IMAP-Science-Operations-Center/imap_processing.git@dev

# Create the /mnt/spice directory
RUN mkdir -p /mnt/spice

# Define the entrypoint of the container
ENTRYPOINT ["imap_cli"]
