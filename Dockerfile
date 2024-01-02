# Dockerfile that installs imap_processing and its dependencies

# imap-processing
# ----------
FROM public.ecr.aws/docker/library/python:3.10-slim AS imap-processing
USER root

# Location for Core package installation location.
# This can be used later by images that inherit from this one
ENV IMAP_PROCESS_DIRECTORY=/opt/imap
WORKDIR $IMAP_PROCESS_DIRECTORY

# Turn off interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# TODO: install imap_processing instead of copying it once a new version is released
# Install imap_processing
RUN pip install imap_processing

# Install only the imap_processing script
COPY imap_processing/run_processing.py $IMAP_PROCESS_DIRECTORY/imap_processing/run_processing.py

# Create the /mnt/spice directory
RUN mkdir -p /mnt/spice

# Define the entrypoint of the container
ENTRYPOINT ["python", "/opt/imap/imap_processing/run_processing.py"]
