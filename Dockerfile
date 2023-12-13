# Dockerfile that installs imap_processing and its dependencies

# imap-processing
# ----------
FROM public.ecr.aws/docker/library/python:3.10-slim AS imap-process
USER root

# Location for Core package installation location. This can be used later by images that inherit from this one
ENV IMAP_PROCESS_DIRECTORY=/opt/imap
WORKDIR $IMAP_PROCESS_DIRECTORY

# Turn off interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Install curl to download and install Poetry
RUN apt-get update && apt-get install -y curl

# Create virtual environment and permanently activate it for this image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# This adds not only the venv python executable but also all installed entrypoints to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
RUN pip install --upgrade pip

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# Add poetry to path
ENV PATH="$PATH:/root/.local/bin"

# Copy necessary files over (except for dockerignore-d files)
COPY imap_processing $IMAP_PROCESS_DIRECTORY/imap_processing
COPY pyproject.toml $IMAP_PROCESS_DIRECTORY

# Install imap_processing and all its dependencies according to pyproject.toml
RUN poetry install

# Define the entrypoint of the container. Passing arguments when running the
# container will be passed as arguments to the function
ENTRYPOINT ["python", "run_processing.py"]
