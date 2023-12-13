# Dockerfile that installs imap_processing and its dependencies

# imap-processing
# ----------
FROM public.ecr.aws/docker/library/python:3.10-slim AS imap-processing
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

# Copy necessary files over
COPY imap_processing $IMAP_PROCESS_DIRECTORY/imap_processing
COPY pyproject.toml $IMAP_PROCESS_DIRECTORY/

# Update this line to copy run_processing.py from the correct location
COPY imap_processing/run_processing.py /opt/imap/

# Install imap_processing and all its dependencies according to pyproject.toml
RUN poetry install

# Make the script executable
RUN chmod +x $IMAP_PROCESS_DIRECTORY/run_processing.py

# Set the working directory to the directory containing the script
WORKDIR $IMAP_PROCESS_DIRECTORY

# Define the entrypoint and default command
ENTRYPOINT ["python", "run_processing.py"]
#CMD ["run_processing.py"]

