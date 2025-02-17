# OS
FROM mcr.microsoft.com/devcontainers/python:3.11

# This installs a graphical interface so we can plot the data if needed
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y libgtk-3-dev

# Installs poetry and ipython
RUN pip install "poetry>=1.0.0,<2.0" ipython

WORKDIR /workspaces/imap_processing
