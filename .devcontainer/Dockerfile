# OS
FROM mcr.microsoft.com/devcontainers/python:3.11

# This installs a graphical interface so we can plot the data if needed
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y libgtk-3-dev

# Installs poetry and ipython
RUN pip install "poetry>=2.0,<3.0" ipython

# Create the poetry virtualenv at /workspaces/imap_processing/.venv so VS Code can find it
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /workspaces/imap_processing
