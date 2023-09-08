# OS
FROM mcr.microsoft.com/devcontainers/python:3.11
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y firefox-esr libgtk-3-dev

RUN pip install poetry

# use pyenv to install all the other versions as the vscode user
ARG PYTHON_VERSIONS="3.9 3.10 3.11"
USER vscode
ENV PYENV_ROOT="/home/vscode/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN curl https://pyenv.run | bash
RUN $PYENV_ROOT/bin/pyenv install -v ${PYTHON_VERSIONS}

WORKDIR /workspaces/imap_processing
