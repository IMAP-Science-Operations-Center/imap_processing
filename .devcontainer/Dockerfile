# OS
ARG VARIANT=bullseye
FROM --platform=linux/amd64 mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install -y firefox-esr
RUN sudo apt-get update
RUN sudo apt-get install -y libgtk-3-dev
RUN sudo apt-get install -y python3 python3-pip

RUN pip install poetry

WORKDIR /workspaces/imap_processing
