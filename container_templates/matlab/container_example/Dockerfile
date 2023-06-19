# Modified from https://github.com/mathworks-ref-arch/matlab-dockerfile.

# To specify which MATLAB release to install in the container, edit the value of the MATLAB_RELEASE argument.
ARG MATLAB_RELEASE=r2021b

# When you start the build stage, this Dockerfile by default uses the Ubuntu-based matlab-deps image.
# To check the available matlab-deps images, see: https://hub.docker.com/r/mathworks/matlab-deps
FROM mathworks/matlab-deps:${MATLAB_RELEASE}

# Declare the global argument to use at the current build stage
ARG MATLAB_RELEASE

# Install mpm dependencies
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install --no-install-recommends --yes \
        wget \
        unzip \
        ca-certificates && \
    apt-get clean && apt-get autoremove

# Run mpm to install MATLAB in the target location and delete the mpm installation afterwards.
# If mpm fails to install successfully then output the logfile to the terminal, otherwise cleanup.
RUN wget -q https://www.mathworks.com/mpm/glnxa64/mpm && \ 
    chmod +x mpm && \
    ./mpm install \
        --release=${MATLAB_RELEASE} \
        --destination=/opt/matlab \
        --products MATLAB || \
    (echo "MPM Installation Failure. See below for more information:" && cat /tmp/mathworks_root.log && false) && \
    rm -f mpm /tmp/mathworks_root.log && \
    ln -s /opt/matlab/bin/matlab /usr/local/bin/matlab

# Add "matlab" user and grant sudo permission.
RUN adduser --shell /bin/bash --disabled-password --gecos "" matlab && \
    echo "matlab ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/matlab && \
    chmod 0440 /etc/sudoers.d/matlab

ARG LICENSE_SERVER

# Specify the host and port of the machine that serves the network licenses 
ENV MLM_LICENSE_FILE=$LICENSE_SERVER

# Set user and work directory
WORKDIR /home/matlab

# copy the content of the local directory to the working directory
COPY src/ .

ENTRYPOINT ["matlab", "-r"]
