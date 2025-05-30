FROM nvcr.io/nvidia/pytorch:24.01-py3
ENV DEBIAN_FRONTEND="noninteractive" TERM="xterm"

RUN apt update && apt -y install cmake libgtk2.0-dev  pkg-config

# -----------------------------------
# Install libtorch: https://docs.pytorch.org/cppdocs/installing.html
# -----------------------------------
RUN wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip \
    && unzip libtorch-shared-with-deps-latest.zip -d /usr/local/libtorch \
    && rm libtorch-shared-with-deps-latest.zip

# -------- Add USER -----------
# Source: https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG USERNAME={USERNAME}
ARG USER_UID={USER_UID}
ARG USER_GID={USER_GID}

# Create the user
RUN if ! getent group $USER_GID; then groupadd --gid $USER_GID $USERNAME; fi \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME/