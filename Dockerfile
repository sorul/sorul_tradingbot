FROM ghcr.io/linuxserver/baseimage-kasmvnc:debianbookworm

ENV TITLE=Metatrader5
ENV WINEPREFIX="/config/.wine"

# Update package lists and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Install system dependencies
RUN apt-get install -y \
  python3-pip \
  python3-venv \
  wget \
  curl \
  xvfb

# Create and activate a virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip within the virtual environment
RUN pip install --upgrade pip

# Install Wine64 and its dependencies
RUN dpkg --add-architecture amd64 && apt-get update && \
    apt-get install -y wine64:amd64 libwine:amd64

# Create the Wine directory and configure the Wine64 environment
RUN mkdir -p ${WINEPREFIX} && winecfg

# Clean the apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY /metatrader /metatrader
RUN chmod +x /metatrader/docker_mt5_start.sh
COPY /metatrader/root /

EXPOSE 3000
VOLUME /config
