FROM ghcr.io/linuxserver/baseimage-kasmvnc:debianbookworm

ENV TITLE=Metatrader5
ENV WINEPREFIX="/config/.wine"

# Update package lists. Avoid full apt upgrade here: MetaTrader lives in the
# persistent /config volume and can auto-update independently of the image.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

# Install system dependencies
RUN apt-get install -y --no-install-recommends \
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

# Install Wine's 64-bit and 32-bit runtimes. The base image runs natively on
# the host; Wine/MetaTrader processes are emulated via binfmt/qemu.
RUN dpkg --add-architecture amd64 && \
    dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y \
      wine64:amd64 \
      libwine:amd64 \
      wine32:i386 \
      libwine:i386

# Create the Wine directory. The actual Wine prefix lives on the /config volume
# and is initialized at runtime by docker_mt5_start.sh.
RUN mkdir -p ${WINEPREFIX}

# Clean the apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY metatrader/docker_mt5_start.sh /metatrader/docker_mt5_start.sh
RUN chmod +x /metatrader/docker_mt5_start.sh
COPY metatrader/root /

EXPOSE 3000
VOLUME /config
