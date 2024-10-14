FROM ghcr.io/linuxserver/baseimage-kasmvnc:debianbookworm

ENV TITLE=Metatrader5
ENV WINEPREFIX="/config/.wine"

# Update package lists and upgrade packages
RUN apt-get update && apt-get upgrade -y

# Instalar dependencias del sistema
RUN apt-get install -y \
  python3-pip \
  python3-venv \
  wget \
  curl \
  xvfb

# Crear y activar un entorno virtual
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Actualizar pip dentro del entorno virtual
RUN pip install --upgrade pip

# Instalar Wine64 y sus dependencias
RUN dpkg --add-architecture amd64 && apt-get update && \
    apt-get install -y wine64:amd64 libwine:amd64

# Crear el directorio de Wine y configurar el entorno de Wine64
RUN mkdir -p ${WINEPREFIX} && winecfg

# Limpiar la cache de apt
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY /metatrader /metatrader
RUN chmod +x /metatrader/docker_mt5_start.sh
COPY /metatrader/root /

EXPOSE 3000
VOLUME /config
