FROM python:3.8.1-slim-buster
# FROM arm32v7/python:3.8.1-slim-buster

WORKDIR /app

# Install dependecies
# RUN apt update \ apt-get install -y curl gcc python3-dev libffi-dev

# Install dependencies
RUN apt update \
  && apt-get install -y \
  python3-venv \
  python3-dev \
  libffi-dev \
  gcc \
  curl \
  libssl-dev

# && apt-get install -y pipx && pipx ensurepath

# variables de entorno de libssl-dev:
RUN export PKG_CONFIG_PATH=/usr/lib/pkgconfig:$PKG_CONFIG_PATH
# RUN export OPENSSL_STATIC=1
# RUN export OPENSSL_LIB_DIR=/usr/lib
# RUN export OPENSSL_INCLUDE_DIR=/usr/include/openssl

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# RUN export PATH="$HOME/.cargo/bin:$PATH"

# Install poetry
# RUN pipx install poetry

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY sorul_tradingbot ./sorul_tradingbot