#!/bin/bash
PATH=/home/pi/.poetry/bin:/home/pi/.pyenv/shims:/home/pi/.pyenv/bin:/home/pi/.poetry/bin:/home/pi/.pyenv/shims:/home/pi/.pyenv/bin:/home/pi/.cargo/bin:/home/pi/.poetry/bin:/home/pi/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/games:/usr/games
SHELL=/usr/bin/bash
DISPLAY=:10.0
XAUTHORITY=/home/pi/.Xauthority
touch ~/.bashrc
cd /home/pi/Documents/sorul_tradingbot
source .env
poetry run run_forex
