#!/usr/bin/env bash

# With this we choose the wine profile (folder)  
export WINEPREFIX=/home/pi/.wine
DISPLAY=:10.0

# 30 second delay to give time to the graphical environment
sleep 30

# Run the Wine program in the background and redirect the output
xvfb-run -a /usr/local/bin/wine "/home/pi/.wine/drive_c/Program Files/MetaTrader/terminal.exe"
