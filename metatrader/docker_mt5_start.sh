#!/bin/bash

# Configuration variables
WINEPREFIX="/config/.wine"
mt5file="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
wine_executable="wine"
mono_url="https://dl.winehq.org/wine/wine-mono/8.0.0/wine-mono-8.0.0-x86.msi"
mt5setup_url="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"

# Function to display a graphical message
show_message() {
    echo $1
}

# Function to check if a dependency is installed
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "$1 is not installed. Please install it to continue."
        exit 1
    fi
}

# Check for necessary dependencies
check_dependency "curl"
check_dependency "$wine_executable"


# Check if the MT_BACKGROUND variable is set to "true"
# To run MT in background without a graphic terminal
if [ "$MT_BACKGROUND" = "true" ]; then
  echo "Starting Xvfb..."
  Xvfb :10 -screen 0 800x600x16 &
  export DISPLAY=:10.0
else
  echo "Xvfb will not start. MT_BACKGROUND variable is not set or is false."
fi

# Install Mono if not present
if [ ! -e "/config/.wine/drive_c/windows/mono" ]; then
    show_message "[1/4] Downloading and installing Mono..."
    curl -o /config/.wine/drive_c/mono.msi $mono_url
    WINEDLLOVERRIDES=mscoree=d $wine_executable msiexec /i /config/.wine/drive_c/mono.msi /qn
    rm /config/.wine/drive_c/mono.msi
    show_message "[1/4] Mono installed."
else
    show_message "[1/4] Mono is already installed."
fi

# Check if MetaTrader 5 is already installed
if [ -e "$mt5file" ]; then
    show_message "[2/4] File $mt5file already exists."
else
    show_message "[2/4] File $mt5file is not installed. Installing..."

    # Set Windows 10 mode in Wine and download and install MT5
    $wine_executable reg add "HKEY_CURRENT_USER\\Software\\Wine" /v Version /t REG_SZ /d "win10" /f
    show_message "[3/4] Downloading MT5 installer..."
    curl -o /config/.wine/drive_c/mt5setup.exe $mt5setup_url
    show_message "[3/4] Installing MetaTrader 5..."
    $wine_executable "/config/.wine/drive_c/mt5setup.exe" "/auto" &
    wait
    rm -f /config/.wine/drive_c/mt5setup.exe
fi

# Recheck if MetaTrader 5 is installed
if [ -e "$mt5file" ]; then
    show_message "[4/4] File $mt5file is installed. Running MT5..."
    chmod -R 777 "/config/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Experts/Advisors"
    $wine_executable "$mt5file" &
else
    show_message "[4/4] File $mt5file is not installed. MT5 cannot be run."
fi
