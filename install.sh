#!/bin/bash
sudo apt-get update
sudo apt-get install -y curl python3-pip portaudio19-dev
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
