#!/bin/bash

# Installer les dépendances système nécessaires
apt-get update
apt-get install -y build-essential

# Mettre à jour pip, setuptools et wheel
pip install --upgrade pip setuptools wheel