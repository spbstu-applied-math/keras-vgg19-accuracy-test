#!/bin/bash
apt update
apt install virtualenv
virtualenv --python="/usr/bin/python3.9" venv
source ./venv/bin/activate
pip install -r requirements.txt
read -sp "Enter access_key_id: " access_key_id
read -sp "Enter secret_access_key: " secret_access_key
dvc remote modify --local spbstu-applied-math-yandex access_key_id "$access_key_id"
dvc remote modify --local spbstu-applied-math-yandex secret_access_key "$secret_access_key"
dvc pull