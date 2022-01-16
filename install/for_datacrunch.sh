source ./functions.sh

# DataCrunch.io needed these
apt-get upgrade python3
which apt-get && apt-get update
which apt && apt install python3.8-venv
download_file \
  'https://bootstrap.pypa.io/get-pip.py' \
  get-pip.py
python3 get-pip.py
python3 -m venv .
source bin/activate
