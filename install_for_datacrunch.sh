# DataCrunch.io needed these

source ./install/functions.sh
# TODO(usergenic): consider virtualenv and --user install of dependencies
# https://gist.github.com/saurabhshri/46e4069164b87a708b39d947e4527298#gistcomment-2271969

apt-get upgrade python3
which apt-get && apt-get update
which apt && apt install python3.8-venv
download_file \
  'https://bootstrap.pypa.io/get-pip.py' \
  get-pip.py
python3 get-pip.py
python3 -m venv .
source bin/activate
python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

source ./install.sh