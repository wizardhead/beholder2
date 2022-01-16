# This script will install all of the dependencies to run Beholder locally.

source ./install/functions.sh
# TODO(usergenic): consider virtualenv and --user install of dependencies
# https://gist.github.com/saurabhshri/46e4069164b87a708b39d947e4527298#gistcomment-2271969

if [ $DATACRUNCH ]; then source ./install/for_datacrunch.sh; fi

python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install -r ./install/pip-package-list.txt

mkdir -p ext/data
mkdir -p ext/lib

download_file \
  'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' \
  ext/data/vqgan_imagenet_f16_1024.yaml

download_file \
  'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' \
  ext/data/vqgan_imagenet_f16_1024.ckpt

download_file \
  'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' \
  ext/data/vqgan_imagenet_f16_16384.yaml

download_file \
  'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' \
  ext/data/vqgan_imagenet_f16_16384.ckpt

download_file \
  'https://drive.google.com/u/0/uc?id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_&export=download' \
  ext/data/RIFE_log.zip

unzip_file ext/data/RIFE_log.zip ext/data/RIFE
clone_git_repo 'https://github.com/hzwer/arXiv2020-RIFE' ext/lib/RIFE
sym_link ext/lib/RIFE RIFE
sym_link ../../data/RIFE/train_log ext/lib/RIFE/train_log

clone_git_repo 'https://github.com/openai/CLIP' \
  ext/lib/CLIP
sym_link ext/lib/CLIP CLIP

clone_git_repo 'https://github.com/CompVis/taming-transformers' \
  ext/lib/taming-transformers
sym_link ext/lib/taming-transformers/taming taming
sym_link ext/lib/taming-transformers taming_transformers

download_file \
  'https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1' \
  ext/lib/taming-transformers/taming/modules/autoencoder/lpips/vgg.pth

install_package ffmpeg $(which ffmpeg)
install_package imagemagick $(which convert)
