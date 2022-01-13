function clone_git_repo() {
  local fromURL=$1
  local toPath=$2
  mkdir -p $(dirname $toPath)
  if [ -d $toPath ]; then
    echo "Directory $toPath already exists. Skipping."
  else
    git clone $fromURL $toPath
  fi
}

function download_file() {
  local fromURL=$1
  local toPath=$2
  mkdir -p $(dirname $toPath)
  if [ -f $toPath ]; then
    echo "File $toPath already exists. Skipping."
  else
    echo "Downloading $fromURL -> $toPath"
    curl -L -o $toPath $fromURL
  fi
}

function install_package() {
  local package=$1
  local check=$2
  if [ $check ]; then
    echo $package already installed. Skipping.
  else
    if [ $(which brew) ]; then
      echo Installing $package with brew...
      return $(brew install $package)
    elif [ $(which apt-get) ]; then
      echo Installing $package with apt-get...
      return $(sudo apt-get install $package)
    else
      echo "Could not find package manager to install $package. Skipping."
    fi
  fi
}

function sym_link() {
  local destination=$1
  local link=$2
  if [ -e $link ]; then
    echo "Symlink $link already exists. Skipping."
  else
    mkdir -p $(dirname $link)
    echo "Creating symlink $link -> $destination"
    rm -f $link
    ln -s $destination $link
  fi
}

function unzip_file() {
  local file=$1
  local destination=$2
  if [ -d $destination ]; then
    echo "Directory $destination already exists. Skipping."
  else
    echo "Unzipping $file -> $destination"
    mkdir -p $destination
    unzip -q $file -d $destination
  fi
}
