#!/bin/bash
#####################################
## File name : install_pybase.sh
## Create date : 2018-11-25 16:03
## Modified date : 2019-01-27 22:34
## Author : DARREN
## Describe : not set
## Email : lzygzh@126.com
####################################

realpath=$(readlink -f "$0")
export basedir=$(dirname "$realpath")
export filename=$(basename "$realpath")
export PATH=$PATH:$basedir/dlbase
export PATH=$PATH:$basedir/dlproc
#base sh file
. dlbase.sh
#function sh file
. etc.sh

function dlgit_down_pybase(){
    down_url="https://github.com/darr/pybase.git"
    folder="./pybase"
    dlgit_clone_git $down_url $folder
}

function dlgit_rm_pybase(){
    rm -rf ./pybase
}

dlgit_down_pybase

source $env_path/py2env/bin/activate
pybase_path=./pybase
cd $pybase_path
pwd
bash ./set_up.sh
cd ..
deactivate

source $env_path/py3env/bin/activate
pybase_path=./pybase
cd $pybase_path
pwd
bash ./set_up.sh
deactivate
