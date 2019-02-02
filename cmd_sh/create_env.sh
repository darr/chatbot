#!/bin/bash
#####################################
## File name : create_env.sh
## Create date : 2018-11-25 15:54
## Modified date : 2019-02-02 14:01
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
#假设已经安装了vitralenv　并且环境中有Python2 和python3

rm -rf $env_path
mkdir $env_path
cd $env_path
virtualenv -p /usr/bin/python2 py2env
source $env_path/py2env/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torch

deactivate
virtualenv -p /usr/bin/python3 py3env
source $env_path/py3env/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade numpy
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade torch
deactivate
