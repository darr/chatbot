#!/bin/bash
#####################################
## File name : check_code.sh
## Create date : 2018-11-25 15:57
## Modified date : 2019-02-02 14:09
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

source $env_path/py2env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf model.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf chatbot_test.py
pylint --rcfile=pylint.conf chatbot_train.py
pylint --rcfile=pylint.conf corpus_dataset.py
pylint --rcfile=pylint.conf etc.py
pylint --rcfile=pylint.conf vocabulary.py
pip freeze > python3_requiements.txt
deactivate

source $env_path/py3env/bin/activate
pylint --rcfile=pylint.conf main.py
pylint --rcfile=pylint.conf model.py
pylint --rcfile=pylint.conf graph.py
pylint --rcfile=pylint.conf chatbot_test.py
pylint --rcfile=pylint.conf chatbot_train.py
pylint --rcfile=pylint.conf corpus_dataset.py
pylint --rcfile=pylint.conf etc.py
pylint --rcfile=pylint.conf vocabulary.py
pip freeze > python2_requiements.txt
deactivate
