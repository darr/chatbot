#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-02-02 13:44
# Modified date : 2019-02-02 13:45
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from chatbot_train import run_train
from chatbot_test import run_test

def run():
    run_train()
    run_test()

run()
