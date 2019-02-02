#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc.py
# Create date : 2019-01-17 22:50
# Modified date : 2019-02-02 14:10
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch

config = {}
config["corpus_name"] = "cornell movie-dialogs corpus"
config["corpus_path"] = "./data/%s" % config["corpus_name"]
config["delimiter"] = '\t'

config["formatted_file_name"] = "formatted_movie_lines.txt"
config["conversation_file_name"] = "movie_conversations.txt"
config["lines_file_name"] = "movie_lines.txt"

config["movie_lines_fields"] = ["lineID", "characterID", "movieID", "character", "text"]
config["movie_conversations_fields"] = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

config["model_name"] = 'cb_model'
config["attn_model"] = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
config["hidden_size"] = 500
config["encoder_n_layers"] = 2
config["decoder_n_layers"] = 2
config["dropout"] = 0.1
config["print_every"] = 20
config["save_every"] = 500
config["n_iteration"] = 1000
config["encoder_n_layers"] = 2
config["decoder_n_layers"] = 2
config["clip"] = 50.0
config["learning_rate"] = 0.0001
config["decoder_learning_ratio"] = 5.0
config["batch_size"] = 64
config["save_dir"] = "./data/save"
config["checkpoint_iter"] = 4000
config["min_count"] = 3  # Minimum word count threshold for trimming
config["max_length"] = 10
config["teacher_forcing_ratio"] = 1.0
config["train_load_checkpoint_file"] = True

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
config["device"] = device
