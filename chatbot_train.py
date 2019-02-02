# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import corpus_dataset
import graph
import etc

#   def print_some(lt, n=10):
#       for item in lt[0:n]:
#           print(item)

def run_train():
    config = etc.config
    voc, pairs = corpus_dataset.load_vocabulary_and_pairs(config)
    g = graph.CorpusGraph(config)
    print("Create model")
    train_model = g.create_train_model(voc)
    print("Starting Training!")
    g.trainIters(voc, pairs, train_model)
#    print("Starting evaluate!")
#    g.evaluate_input(voc, train_model)
