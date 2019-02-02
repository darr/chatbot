# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import corpus_dataset
import graph
import etc

def run_test():
    config = etc.config
    voc, pairs = corpus_dataset.load_vocabulary_and_pairs(config)
    g = graph.CorpusGraph(config)
    train_model = g.create_train_model(voc, "test")
    g.evaluate_input(voc, train_model)

