#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : corpus_dataset.py
# Create date : 2019-01-16 11:16
# Modified date : 2019-02-02 14:55
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import re
import csv
import codecs
import unicodedata
import vocabulary

def _check_is_have_file(file_name):
    return os.path.exists(file_name)

def _filter_pair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def _filter_pairs(pairs, max_length):
    return [pair for pair in pairs if _filter_pair(pair, max_length)]

def _read_vocabulary(datafile, corpus_name):
    print("Reading lines...")
    lines = open(datafile, encoding='utf-8'). read().strip().split('\n')
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    voc = vocabulary.Voc(corpus_name)
    return voc, pairs

def _unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def _get_delimiter(config):
    delimiter = config["delimiter"]
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))
    return delimiter

def _get_object(line, fields):
    values = line.split(" +++$+++ ")
    obj = {}
    for i, field in enumerate(fields):
        obj[field] = values[i]
    return obj

def _load_lines(config):
    lines_file_name = config["lines_file_name"]
    corpus_path = config["corpus_path"]
    lines_file_full_path = "%s/%s" % (corpus_path, lines_file_name)
    fields = config["movie_lines_fields"]

    lines = {}
    f = open(lines_file_full_path, 'r', encoding='iso-8859-1')
    for line in f:
        line_obj = _get_object(line, fields)
        lines[line_obj['lineID']] = line_obj
    f.close()
    return lines

def _cellect_lines(conv_obj, lines):
    # Convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
    line_ids = eval(conv_obj["utteranceIDs"])
    # Reassemble lines
    conv_obj["lines"] = []
    for line_id in line_ids:
        conv_obj["lines"].append(lines[line_id])
    return conv_obj

def _load_conversations(lines, config):
    conversations = []
    corpus_path = config["corpus_path"]
    conversation_file_name = config["conversation_file_name"]
    conversation_file_full_path = "%s/%s" % (corpus_path, conversation_file_name)
    fields = config["movie_conversations_fields"]
    f = open(conversation_file_full_path, 'r', encoding='iso-8859-1')
    for line in f:
        conv_obj = _get_object(line, fields)
        conv_obj = _cellect_lines(conv_obj, lines)
        conversations.append(conv_obj)
    f.close()
    return conversations

def _get_conversations(config):
    lines = {}
    conversations = []

    lines = _load_lines(config)
    print("lines count:", len(lines))
    conversations = _load_conversations(lines, config)
    print("conversations count:", len(conversations))
    return conversations

def _extract_sentence_pairs(conversations):
    pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                pairs.append([inputLine, targetLine])
    return pairs

def _load_formatted_data(config):
    max_length = config["max_length"]
    corpus_name = config["corpus_name"]

    formatted_file_full_path = get_formatted_file_full_path(config)
    print("Start preparing training data ...")
    voc, pairs = _read_vocabulary(formatted_file_full_path, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = _filter_pairs(pairs, max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def _trim_rare_words(voc, pairs, min_count):
    voc.trim(min_count)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def _write_newly_formatted_file(config):
    formatted_file_full_path = get_formatted_file_full_path(config)
    if not _check_is_have_file(formatted_file_full_path):
        delimiter = _get_delimiter(config)
        conversations = _get_conversations(config)
        outputfile = open(formatted_file_full_path, 'w', encoding='utf-8')
        pairs = _extract_sentence_pairs(conversations)
        print("pairs count:", len(pairs))
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        print("\nWriting newly formatted file...")
        for pair in pairs:
            writer.writerow(pair)
    else:
        print("%s already has the formatted file,so we do not write" % formatted_file_full_path)

def load_vocabulary_and_pairs(config):
    _write_newly_formatted_file(config)
    voc, pairs = _load_formatted_data(config)
    pairs = _trim_rare_words(voc, pairs, config["min_count"])
    return voc, pairs

def get_formatted_file_full_path(config):
    formatted_file_name = config["formatted_file_name"]
    corpus_path = config["corpus_path"]
    formatted_file_full_path = "%s/%s" % (corpus_path, formatted_file_name)
    return formatted_file_full_path

def normalize_string(s):
    s = _unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
