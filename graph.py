#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2019-01-16 11:44
# Modified date : 2019-02-02 14:55
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import itertools
import random
import torch
import torch.nn as nn
from torch import optim

import vocabulary
import model
import corpus_dataset

def _get_training_batches(voc, pairs, batch_size, n_iteration):
    training_batches = []
    for i in range(n_iteration):
        lt = [random.choice(pairs) for _ in range(batch_size)]
        batch = _batch2TrainData(voc, lt)
        training_batches.append(batch)
    return training_batches

def _zero_padding(l, fillvalue=vocabulary.PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def _binary_matrix(lt):
    m = []
    for i, seq in enumerate(lt):
        m.append([])
        for token in seq:
            if token == vocabulary.PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def _get_indexes_batch(lt, voc):
    indexes_batch = [_indexes_from_sentence(voc, sentence) for sentence in lt]
    return indexes_batch

def _input_var(batch, voc):
    indexes_batch = _get_indexes_batch(batch, voc)
    padList = _zero_padding(indexes_batch)
    variable = torch.LongTensor(padList)
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    return variable, lengths

def _output_var(batch, voc):
    indexes_batch = _get_indexes_batch(batch, voc)
    padList = _zero_padding(indexes_batch)
    variable = torch.LongTensor(padList)

    max_target_len = max([len(indexes) for indexes in indexes_batch])
    mask = _binary_matrix(padList)
    mask = torch.ByteTensor(mask)

    return variable, mask, max_target_len

def _indexes_from_sentence(voc, sentence):
    #return [voc.word2index[word] for word in sentence.split(' ')] + [vocabulary.EOS_token]
    index_lt = []
    for word in sentence.split(' '):
        i = voc.word2index[word]
        index_lt.append(i)
    index_lt.append(vocabulary.EOS_token)
    return index_lt

def _batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    input_variable, lengths = _input_var(input_batch, voc)
    target_variable, mask, max_target_len = _output_var(output_batch, voc)
    return input_variable, lengths, target_variable, mask, max_target_len

def _maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

class CorpusGraph(nn.Module):
    def __init__(self, config):
        super(CorpusGraph, self).__init__()
        self.model_name = config["model_name"]
        self.save_dir = config["save_dir"]
        self.corpus_name = config["corpus_name"]
        self.encoder_n_layers = config["encoder_n_layers"]
        self.decoder_n_layers = config["decoder_n_layers"]
        self.hidden_size = config["hidden_size"]
        self.checkpoint_iter = config["checkpoint_iter"]
        self.learning_rate = config["learning_rate"]
        self.decoder_learning_ratio = config["decoder_learning_ratio"]
        self.dropout = config["dropout"]
        self.attn_model = config["attn_model"]
        self.device = config["device"]
        self.print_every = config["print_every"]
        self.save_every = config["save_every"]
        self.n_iteration = config["n_iteration"]
        self.batch_size = config["batch_size"]
        self.clip = config["clip"]
        self.max_length = config["max_length"]
        self.teacher_forcing_ratio = config["teacher_forcing_ratio"]
        self.train_load_checkpoint_file = config["train_load_checkpoint_file"]

    def _evaluate(self, voc, sentence, train_model):
        encoder = train_model["encoder"]
        decoder = train_model["decoder"]
        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()

        searcher = model.GreedySearchDecoder(encoder, decoder)
        indexes_batch = [_indexes_from_sentence(voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        input_batch = input_batch.to(self.device)
        lengths = lengths.to(self.device)
        tokens, scores = searcher(input_batch, lengths, self.max_length, self.device)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    def _choose_use_teacher_forcing(self):
        return True if random.random() < self.teacher_forcing_ratio else False

    def _train_step(self, decoder, decoder_input, decoder_hidden, encoder_outputs, target_variable, mask, max_target_len):
        loss = 0
        print_losses = []
        n_totals = 0

        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            if self._choose_use_teacher_forcing():
                decoder_input = target_variable[t].view(1, -1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(self.batch_size)]])
                decoder_input = decoder_input.to(self.device)

            mask_loss, nTotal = _maskNLLLoss(decoder_output, target_variable[t], mask[t], self.device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

        return loss, print_losses, n_totals

    def _train_init(self, input_variable, lengths, target_variable, mask, train_model):
        encoder = train_model["encoder"]
        decoder = train_model["decoder"]
        encoder_optimizer = train_model["encoder_optimizer"]
        decoder_optimizer = train_model["decoder_optimizer"]
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_variable = input_variable.to(self.device)
        lengths = lengths.to(self.device)
        target_variable = target_variable.to(self.device)
        mask = mask.to(self.device)
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
        decoder_input = torch.LongTensor([[vocabulary.SOS_token for _ in range(self.batch_size)]])
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden[:decoder.n_layers]

        return decoder, decoder_input, decoder_hidden, encoder_outputs, target_variable, mask

    def _train_backward(self, loss, train_model):
        encoder = train_model["encoder"]
        decoder = train_model["decoder"]
        encoder_optimizer = train_model["encoder_optimizer"]
        decoder_optimizer = train_model["decoder_optimizer"]
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), self.clip)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), self.clip)
        encoder_optimizer.step()
        decoder_optimizer.step()

    def _train(self, input_variable, lengths, target_variable, mask, max_target_len, train_model):
        decoder, decoder_input, decoder_hidden, encoder_outputs, target_variable, mask = self._train_init(input_variable, lengths, target_variable, mask, train_model)
        loss, print_losses, n_totals = self._train_step(decoder, decoder_input, decoder_hidden, encoder_outputs, target_variable, mask, max_target_len)
        self._train_backward(loss, train_model)
        return sum(print_losses) / n_totals

    def _save_model_dict(self, train_model, iteration, voc, loss):
        model_dict = self._get_model_dict(train_model, iteration, voc, loss)
        checkpoint_file_full_path = self._get_checkpoint_file_full_name()
        torch.save(model_dict, checkpoint_file_full_path)

    def _show_batches(self, batches):
        input_variable, lengths, target_variable, mask, max_target_len = batches
        print("input_variable:", input_variable)
        print("lengths:", lengths)
        print("target_variable:", target_variable)
        print("mask:", mask)
        print("max_target_len:", max_target_len)

    def _show_train_state(self, print_loss, iteration):
        print_loss_avg = print_loss / self.print_every
        print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / self.n_iteration * 100, print_loss_avg))
        print_loss = 0
        return print_loss

    def _get_model_dict(self, train_model, iteration, voc, loss):
        model_dict = {}
        model_dict["en"] = train_model["encoder"].state_dict()
        model_dict["de"] = train_model["decoder"].state_dict()
        model_dict["en_opt"] = train_model["encoder_optimizer"].state_dict()
        model_dict["de_opt"] = train_model["decoder_optimizer"].state_dict()
        model_dict["embedding"] = train_model["embedding"].state_dict()
        model_dict["iteration"] = iteration
        model_dict["loss"] = loss
        model_dict["voc_dict"] = voc.__dict__
        return model_dict

    def _load_checkpoint(self, train_model, voc, checkpoint):
        train_model["encoder"].load_state_dict(checkpoint['en'])
        train_model["decoder"].load_state_dict(checkpoint['de'])
        train_model["encoder_optimizer"].load_state_dict(checkpoint['en_opt'])
        train_model["decoder_optimizer"].load_state_dict(checkpoint['de_opt'])
        train_model["embedding"].load_state_dict(checkpoint['embedding'])
        voc.__dict__ = checkpoint['voc_dict']
        train_model["iteration"] = checkpoint["iteration"]
        return train_model

    def _train_load_checkpoint(self, train_model, voc):
        loadFilename = self._get_checkpoint_file_full_name()
        if os.path.exists(loadFilename) and self.train_load_checkpoint_file:
            checkpoint = torch.load(loadFilename)
            train_model = self._load_checkpoint(train_model, voc, checkpoint)
        return train_model

    def _test_load_checkpoint(self, train_model, voc):
        loadFilename = self._get_checkpoint_file_full_name()
        if os.path.exists(loadFilename) and self.train_load_checkpoint_file:
            checkpoint = torch.load(loadFilename)
            # If loading a model trained on GPU to CPU
            checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            train_model = self._load_checkpoint(train_model, voc, checkpoint)
        return train_model

    def _get_save_directory(self):
        directory = os.path.join(self.save_dir,
                                 self.model_name,
                                 self.corpus_name,
                                 '{}-{}_{}'.format(self.encoder_n_layers,
                                                   self.decoder_n_layers,
                                                   self.hidden_size))
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory

    def _get_checkpoint_file_full_name(self):
        directory = self._get_save_directory()
        checkpoint_file_name = "checkpoint.tar"
        checkpoint_file_full_name = "%s/%s" % (directory, checkpoint_file_name)
        return checkpoint_file_full_name

    def create_train_model(self, voc, status="train"):
        embedding = nn.Embedding(voc.num_words, self.hidden_size)
        encoder = model.EncoderRNN(self.hidden_size, embedding, self.encoder_n_layers, self.dropout)
        encoder = encoder.to(self.device)
        decoder = model.LuongAttnDecoderRNN(self.attn_model, embedding, self.hidden_size, voc.num_words, self.decoder_n_layers, self.dropout)
        decoder = decoder.to(self.device)
        #Ensure dropout layers are in train mode
        encoder.train()
        decoder.train()
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), self.learning_rate*self.decoder_learning_ratio)

        train_model = {}
        train_model["encoder"] = encoder
        train_model["decoder"] = decoder
        train_model["encoder_optimizer"] = encoder_optimizer
        train_model["decoder_optimizer"] = decoder_optimizer
        train_model["embedding"] = embedding
        train_model["iteration"] = 0

        if status == "train":
            train_model = self._train_load_checkpoint(train_model, voc)
        else:
            train_model = self._test_load_checkpoint(train_model, voc)
        return train_model

    def trainIters(self, voc, pairs, train_model):
        training_batches = _get_training_batches(voc, pairs, self.batch_size, self.n_iteration)
        print_loss = 0
        base_iteration = train_model['iteration'] + 1
        start_iteration = 1

        for iteration in range(start_iteration, self.n_iteration + 1):
            training_batch = training_batches[iteration - 1]
            #self._show_batches(training_batch)
            input_variable, lengths, target_variable, mask, max_target_len = training_batch
            loss = self._train(input_variable, lengths, target_variable, mask, max_target_len, train_model)
            print_loss += loss
            cur_iteration = base_iteration + iteration

            if iteration % self.print_every == 0:
                print_loss = self._show_train_state(print_loss, cur_iteration)

            if iteration % self.save_every == 0:
                self._save_model_dict(train_model, cur_iteration, voc, loss)

    def evaluate_input(self, voc, train_model):
        input_sentence = ''
        while(1):
            try:
                input_sentence = input('> ')
                if input_sentence == 'q' or input_sentence == 'quit': break
                input_sentence = corpus_dataset.normalize_string(input_sentence)
                output_words = self._evaluate(voc, input_sentence, train_model)
                output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))
            except KeyError:
                print("Error: Encountered unknown word.")
