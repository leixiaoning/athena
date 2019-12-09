# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
""" Text featurizer """

import os
import re
import warnings
from collections import defaultdict
import sentencepiece as spm


class Vocabulary:
    """ Vocabulary

    Interface::
        decode: Convert a list of ids to a sentence, with space inserted
        encode: Convert a sentence to a list of ids, with special tokens added
    """

    def __init__(self, vocab_file):
        """Initialize vocabulary.
        Args:
            vocab_file: Vocabulary file name.
        """
        super().__init__()
        if vocab_file is None or not os.path.exists(vocab_file):
            warnings.warn(
                "[Warning] the vocab {} is not exists, make sure you are "
                "generating it, otherwise you should check it!".format(vocab_file)
            )

        self.stoi = defaultdict(self._default_unk_index)
        self.itos = defaultdict(self._default_unk_symbol)
        self.space, self.unk = "<space>", "<unk>"
        self.unk_index, self.max_index = 0, 0

        with open(vocab_file, "r", encoding="utf-8") as vocab:
            for line in vocab:
                if line.startswith("#"):
                    continue
                word, index = line.split()
                index = int(index)
                self.itos[index] = word
                self.stoi[word] = index
                if word == self.unk:
                    self.unk_index = index
                if index > self.max_index:
                    self.max_index = index

        # special deal with the space maybe used in English datasets
        if self.stoi[self.space] != self.unk_index:
            self.stoi[" "] = self.stoi[self.space]
            self.itos[self.stoi[self.space]] = " "

    def _default_unk_index(self):
        return self.unk_index

    def _default_unk_symbol(self):
        return self.unk

    def __len__(self):
        return self.max_index + 1

    def decode(self, ids):
        """Convert a list of ids to a sentence."""
        return "".join([self.itos[id] for id in ids])

    def encode(self, sentence):
        """Convert a sentence to a list of ids, with special tokens added."""
        return [self.stoi[token.lower()] for token in list(sentence.strip())]

    def __call__(self, inputs):
        if isinstance(inputs, list):
            return self.decode(inputs)
        elif isinstance(inputs, int):
            return self.itos[inputs]
        elif isinstance(inputs, str):
            return self.encode(inputs)
        else:
            raise ValueError("unsupported input")


class TextFeaturizer(Vocabulary):
    """ Text Featurizer """

    def __init__(self, vocab_file):
        super().__init__(vocab_file)
        # Default Unicode for python3
        self.punct_tokens = r"＇｛｝［］＼｜｀～＠＃＄％＾＆＊（）"
        self.punct_tokens += r"＿＋，。、‘’“”《》？：；【】——~！@"
        self.punct_tokens += r"￥%……&（）,.?<>:;\[\]|`\!@#$%^&()+?\"/_-"

    def delete_punct(self, tokens):
        """ delete punctuation tokens """
        return re.sub("[{}]".format(self.punct_tokens), "", tokens)


class SentencePieceFeaturizer:
    """ TODO: docstring """

    def __init__(self, spm_file):
        self.unk_index = 0
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_file)

    def __len__(self):
        return self.sp.GetPieceSize()

    def encode(self, sentence):
        """Convert a sentence to a list of ids by sentence piece model"""
        sentence = sentence.upper()
        return [self.sp.EncodeAsIds(sentence)]

    def decode(self, ids):
        """Conver a list of ids to a sentence"""
        return self.sp.DecodeIds(ids)
