#!/usr/bin/env python
# !-*-coding:utf-8 -*-
from c2nl.inputters.vocabulary import Vocabulary, BOS_WORD, EOS_WORD

class AST(object):
    """
    Code containing annotated text, original text, selection label and
    all the extractive spans that can be an answer for the associated question.
    """

    def __init__(self, _id=None):
        self._id = _id
        self._text = None
        self._tokens = []
        self.ast_vocab = None  # required for Copy Attention

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, param: str) -> None:
        self._text = param

    @property
    def tokens(self) -> list:
        return self._tokens

    @tokens.setter
    def tokens(self, param: list) -> None:
        assert isinstance(param, list)
        self._tokens = param
        self.form_src_vocab()

    def form_src_vocab(self) -> None:
        self.ast_vocab = Vocabulary()
        assert self.ast_vocab.remove(BOS_WORD)
        assert self.ast_vocab.remove(EOS_WORD)
        self.ast_vocab.add_tokens(self.tokens)

    def vectorize(self, word_dict, _type='word') -> list:
        if _type == 'word':
            return [word_dict[w] for w in self.tokens]
        elif _type == 'char':
            return [word_dict.word_to_char_ids(w).tolist() for w in self.tokens]
        else:
            assert False