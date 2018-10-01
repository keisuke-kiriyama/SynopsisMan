from itertools import chain
import re
import numpy as np
import MeCab
from gensim import corpora, matutils

def wakati(line):
    m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
    wakati = m.parse(line).replace('\n', '')
    return wakati

def cleaning(line):
    line = line.replace('\u3000', '')
    line = line.replace('\n', '')
    line = line.replace(' ', '')
    return line

def tagger():
    return MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

def wakati_tagger():
    return MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')


def convert_serif_marker(contents):
    contents = contents.replace('『', '「')
    contents = contents.replace('』', '」')
    return contents

def get_wakati_lines(lines):
    """
    文のリストを、分かち書きが行われた文のリストに変換
    :param lines: list
    :return: list
    """
    return [wakati(line).split() for line in lines]

def get_morph_info(contents_lines):
    """
    形態素情報のリストを返す
    :param contents_lines: list
    本文の文を要素とするリスト
    :return: list
    """
    contents = ''.join(contents_lines)
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    parsed_contents = tagger.parse(contents)
    if not parsed_contents:
        # 長文でパースに失敗した場合など
        parsed_lines = [tagger.parse(line) for line in contents_lines]
        morph_lines = list(chain.from_iterable([line.split('\n') for line in parsed_lines]))
        return [re.split('[\t,]',morph) for morph in morph_lines if morph not in ['', 'EOS']]
    return [re.split('[\t,]', morph) for morph in parsed_contents.split('\n') if morph not in ['', 'EOS']]

def remove_stop_word(sentence):
    """
    文中の名詞、形容詞、動詞、副詞のリストを返却
    :param sentence: str
    :return: list
    """
    part = ['名詞', '動詞', '形容詞', '副詞']
    m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    morphs = m.parse(sentence).split('\n')
    removed = []
    for morph in morphs:
        splited = re.split('[,\t]', morph)
        if len(splited) < 2: continue
        if splited[1] in part:
            removed.append(splited[0])
    return removed

def get_BoW_vectors(contents_lines, synopsis_lines):
    """
    文のリストから各文のBoWベクトルのリストを返す
    :param contents_lines: list
    本文の各文を要素とするリスト
    :param synopsis_lines: list
    あらすじの各文を要素とするリスト
    :return: ([np.array], [np.array])
    """
    print('creating BoW vectors...')
    removed_contents_lines = [remove_stop_word(cleaning(line)) for line in contents_lines]
    removed_synopsis_lines = [remove_stop_word(cleaning(line)) for line in synopsis_lines]
    all_lines = removed_contents_lines + removed_synopsis_lines
    vocaburaly = corpora.Dictionary(all_lines)
    contents_BoWs = [vocaburaly.doc2bow(line) for line in removed_contents_lines]
    synopsis_BoWs = [vocaburaly.doc2bow(line) for line in removed_synopsis_lines]
    contents_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in contents_BoWs]
    synopsis_vectors = [np.array(matutils.corpus2dense([bow], num_terms=len(vocaburaly)).T[0]) for bow in synopsis_BoWs]
    return contents_vectors, synopsis_vectors

def cos_sim(v1, v2):
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
