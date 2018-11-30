import os
import joblib
from rouge import Rouge
import sys

from util.corpus_accessor import CorpusAccessor
from util.text_processor import wakati
from util.paths import ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH

corpus_accessor = CorpusAccessor()
rouge = Rouge()

def rouge_between_contents_and_synopsis_sentence(ncode):
    """
    小説本文各文にあらすじ文とのもっとも高いROUGEスコアを付与する
    データは文番号をkey、もっとも高いROUGEをvalueとする辞書で保存される
    [1: ROUGE, 2: ROUGE, ... , n: ROUGE]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    file_path = os.path.join(ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH, ncode + '.txt')
    if os.path.isfile(file_path):
        return
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return

    wakati_contents_lines = [wakati(line) for line in contents_lines]
    wakati_synopsis_lines = [wakati(line) for line in synopsis_lines]

    similarity_dict = dict()
    for line_idx, contents_line in enumerate(wakati_contents_lines):
        if line_idx % 50 == 0:
            print('contents progress: {:.1f}%'.format(line_idx / len(contents_lines) * 100))

        if contents_line == '':
            similarity_dict[line_idx] = 0
            continue

        scores = [rouge.get_scores(hyps=contents_line, refs=synopsis_line, avg=False)[0]['rouge-1']['r'] for synopsis_line in wakati_synopsis_lines if not synopsis_line == '']
        if len(scores) == 0:
            similarity_dict[line_idx] = 0
            continue

        max_similarity = max(scores)
        similarity_dict[line_idx] = max_similarity
    print('[INFO] saving data: {}'.format(ncode))
    with open(file_path, 'wb') as f:
        joblib.dump(similarity_dict, f, compress=3)

def construct():
    """
    全小説のデータを構築する
    """
    sys.setrecursionlimit(40000)
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        rouge_between_contents_and_synopsis_sentence(ncode)

if __name__ == '__main__':
    ncode = 'n8276cl'
    rouge_between_contents_and_synopsis_sentence(ncode)
