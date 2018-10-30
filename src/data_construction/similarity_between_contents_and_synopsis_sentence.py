import os
import joblib

from util.corpus_accessor import CorpusAccessor
from util.text_processor import get_BoW_vectors, cos_sim
from util.paths import SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH

corpus_accessor = CorpusAccessor()

def most_similarity_between_contents_and_synopsis_sentence(ncode):
    """
    小説本文各文にあらすじ文とのもっとも高い類似度を付与する
    類似度はBoWベクトルのcos類似度により与えられる
    データは文番号をkey、もっとも高い類似度をvalueとする辞書で保存される
    [1: 類似度, 2: 類似度, ... , n: 類似度]
    """
    print('[PROCESS NCODE]: {}'.format(ncode))
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return

    contents_BoW_vectors, synopsis_BoW_vectors = get_BoW_vectors(contents_lines, synopsis_lines)

    similarity_dict = dict()
    for line_idx, contents_BoW_vector in enumerate(contents_BoW_vectors):
        if line_idx % 50 == 0:
            print('contents progress: {:.1f}%'.format(line_idx / len(contents_lines) * 100))

        similarities = [cos_sim(contents_BoW_vector, synopsis_BoW_vector) for synopsis_BoW_vector in synopsis_BoW_vectors]
        max_similarity = max(similarities)
        similarity_dict[line_idx] = max_similarity
    file_path = os.path.join(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH, ncode + '.txt')
    print('[INFO] saving data: {}'.format(ncode))
    with open(file_path, 'wb') as f:
        joblib.dump(similarity_dict, f, compress=3)



def construct():
    """
    全小説のデータを構築する
    """
    for i, ncode in enumerate(corpus_accessor.ncodes):
        print('[INFO] num of constructed data: {}'.format(i))
        most_similarity_between_contents_and_synopsis_sentence(ncode)

if __name__ == '__main__':
    ncode = 'n8276cl'
    most_similarity_between_contents_and_synopsis_sentence(ncode)
