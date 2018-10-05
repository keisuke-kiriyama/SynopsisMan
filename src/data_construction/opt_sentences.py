import os
import joblib
from rouge import Rouge

from util.paths import OPT_SENTENCES_CONTENTS_DIR_PATH
from data_supplier import similarity_data_supplier
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()
rouge = Rouge()

def construct_opt_sentences_data(ncode, max_sentence_count=30):
    """
    理想的な文を選択する際に選択される文のインデックスを示す
    また、その時のROUGEスコアと、重要度の閾値も保存する
    {
    opt_sentence_index: np.array,
    threshold: float,
    rouge:
        {
        f: float,
        r: float,
        p: float
        }
    }
    """
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return

    similarity_data = similarity_data_supplier.load_from_ncode(ncode)


if __name__ == '__main__':
    ncode = 'n7494cw'
    construct_opt_sentences_data(ncode)



