import os
from rouge import Rouge
import numpy as np
import joblib
import sys

from util.paths import OPT_SENTENCES_DIR_PATH
from data_supplier import similarity_data_supplier
from util.corpus_accessor import CorpusAccessor
from util.text_processor import wakati

corpus_accessor = CorpusAccessor()
rouge = Rouge()

def create_opt_sentences_data_dir_path(short_rate, long_rate, min_sentence_count, max_sentence_count):
    return 'short_' + str(short_rate) \
           + '_long_' + str(long_rate) \
           + '_min_' + str(min_sentence_count) \
           + '_max_' + str(max_sentence_count)

def construct_opt_sentences_data(ncode, save_dir_path, short_rate, long_rate, min_sentence_count, max_sentence_count):
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
    if not len(contents_lines) > max_sentence_count:
        return

    contents_lines = np.array(contents_lines)
    contents_len = len(''.join(contents_lines))

    # 構築するデータ
    opt_data = dict()

    # 参照あらすじ
    ref = wakati(''.join(synopsis_lines))
    # 用いる要約率の閾値
    rate = long_rate if corpus_accessor.is_long(ncode) else short_rate

    similarity_data = similarity_data_supplier.load(ncode)
    high_score_line_indexes = np.argsort(-np.array(list(similarity_data.values())))[:max_sentence_count]

    hyp = contents_lines[high_score_line_indexes[:min_sentence_count]]
    for sentence_index in high_score_line_indexes[min_sentence_count:]:
        if len(''.join(np.append(hyp, contents_lines[sentence_index]))) / contents_len < rate:
            hyp = np.append(hyp, contents_lines[sentence_index])
        else:
            break

    opt_data['opt_sentence_index'] = high_score_line_indexes[:len(hyp)]
    opt_data['threshold'] = similarity_data[high_score_line_indexes[len(hyp) - 1]]

    hyp = wakati(''.join(hyp))
    score = rouge.get_scores(hyps=hyp, refs=ref, avg=False)[0]['rouge-1']
    opt_data['rouge'] = {'f': score['f'], 'p': score['p'], 'r': score['r']}

    file_path = os.path.join(save_dir_path, ncode + '.txt')
    with open(file_path, 'wb') as f:
        joblib.dump(opt_data, f, compress=1)


def construct(short_rate, long_rate, min_sentence_count, max_sentence_count):
    dir_name = create_opt_sentences_data_dir_path(short_rate, long_rate, min_sentence_count, max_sentence_count)
    save_dir_path = os.path.join(OPT_SENTENCES_DIR_PATH, dir_name)
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)
    total = len(corpus_accessor.active_ncodes)
    sys.setrecursionlimit(20000)
    for i, ncode in enumerate(corpus_accessor.active_ncodes):
        print('[INFO] progress: {:.1f}%'.format(i / total * 100))
        print('procesing ncode is ', ncode)
        construct_opt_sentences_data(ncode=ncode,
                                     save_dir_path=save_dir_path,
                                     short_rate=short_rate,
                                     long_rate=long_rate,
                                     min_sentence_count=min_sentence_count,
                                     max_sentence_count=max_sentence_count)


if __name__ == '__main__':
    construct(short_rate=5.1, long_rate=1.3, min_sentence_count=1, max_sentence_count=6)




