import numpy as np

from util.corpus_accessor import CorpusAccessor
from data_supplier import similarity_data_supplier

corpus_accessor = CorpusAccessor()

def generate_opt_synopsis(ncode):
    """
    正解データのスコアの大きい順に、参照あらすじと近い文字数のあらすじを生成する
    """
    contents_lines = corpus_accessor.get_contents_lines(ncode)
    synopsis_lines = corpus_accessor.get_synopsis_lines(ncode)
    if not contents_lines or not synopsis_lines:
        return

    contents_lines = np.array(contents_lines)
    # 参照あらすじの長さ
    ref_length = len(''.join(synopsis_lines))
    # 最低文数
    min_sentence_count = 1

    # 類似度のデータ
    similarity_data = similarity_data_supplier.load(ncode)
    # 類似度を降順にソートしたインデックス
    sorted_score_line_indexes = np.argsort(-np.array(list(similarity_data.values())))

    hyp = contents_lines[sorted_score_line_indexes[:min_sentence_count]]
    for sentence_index in sorted_score_line_indexes[min_sentence_count:]:
        if len(''.join(np.append(hyp, contents_lines[sentence_index]))) <= ref_length:
            hyp = np.append(hyp, contents_lines[sentence_index])
        else:
            break
    return ''.join(hyp)

if __name__ == '__main__':
    print(generate_opt_synopsis('n0013da'))





