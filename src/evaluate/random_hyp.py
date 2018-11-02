import numpy as np

from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def generate_random_synopsis(ncode):
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

    random_line_indexes = np.random.permutation(np.arange(len(contents_lines)))

    hyp = contents_lines[random_line_indexes[:min_sentence_count]]
    for sentence_index in random_line_indexes[min_sentence_count:]:
        if len(''.join(np.append(hyp, contents_lines[sentence_index]))) <= ref_length:
            hyp = np.append(hyp, contents_lines[sentence_index])
        else:
            break
    return ''.join(hyp)

if __name__ == '__main__':
    print(generate_random_synopsis('n0013da'))





