import os
import json
from itertools import chain

from src.util import paths
from src.util.text_processor import get_wakati_lines

class DataAccessor:

    def __init__(self):
        # PATHS
        self.contents_data_dir_path = paths.PREPROCESSED_CONTENTS_DATA_DIR_PATH
        self.meta_data_dir_path = paths.PREPROCESSED_META_DATA_DIR_PATH
        self.contents_file_paths = [os.path.join(self.contents_data_dir_path, file_name) for file_name in os.listdir(self.contents_data_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.meta_data_dir_path, file_name) for file_name in os.listdir(self.meta_data_dir_path) if not file_name == '.DS_Store']

    def ncode_from_file_path(self, file_path):
        """
        ファイルpathからncodeを返却する
        """
        return file_path.split('/')[-1].split('.')[0]

    def create_contents_file_path(self, ncode):
        """
        ncodeから本文の情報が格納されているファイルパスを作成する
        """
        dir_path = self.contents_data_dir_path
        return os.path.join(dir_path, ncode + '.json')

    def create_meta_file_path(self, ncode):
        """
        ncodeからメタ情報が格納されているファイルパスを作成する
        """
        dir_path = self.meta_data_dir_path
        return os.path.join(dir_path, ncode+'_meta.json')

    def load(self, file_path):
        json_file = open(file_path, 'r')
        data = json.load(json_file)
        json_file.close()
        return data

    def get_contents_lines(self, ncode):
        """
        本文全文のリストを返却
        :param contents_file_path: str
        :return: list
        """
        contents_file_path = self.create_contents_file_path(ncode=ncode)
        if not contents_file_path in self.contents_file_paths:
            print('nothing ncode')
            return
        return list(chain.from_iterable(self.load(contents_file_path)['contents']))

    def get_synopsis_lines(self, ncode):
        """
        あらすじの文のリストを返却
        :param synopsis_file_path: str
        :return: list
        """
        meta_file_path = self.create_meta_file_path(ncode=ncode)
        if not meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        return self.load(meta_file_path)['story']

    def get_wakati_contents_lines(self, ncode):
        """
        本文の各文を分かち書きしたリストを取得
        :param ncode: str
        :return: list
        """
        contents_lines = self.get_contents_lines(ncode=ncode)
        wakati_contents_lines = get_wakati_lines(contents_lines)
        return wakati_contents_lines

    def get_wakati_synopsis_lines(self, ncode):
        """
        あらすじの各分を分かち書きしたリストを取得
        :param ncode:
        :return:
        """
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        wakati_synopsis_lines = get_wakati_lines(synopsis_lines)
        return wakati_synopsis_lines

    def remove_error_line_indexes_from_contents_lines(self, contents_lines, error_line_indexes):
        """
        本文からエラーがでた行を削除する
        :param contents_lines: list
        :param error_line_indexes: list
        :return: list
        """
        if error_line_indexes.size == 0:
            return contents_lines
        for error_line_index in sorted(error_line_indexes, reverse=True):
            del contents_lines[int(error_line_index)]
        return contents_lines


