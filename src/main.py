import click
import time

from util import paths
import preprocess as p

@click.group()
def cmd():
    pass

@cmd.command()
def check_paths():
    """
    設定されたPATHの確認
    """
    paths.check()

@cmd.command()
def preprocess():
    """
    スクレイピングしたデータを正しく文分割したデータに変換する
    """
    start = time.time()

    p.contents.execute()
    p.meta.execute()

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

def main():
    cmd()

if __name__ == '__main__':
    main()