import click
from util import paths

@click.group()
def cmd():
    pass

@cmd.command()
def check_paths():
    """
    設定されたPATHの確認
    """
    paths.check()

def main():
    cmd()

if __name__ == '__main__':
    main()