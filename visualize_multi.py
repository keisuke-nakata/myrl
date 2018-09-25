import click

from myrl import utils


@click.command()
@click.argument('root_path', type=click.Path(exists=True))
@click.option(
    '-o', '--out_path', type=click.Path(exists=False), default=None, show_default=True,
    help='output figure path, e.g.: results/history.png (If skipped the figure will be saved at ROOT_PATH.)')
@click.option('--window', default=None, type=int, show_default=True, help='moving average window')
@click.option('--title', default=None, show_default=True, help='title of figure. If None, last directory name of root_path is used.')
@click.option('--csvname', default='history.csv', show_default=True, help='csvname to plot. Use history.csv or eval_history.csv.')
def visualize_multi(root_path, out_path, window, title, csvname):
    """
    ROOT_PATH: input history csv root, e.g.: results/UpNDownNoFrameskip-v4

    Assumed directory:

    <ROOT_PATH>: UpNDownNoFrameskip-v4/\n
    ├── some_dqn/\n
    │   ├── 20180903_231555/\n
    │   │   ├── eval_history.csv\n
    │   │   └── history.csv\n
    │   └── 20180909_015845/\n
    │       ├── eval_history.csv\n
    │       └── history.csv\n
    ├── another_dqn/\n
    │   └── 20180903_231708/\n
    │       ├── eval_history.csv\n
    │       └── history.csv\n
    └── the_other_dqn/\n
        ├── 20180909_193947/\n
        │   ├── eval_history.csv\n
        │   └── history.csv\n
        ├── 20180910_002200/\n
        │   ├── eval_history.csv\n
        │   └── history.csv\n
        ├── 20180916_013155/\n
        │   ├── eval_history.csv\n
        │   └── history.csv\n
        └── 20180917_014429/\n
            ├── eval_history.csv\n
            └── history.csv\n

    You can obtain above results csvs from GCS by:

    for path in $(gsutil ls "gs://myrl/results/UpNDownNoFrameskip-v4/*dqn/*/eval_history.csv"); do\n
      gsutil cp $path ${path#gs://myrl/results/};\n
    done;\n
    """
    utils.visualize_multi(root_path=root_path, out_path=out_path, window=window, title=title, csvname=csvname)
    return out_path


if __name__ == '__main__':
    out_path = visualize_multi()
    print(out_path)
