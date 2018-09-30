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

    \b
    <ROOT_PATH>: UpNDownNoFrameskip-v4/
    ├── some_dqn/
    │   ├── 20180903_231555/
    │   │   ├── eval_history.csv
    │   │   └── history.csv
    │   └── 20180909_015845/
    │       ├── eval_history.csv
    │       └── history.csv
    ├── another_dqn/
    │   └── 20180903_231708/
    │       ├── eval_history.csv
    │       └── history.csv
    └── the_other_dqn/
        ├── 20180909_193947/
        │   ├── eval_history.csv
        │   └── history.csv
        ├── 20180910_002200/
        │   ├── eval_history.csv
        │   └── history.csv
        ├── 20180916_013155/
        │   ├── eval_history.csv
        │   └── history.csv
        └── 20180917_014429/
            ├── eval_history.csv
            └── history.csv

    You can obtain above results csvs from GCS by:

    \b
    for path in $(gsutil ls "gs://myrl/results/<env_id>/*dqn/*/*history.csv"); do
      gsutil cp $path ${path#gs://myrl/results/};
    done;
    python visualize_multi.py ./<env_id> --csvname history.csv
    python visualize_multi.py ./<env_id> --csvname eval_history.csv
    """
    utils.visualize_multi(root_path=root_path, out_path=out_path, window=window, title=title, csvname=csvname)
    return out_path


if __name__ == '__main__':
    out_path = visualize_multi()
    print(out_path)
