import click

from myrl import utils


@click.command()
@click.argument('csv_path', type=click.Path(exists=True))
@click.option(
    '-o', '--out_path', type=click.Path(exists=False), default=None, show_default=True,
    help='output figure path, e.g.: results/history.png (If skipped the figure will be saved at the same directory of `csv_path`.)')
@click.option('--window', default=10, type=int, show_default=True, help='moving average window')
def visualize(csv_path, out_path, window):
    """
    CSV_PATH: input history csv path, e.g.: results/history.csv
    """
    utils.visualize(csv_path=csv_path, out_path=out_path, window=window)
    return out_path


if __name__ == '__main__':
    out_path = visualize()
    print(out_path)
